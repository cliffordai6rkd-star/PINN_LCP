import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model.tau_f_sequence import TauFSequenceRegressor
from train.base_trainer import BaseTrainer
from train.nomalizer import Normalizer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SampleIndexDataset(torch.utils.data.Dataset):
    """Return lightweight sample indices instead of materializing CPU windows."""

    def __init__(self, sample_indices):
        self.sample_indices = [int(index) for index in sample_indices]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        return {"sample_idx": self.sample_indices[index]}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a causal LSTM/GRU regressor for tau_f."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/tau_f_sequence.yaml"),
        help="Path to the training config.",
    )
    return parser.parse_args()


class TauFTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        loss_config = config.get("loss") or {}
        joint_weights = loss_config.get("joint_weights")
        self.joint_weights = (
            torch.as_tensor(joint_weights, dtype=torch.float32)
            if joint_weights is not None
            else None
        )
        self.tensor_cache = None
        self.valid_raw_indices_device = None
        self.episode_starts_device = None

    def build_dataset(self):
        # Keep the optional LeRobot dependency out of model-only imports and tests.
        from data_process.dataloader import PINNDataset

        return PINNDataset(self.config, compute_normalizer=False)

    def fit_dataset_normalizer(self, train_dataset):
        if isinstance(train_dataset, torch.utils.data.Subset):
            sample_indices = train_dataset.indices
        else:
            sample_indices = range(len(self.dataset))
        training_frames = self.dataset.covered_raw_indices(sample_indices)
        log.info(
            "fit normalizer from training windows only: raw_frames=%d",
            len(training_frames),
        )
        self._build_device_cache(training_frames)

    def setup(self):
        super().setup()
        self.loader = self._index_loader(self.loader.dataset, shuffle=True)
        if self.val_loader is not None:
            self.val_loader = self._index_loader(
                self.val_loader.dataset,
                shuffle=False,
            )

    def _index_loader(self, source_dataset, shuffle):
        if isinstance(source_dataset, torch.utils.data.Subset):
            sample_indices = source_dataset.indices
        else:
            sample_indices = range(len(self.dataset))

        return torch.utils.data.DataLoader(
            SampleIndexDataset(sample_indices),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def _build_device_cache(self, training_frames):
        model_config = self.config.get("model") or {}
        active_inputs = list(model_config.get("inputs") or ["q", "dq", "ddq", "tau"])
        cache_keys = list(
            dict.fromkeys(
                active_inputs
                + [str(model_config.get("target_key", "tau_f"))]
                + list(self.dataset.normalize_lowdim_keys)
            )
        )
        missing_keys = [
            key for key in cache_keys if key not in self.dataset.lowdim_keys
        ]
        if missing_keys:
            raise KeyError(
                f"Missing dataloader.lowdim_keys for cached inputs: {missing_keys}"
            )

        cpu_cache = self._load_lowdim_columns(cache_keys)
        stats = self._normalizer_stats(cpu_cache, training_frames)
        if stats:
            self.dataset.set_normalizer(Normalizer(stats))

        cache_device = torch.device(self.device)
        self.tensor_cache = {}
        for key, value in cpu_cache.items():
            value = value.to(device=cache_device, dtype=torch.float32)
            if key in stats:
                value = self._normalize_cached_tensor(value, stats[key])
            self.tensor_cache[key] = value.contiguous()

        valid_raw_indices = self.dataset.valid_indices
        episode_starts = [
            self.dataset.raw_idx_to_episode_start[raw_idx]
            for raw_idx in valid_raw_indices
        ]
        self.valid_raw_indices_device = torch.as_tensor(
            valid_raw_indices,
            device=cache_device,
            dtype=torch.long,
        )
        self.episode_starts_device = torch.as_tensor(
            episode_starts,
            device=cache_device,
            dtype=torch.long,
        )

        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.tensor_cache.values()
        )
        log.info(
            "low-dimensional sequence cache ready: device=%s keys=%s memory=%.2f MiB",
            cache_device,
            cache_keys,
            cache_bytes / (1024**2),
        )

    def _load_lowdim_columns(self, cache_keys):
        dataset_keys = [self.dataset.lowdim_keys[key] for key in cache_keys]
        hf_dataset = self.dataset.stats_dataset

        try:
            formatted_dataset = hf_dataset.with_format(
                "torch",
                columns=dataset_keys,
                output_all_columns=False,
            )
            columns = formatted_dataset[:]
            cache = {
                key: self._column_to_tensor(columns[dataset_key])
                for key, dataset_key in zip(cache_keys, dataset_keys)
            }
        except (AttributeError, TypeError, ValueError) as exc:
            log.warning(
                "bulk tensor loading unavailable, falling back to one-pass rows: %s",
                exc,
            )
            buffers = {key: [] for key in cache_keys}
            for raw_idx in range(len(hf_dataset)):
                frame = hf_dataset[raw_idx]
                for key, dataset_key in zip(cache_keys, dataset_keys):
                    buffers[key].append(torch.as_tensor(frame[dataset_key]))
            cache = {key: torch.stack(values, dim=0) for key, values in buffers.items()}

        expected_length = len(hf_dataset)
        for key, value in cache.items():
            if value.shape[0] != expected_length:
                raise ValueError(
                    f"Cached field {key!r} has {value.shape[0]} frames, "
                    f"expected {expected_length}."
                )
        return cache

    @staticmethod
    def _column_to_tensor(column):
        if torch.is_tensor(column):
            return column
        if isinstance(column, list) and column and torch.is_tensor(column[0]):
            return torch.stack(column, dim=0)
        return torch.as_tensor(column)

    def _normalizer_stats(self, cpu_cache, training_frames):
        if self.dataset.normalize_mode is None:
            return {}

        training_indices = torch.as_tensor(training_frames, dtype=torch.long)
        stats = {}
        for key in self.dataset.normalize_lowdim_keys:
            if key not in cpu_cache:
                continue
            values = (
                cpu_cache[key]
                .to(dtype=torch.float32)
                .index_select(
                    0,
                    training_indices,
                )
            )
            stats[key] = {
                "mean": values.mean(dim=0),
                "std": values.std(dim=0),
                "min": values.min(dim=0).values,
                "max": values.max(dim=0).values,
                "q01": torch.quantile(values, 0.01, dim=0),
                "q99": torch.quantile(values, 0.99, dim=0),
            }
        return stats

    def _normalize_cached_tensor(self, value, stats):
        eps = float(self.dataset.normalizer.eps)
        mode = self.dataset.normalize_mode
        stats = {
            name: statistic.to(device=value.device, dtype=value.dtype)
            for name, statistic in stats.items()
        }
        if mode == "gaussian":
            return (value - stats["mean"]) / (stats["std"] + eps)
        if mode == "limit":
            return 2 * (value - stats["min"]) / (stats["max"] - stats["min"] + eps) - 1
        if mode == "quantile":
            normalized = (
                2 * (value - stats["q01"]) / (stats["q99"] - stats["q01"] + eps) - 1
            )
            return torch.clamp(normalized, -1.0, 1.0)
        raise ValueError(f"unknown normalize mode: {mode}")

    def _cached_batch(self, sample_indices):
        if self.tensor_cache is None:
            raise RuntimeError("device tensor cache has not been initialized")

        sample_indices = sample_indices.to(
            device=self.valid_raw_indices_device.device,
            dtype=torch.long,
        )
        raw_indices = self.valid_raw_indices_device.index_select(0, sample_indices)
        episode_starts = self.episode_starts_device.index_select(0, sample_indices)
        offsets = torch.arange(
            -self.dataset.horizon + 1,
            1,
            device=raw_indices.device,
        )
        window_indices = raw_indices[:, None] + offsets[None, :]
        window_indices = torch.maximum(window_indices, episode_starts[:, None])

        model = self.config.get("model") or {}
        active_inputs = list(model.get("inputs") or ["q", "dq", "ddq", "tau"])
        target_key = str(model.get("target_key", "tau_f"))
        batch = {key: self.tensor_cache[key][window_indices] for key in active_inputs}
        batch[target_key] = self.tensor_cache[target_key].index_select(0, raw_indices)
        return batch

    def build_model(self):
        return TauFSequenceRegressor(self.config)

    def compute_loss(self, batch):
        if "sample_idx" in batch:
            batch = self._cached_batch(batch["sample_idx"])
        out = self.model(batch)
        prediction = out["tau_f_pred"]
        target = out.get("tau_f_target")
        if target is None:
            raise KeyError("Batch is missing the configured tau_f supervision target.")

        squared_error = F.mse_loss(prediction, target, reduction="none")
        if self.joint_weights is not None:
            weights = self.joint_weights.to(
                device=squared_error.device,
                dtype=squared_error.dtype,
            )
            if weights.numel() != squared_error.shape[-1]:
                raise ValueError(
                    f"loss.joint_weights has {weights.numel()} entries, "
                    f"expected {squared_error.shape[-1]}."
                )
            squared_error = squared_error * weights

        loss = squared_error.mean()
        out["loss_dict"] = {
            "mse": loss.detach(),
            "mae": F.l1_loss(prediction, target).detach(),
        }
        return loss, out


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    log.info("tau_f sequence train config: %s", config)
    trainer = TauFTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
