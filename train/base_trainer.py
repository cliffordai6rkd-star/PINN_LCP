import argparse
import copy
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ModelEMA:
    """Maintain an inference model with exponentially averaged parameters."""

    def __init__(self, model, decay=0.999, update_after_step=0, update_every=1):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        if update_after_step < 0:
            raise ValueError("EMA update_after_step must be non-negative.")
        if update_every < 1:
            raise ValueError("EMA update_every must be at least 1.")

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        if step % self.update_every != 0:
            return

        source_parameters = dict(model.named_parameters())
        for name, averaged in self.model.named_parameters():
            source = source_parameters[name].detach()
            if step <= self.update_after_step:
                averaged.copy_(source)
            else:
                averaged.lerp_(source, 1.0 - self.decay)

        source_buffers = dict(model.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(source_buffers[name].detach())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/dataset_test_cfg.yaml"),
    )
    return parser.parse_args()

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.train_config = config.get("train") or {}

        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = self.train_config.get("device", default_device)
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was configured, but torch.cuda.is_available() "
                "returned False. Check the NVIDIA driver or set train.device=cpu "
                "for a smoke test."
            )
        if not (str(self.device).startswith("cuda") or str(self.device) == "cpu"):
            raise ValueError(
                f"train.device must be a CUDA device or cpu, got {self.device!r}."
            )

        self.val_ratio = float(self.train_config.get("val_ratio", 0.1))
        self.split_mode = self.train_config.get("split_mode", "episode")
        configured_val_episodes = self.train_config.get("val_episode_indices")
        if configured_val_episodes is None:
            self.val_episode_indices = None
        else:
            if not isinstance(configured_val_episodes, (list, tuple)):
                raise ValueError(
                    "train.val_episode_indices must be a list of episode indices"
                )
            if not configured_val_episodes:
                raise ValueError("train.val_episode_indices must not be empty")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in configured_val_episodes
            ):
                raise ValueError(
                    "train.val_episode_indices must contain integer episode indices"
                )
            if len(set(configured_val_episodes)) != len(configured_val_episodes):
                raise ValueError(
                    "train.val_episode_indices must not contain duplicates"
                )
            self.val_episode_indices = tuple(configured_val_episodes)
            if self.split_mode != "episode":
                raise ValueError(
                    "train.val_episode_indices requires train.split_mode=episode"
                )
        self.seed = int(self.train_config.get("seed", 42))
        self.deterministic = bool(
            self.train_config.get("deterministic", True)
        )
        self.cudnn_benchmark = bool(
            self.train_config.get("cudnn_benchmark", False)
        )
        if self.deterministic and self.cudnn_benchmark:
            raise ValueError(
                "train.deterministic=true is incompatible with "
                "train.cudnn_benchmark=true"
            )
        self.val_loader = None
        purged_kfold_config = self.train_config.get("purged_kfold") or {}
        if not isinstance(purged_kfold_config, dict):
            raise ValueError("train.purged_kfold must be a mapping or null")
        self.purged_kfold_num_folds = int(
            purged_kfold_config.get("num_folds", 3)
        )
        self.purged_kfold_fold_index = int(
            purged_kfold_config.get("fold_index", 0)
        )
        if self.split_mode == "purged_kfold":
            if self.purged_kfold_num_folds < 2:
                raise ValueError("train.purged_kfold.num_folds must be at least 2")
            if not (
                0
                <= self.purged_kfold_fold_index
                < self.purged_kfold_num_folds
            ):
                raise ValueError(
                    "train.purged_kfold.fold_index must be in "
                    f"[0, {self.purged_kfold_num_folds - 1}]"
                )
        self.current_split_metadata = None

        self.batch_size = int(self.train_config.get("batch_size", 64))
        self.num_workers = int(self.train_config.get("num_workers", 4))
        self.pin_memory = bool(self.train_config.get("pin_memory", False))
        self.non_blocking_transfer = bool(
            self.train_config.get("non_blocking_transfer", self.pin_memory)
        )
        self.persistent_workers = bool(
            self.train_config.get("persistent_workers", False)
        )
        self.prefetch_factor = int(self.train_config.get("prefetch_factor", 2))
        if self.prefetch_factor < 1:
            raise ValueError("train.prefetch_factor must be at least 1")
        if self.persistent_workers and self.num_workers <= 0:
            raise ValueError(
                "train.persistent_workers requires train.num_workers > 0"
            )
        self.lr = float(self.train_config.get("lr", 1e-4))
        self.weight_decay = float(self.train_config.get("weight_decay", 1e-4))
        self.num_epochs = int(self.train_config.get("num_epochs", 20))
        configured_max_steps = self.train_config.get("max_train_steps")
        self.max_train_steps = (
            None if configured_max_steps is None else int(configured_max_steps)
        )
        self.checkpoint_every_steps = int(
            self.train_config.get("checkpoint_every_steps", 0)
        )
        self.save_latest_checkpoint = bool(
            self.train_config.get("save_latest_checkpoint", True)
        )
        self.step_based_training = (
            self.max_train_steps is not None or self.checkpoint_every_steps > 0
        )
        self.max_optimizer_steps = None
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("train.max_train_steps must be positive")
        if self.checkpoint_every_steps < 0:
            raise ValueError("train.checkpoint_every_steps must be non-negative")
        gradient_clip_norm = self.train_config.get(
            "gradient_clip_norm",
            self.train_config.get("max_grad_norm"),
        )
        self.gradient_clip_norm = (
            float(gradient_clip_norm) if gradient_clip_norm is not None else None
        )
        self.monitor_key = self.train_config.get("monitor_key", "val_loss")
        self.scheduler_monitor_key = self.train_config.get(
            "scheduler_monitor_key",
            self.monitor_key,
        )
        self.early_stopping_monitor_key = self.train_config.get(
            "early_stopping_monitor_key",
            self.monitor_key,
        )
        configured_top_k = self.train_config.get("top_k", 3)
        self.top_k = 3 if configured_top_k is None else int(configured_top_k)
        if self.top_k < 1:
            raise ValueError("train.top_k must be at least 1")
        self.best_checkpoints = []

        early_stopping = self.train_config.get("early_stopping") or {}
        self.early_stopping_enabled = bool(early_stopping.get("enabled", False))
        self.early_stopping_patience = int(early_stopping.get("patience", 20))
        self.early_stopping_warmup = int(early_stopping.get("warmup_epochs", 0))
        self.early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
        self.early_stopping_best = float("inf")
        self.early_stopping_bad_epochs = 0

        self.dataset = None
        self.loader = None
        self.model = None
        self.optimizer = None

        self.output_dir = Path(
            self.train_config.get("output_dir", "outputs/torque_world_model")
        )
        self.ckpt_dir = self.output_dir / "checkpoints"

        self.global_step = 0
        self._last_step_checkpoint = None
        self.loss_history = []
        self.last_train_epoch_metrics = {}
        self.last_train_eval_epoch_metrics = {}
        self.last_val_epoch_metrics = {}

        train_eval_config = self.train_config.get("train_eval") or {}
        self.train_eval_enabled = bool(train_eval_config.get("enabled", False))

        self.last_summary = None

        scheduler_config = self.train_config.get("scheduler")
        if scheduler_config is not None and not isinstance(
            scheduler_config,
            dict,
        ):
            raise ValueError("train.scheduler must be a mapping or null")
        self.scheduler_config = (
            None if scheduler_config is None else dict(scheduler_config)
        )
        if self.scheduler_config is not None:
            scheduler_name = str(
                self.scheduler_config.get("name", "none")
            ).strip().lower()
            aliases = {
                "cosine_annealing": "cosine",
                "cosineannealing": "cosine",
            }
            self.scheduler_config["name"] = aliases.get(
                scheduler_name,
                scheduler_name,
            )
        self.scheduler = None

        ema_config = self.train_config.get("ema") or {}
        self.ema_enabled = bool(ema_config.get("enabled", False))
        self.ema_decay = float(ema_config.get("decay", 0.999))
        self.ema_update_after_step = int(ema_config.get("update_after_step", 0))
        self.ema_update_every = int(ema_config.get("update_every", 1))
        self.ema_use_for_validation = bool(
            ema_config.get("use_for_validation", True)
        )
        self.ema = None

        self.wandb_config = self.train_config.get("wandb") or {}
        self.wandb_enabled = bool(self.wandb_config.get("enabled", False))
        self.wandb_log_every_steps = int(
            self.wandb_config.get("log_every_steps", 10)
        )
        if self.wandb_log_every_steps < 1:
            raise ValueError("wandb.log_every_steps must be at least 1.")
        self.wandb_run = None

        amp_config = self.train_config.get("amp") or {}
        if not isinstance(amp_config, dict):
            raise ValueError("train.amp must be a mapping or null")
        self.amp_enabled = bool(amp_config.get("enabled", False))
        amp_dtype = str(amp_config.get("dtype", "bfloat16")).lower()
        amp_dtypes = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if amp_dtype not in amp_dtypes:
            raise ValueError(
                "train.amp.dtype must be bfloat16/bf16 or float16/fp16"
            )
        self.amp_dtype = amp_dtypes[amp_dtype]
        if self.amp_enabled and not str(self.device).startswith("cuda"):
            raise ValueError("train.amp.enabled requires a CUDA device")
        self.amp_scaler = None
        if self.amp_enabled and self.amp_dtype == torch.float16:
            self.amp_scaler = torch.amp.GradScaler("cuda")

    def set_seed(self):
        import random
        import numpy as np
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.deterministic

    def batch_to_device(self, batch):
        new_batch = {}

        for k, v in batch.items():
            if torch.is_tensor(v):
                # log.info(f"{v} is tensor")
                new_batch[k] = v.to(
                    self.device,
                    non_blocking=self.non_blocking_transfer,
                )
            else:
                log.warning(f"{v} is not tensor")
                new_batch[k] = v

        return new_batch

    def autocast_context(self, enabled=None):
        """Return the configured CUDA autocast context for model execution."""

        if enabled is False or not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )

    def _dataloader_kwargs(self, *, shuffle):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs
    
    def build_dataset(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError

    def fit_dataset_normalizer(self, train_dataset):
        """Hook for datasets that must fit normalization after the split."""
        return None

    def build_scheduler(self):
        if self.scheduler_config is None:
            return None

        name = self.scheduler_config.get("name", "none")

        if name == "none":
            return None

        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(self.scheduler_config.get("factor", 0.5)),
                patience=int(self.scheduler_config.get("patience", 20)),
                min_lr=float(self.scheduler_config.get("min_lr", 1e-6)),
            )
        
        if name == "cosine":
            t_max = int(
                self.scheduler_config.get(
                    "T_max",
                    getattr(self, "max_optimizer_steps", self.num_epochs),
                )
            )
            eta_min = float(self.scheduler_config.get("eta_min", 1e-6))
            if t_max <= 0:
                raise ValueError("cosine scheduler T_max must be positive")
            if eta_min < 0.0:
                raise ValueError("cosine scheduler eta_min must be non-negative")
            if eta_min > self.lr:
                raise ValueError(
                    "cosine scheduler eta_min must not exceed train.lr"
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        raise ValueError(f"Unsupported scheduler: {name}")

    def split_dataset_by_episode(self):
        episodes = list(self.dataset.dataset.meta.episodes)
        if len(episodes) < 2:
            raise ValueError(
                "episode split requires at least two episodes; use "
                "split_mode=purged_temporal for a single time-series episode."
            )

        episode_indices = [
            int(episode.get("episode_index", position))
            for position, episode in enumerate(episodes)
        ]
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode metadata contains duplicate episode_index values")

        if self.val_episode_indices is not None:
            val_episode_indices = set(self.val_episode_indices)
            unknown = sorted(val_episode_indices - set(episode_indices))
            if unknown:
                raise ValueError(
                    "train.val_episode_indices contains unknown episode indices: "
                    f"{unknown}; available={sorted(episode_indices)}"
                )
            if len(val_episode_indices) >= len(episodes):
                raise ValueError(
                    "train.val_episode_indices must leave at least one training episode"
                )
        else:
            num_val_episodes = int(len(episodes) * self.val_ratio)
            num_val_episodes = max(1, num_val_episodes)
            num_val_episodes = min(num_val_episodes, len(episodes) - 1)

            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(episodes), generator=generator).tolist()
            val_episode_indices = {
                episode_indices[position] for position in perm[:num_val_episodes]
            }

        episode_start_to_idx = {
            int(ep["dataset_from_index"]): episode_indices[position]
            for position, ep in enumerate(episodes)
        }

        train_indices = []
        val_indices = []

        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            episode_idx = episode_start_to_idx[episode_start]

            if episode_idx in val_episode_indices:
                val_indices.append(sample_idx)
            else:
                train_indices.append(sample_idx)

        log.info(
            f"episode split: "
            f"train_episodes={len(episodes) - len(val_episode_indices)} "
            f"val_episodes={len(val_episode_indices)} "
            f"train_samples={len(train_indices)} "
            f"val_samples={len(val_indices)}"
        )
        log.info(f"val episode indices: {sorted(val_episode_indices)}")

        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)

        return train_dataset, val_dataset

    def split_dataset_purged_temporal(self):
        """Chronological split with disjoint raw-frame windows on both sides."""
        horizon = int(getattr(self.dataset, "horizon", 1))
        train_indices = []
        val_indices = []
        purged_samples = 0
        samples_by_episode_start = {}
        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            samples_by_episode_start.setdefault(episode_start, []).append(
                (sample_idx, raw_idx)
            )

        for episode in self.dataset.dataset.meta.episodes:
            episode_start = int(episode["dataset_from_index"])
            episode_samples = samples_by_episode_start.get(episode_start, [])
            if len(episode_samples) < 2:
                continue

            num_val = max(1, int(len(episode_samples) * self.val_ratio))
            num_val = min(num_val, len(episode_samples) - 1)
            first_val_position = len(episode_samples) - num_val
            first_val_raw_idx = episode_samples[first_val_position][1]
            last_train_raw_idx = first_val_raw_idx - horizon

            for sample_idx, raw_idx in episode_samples:
                if raw_idx <= last_train_raw_idx:
                    train_indices.append(sample_idx)
                elif raw_idx >= first_val_raw_idx:
                    val_indices.append(sample_idx)
                else:
                    purged_samples += 1

        if not train_indices or not val_indices:
            raise ValueError(
                "purged_temporal split produced an empty train or validation set; "
                "reduce dataloader.horizon or train.val_ratio, or collect a longer episode."
            )

        log.info(
            "purged temporal split: train_samples=%d val_samples=%d "
            "purged_samples=%d horizon=%d",
            len(train_indices),
            len(val_indices),
            purged_samples,
            horizon,
        )
        return (
            torch.utils.data.Subset(self.dataset, train_indices),
            torch.utils.data.Subset(self.dataset, val_indices),
        )

    def split_dataset_purged_kfold(self, fold_index=None, num_folds=None):
        """Hold out one contiguous block per episode without window overlap."""
        horizon = int(getattr(self.dataset, "horizon", 1))
        num_folds = int(
            self.purged_kfold_num_folds if num_folds is None else num_folds
        )
        fold_index = int(
            self.purged_kfold_fold_index if fold_index is None else fold_index
        )
        if num_folds < 2:
            raise ValueError("purged K-fold requires at least two folds")
        if not 0 <= fold_index < num_folds:
            raise ValueError(
                f"fold_index must be in [0, {num_folds - 1}], got {fold_index}"
            )

        samples_by_episode_start = {}
        for sample_idx, raw_idx in enumerate(self.dataset.valid_indices):
            episode_start = self.dataset.raw_idx_to_episode_start[raw_idx]
            samples_by_episode_start.setdefault(int(episode_start), []).append(
                (sample_idx, int(raw_idx))
            )

        train_indices = []
        val_indices = []
        purged_samples = 0
        episode_metadata = []

        for episode_position, episode in enumerate(
            self.dataset.dataset.meta.episodes
        ):
            episode_start = int(episode["dataset_from_index"])
            episode_samples = sorted(
                samples_by_episode_start.get(episode_start, []),
                key=lambda item: item[1],
            )
            sample_count = len(episode_samples)
            if sample_count < num_folds:
                raise ValueError(
                    "purged_kfold requires every episode to contain at least "
                    f"num_folds valid windows; episode {episode_position} has "
                    f"{sample_count}, num_folds={num_folds}."
                )

            val_start_position = sample_count * fold_index // num_folds
            val_end_position = sample_count * (fold_index + 1) // num_folds
            validation_samples = episode_samples[
                val_start_position:val_end_position
            ]
            first_val_raw_idx = validation_samples[0][1]
            last_val_raw_idx = validation_samples[-1][1]
            first_val_frame = max(
                episode_start,
                first_val_raw_idx - horizon + 1,
            )

            episode_train_count = 0
            episode_purged_count = 0
            for sample_position, (sample_idx, raw_idx) in enumerate(
                episode_samples
            ):
                if val_start_position <= sample_position < val_end_position:
                    val_indices.append(sample_idx)
                    continue

                first_window_frame = max(
                    episode_start,
                    raw_idx - horizon + 1,
                )
                if (
                    raw_idx < first_val_frame
                    or first_window_frame > last_val_raw_idx
                ):
                    train_indices.append(sample_idx)
                    episode_train_count += 1
                else:
                    purged_samples += 1
                    episode_purged_count += 1

            if episode_train_count == 0:
                raise ValueError(
                    "purged_kfold removed all training windows from episode "
                    f"{episode_position} in fold {fold_index}; reduce "
                    "dataloader.horizon or train.purged_kfold.num_folds, or "
                    "collect a longer episode."
                )

            episode_metadata.append(
                {
                    "episode_index": int(
                        episode.get("episode_index", episode_position)
                    ),
                    "train_samples": episode_train_count,
                    "val_samples": len(validation_samples),
                    "purged_samples": episode_purged_count,
                    "val_target_start": first_val_raw_idx,
                    "val_target_end": last_val_raw_idx,
                }
            )

        if not train_indices or not val_indices:
            raise ValueError(
                "purged_kfold split produced an empty train or validation set"
            )

        self.current_split_metadata = {
            "mode": "purged_kfold",
            "fold_index": fold_index,
            "num_folds": num_folds,
            "horizon": horizon,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "purged_samples": purged_samples,
            "episodes": episode_metadata,
        }
        log.info(
            "purged K-fold split: fold=%d/%d train_samples=%d val_samples=%d "
            "purged_samples=%d horizon=%d episodes=%d",
            fold_index + 1,
            num_folds,
            len(train_indices),
            len(val_indices),
            purged_samples,
            horizon,
            len(episode_metadata),
        )
        return (
            torch.utils.data.Subset(self.dataset, train_indices),
            torch.utils.data.Subset(self.dataset, val_indices),
        )

    def setup(self):
        self.set_seed()
        self.dataset = self.build_dataset()

        if (
            self.val_ratio > 0
            or self.split_mode == "purged_kfold"
            or self.val_episode_indices is not None
        ):
            if self.split_mode == "episode":
                train_dataset, val_dataset = self.split_dataset_by_episode()
            elif self.split_mode == "sample":
                if int(getattr(self.dataset, "horizon", 1)) > 1:
                    log.warning(
                        "sample split leaks overlapping temporal windows across train and "
                        "validation; use split_mode=episode, purged_temporal, or "
                        "purged_kfold."
                    )
                val_size = int(len(self.dataset) * self.val_ratio)
                train_size = len(self.dataset) - val_size
        
                generator = torch.Generator().manual_seed(self.seed)
                train_dataset, val_dataset = torch.utils.data.random_split(
                    self.dataset,
                    [train_size, val_size],
                    generator=generator,
                )
            elif self.split_mode == "purged_temporal":
                train_dataset, val_dataset = self.split_dataset_purged_temporal()
            elif self.split_mode == "purged_kfold":
                train_dataset, val_dataset = self.split_dataset_purged_kfold()
            else:
                raise ValueError(f"unknown split_mode: {self.split_mode}")
        else:
            train_dataset = self.dataset
            val_dataset = None

        self.fit_dataset_normalizer(train_dataset)

        self.loader = torch.utils.data.DataLoader(
            train_dataset,
            **self._dataloader_kwargs(shuffle=True),
        )

        if val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                **self._dataloader_kwargs(shuffle=False),
            )

        if self.step_based_training:
            if len(self.loader) < 1:
                raise ValueError("step-based training requires a non-empty train loader")
            self.max_optimizer_steps = (
                self.max_train_steps
                if self.max_train_steps is not None
                else self.num_epochs * len(self.loader)
            )
            if self.checkpoint_every_steps <= 0:
                self.checkpoint_every_steps = self.max_optimizer_steps

        self.model = self.build_model().to(self.device)
        if self.ema_enabled:
            self.ema = ModelEMA(
                self.model,
                decay=self.ema_decay,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler()
        self.setup_wandb()

    def setup_wandb(self):
        if not self.wandb_enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed. "
                "Run setup.sh or install the project again."
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": self.wandb_config.get("project", "lcp-pinn"),
            "name": self.wandb_config.get("name"),
            "entity": self.wandb_config.get("entity"),
            "group": self.wandb_config.get("group"),
            "tags": self.wandb_config.get("tags"),
            "notes": self.wandb_config.get("notes"),
            "mode": self.wandb_config.get("mode", "online"),
            "config": self.config,
            "dir": str(self.output_dir),
            "resume": self.wandb_config.get("resume"),
        }
        self.wandb_run = wandb.init(
            **{key: value for key, value in init_kwargs.items() if value is not None}
        )

    def log_wandb(self, metrics, step=None):
        if self.wandb_run is not None:
            metrics = dict(metrics)
            if step is not None:
                metrics.setdefault("trainer/global_step", step)
            self.wandb_run.log(metrics)

    def finish_wandb(self, exit_code=0):
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
            self.wandb_run = None

    @staticmethod
    def _batch_size(batch):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    @staticmethod
    def _accumulate_scalar_metrics(metric_sums, metric_counts, metrics, weight):
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            if not isinstance(value, (int, float)):
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * weight
            metric_counts[key] = metric_counts.get(key, 0) + weight

    @staticmethod
    def _average_scalar_metrics(metric_sums, metric_counts):
        return {
            key: total / max(metric_counts[key], 1)
            for key, total in metric_sums.items()
        }

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        num_samples = 0
        metric_sums = {}
        metric_counts = {}
        pbar = tqdm(
            self.loader,
            desc=f"train epoch {epoch}",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            if (
                self.step_based_training
                and self.max_optimizer_steps is not None
                and self.global_step >= self.max_optimizer_steps
            ):
                break
            batch = self.batch_to_device(batch)

            with self.autocast_context():
                loss, out = self.compute_loss(batch)
            batch_size = self._batch_size(batch)

            self.optimizer.zero_grad(set_to_none=True)
            if self.amp_scaler is not None:
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )
            if self.amp_scaler is not None:
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.global_step += 1
            if self.ema is not None:
                self.ema.update(self.model, self.global_step)
            if (
                self.step_based_training
                and self.scheduler is not None
                and self.scheduler_config.get("name", "none")
                != "reduce_on_plateau"
            ):
                self.scheduler.step()

            total_loss += loss.item() * batch_size
            num_samples += batch_size
            self._accumulate_scalar_metrics(
                metric_sums,
                metric_counts,
                out.get("loss_dict") or {},
                batch_size,
            )

            if self.global_step % self.wandb_log_every_steps == 0:
                step_metrics = {
                    "train/loss": loss.detach().item(),
                    "train/epoch": epoch,
                }
                for key, value in (out.get("loss_dict") or {}).items():
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    if isinstance(value, (int, float)):
                        step_metrics[f"train/{key}"] = value
                self.log_wandb(step_metrics, step=self.global_step)

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "step": self.global_step,
            })

        self.last_train_epoch_metrics = self._average_scalar_metrics(
            metric_sums,
            metric_counts,
        )
        return total_loss / max(num_samples, 1)
    
    @torch.no_grad()
    def evaluate_loader(self, loader, epoch, description):
        training_model = self.model
        if self.ema is not None and self.ema_use_for_validation:
            self.model = self.ema.model
        self.model.eval()

        total_loss = 0.0
        num_samples = 0
        metric_sums = {}
        metric_counts = {}

        pbar = tqdm(
            loader,
            desc=f"{description} epoch {epoch}",
            unit="batch",
            leave=False,
        )

        try:
            for batch in pbar:
                batch = self.batch_to_device(batch)
                with self.autocast_context():
                    loss, out = self.compute_loss(batch)
                batch_size = self._batch_size(batch)

                total_loss += loss.item() * batch_size
                num_samples += batch_size
                self._accumulate_scalar_metrics(
                    metric_sums,
                    metric_counts,
                    out.get("loss_dict") or {},
                    batch_size,
                )

                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                })
        finally:
            self.model = training_model

        metrics = self._average_scalar_metrics(
            metric_sums,
            metric_counts,
        )
        loss = total_loss / max(num_samples, 1)
        return loss, metrics

    def validate_one_epoch(self, epoch):
        if self.val_loader is None:
            self.last_val_epoch_metrics = {}
            return None

        val_loss, self.last_val_epoch_metrics = self.evaluate_loader(
            self.val_loader,
            epoch,
            "val",
        )
        return val_loss

    def evaluate_train_one_epoch(self, epoch):
        if not self.train_eval_enabled:
            self.last_train_eval_epoch_metrics = {}
            return None

        train_eval_loss, self.last_train_eval_epoch_metrics = self.evaluate_loader(
            self.loader,
            epoch,
            "train-eval",
        )
        return train_eval_loss
    
    def train(self):
        try:
            return self._train_impl()
        except BaseException:
            self.finish_wandb(exit_code=1)
            raise

    def _train_impl(self):

        self.setup()

        epoch = 0
        while True:
            if self.step_based_training:
                if self.max_optimizer_steps is None or self.global_step >= self.max_optimizer_steps:
                    break
            elif epoch >= self.num_epochs:
                break

            step_before_epoch = self.global_step
            avg_loss = self.train_one_epoch(epoch)
            if self.global_step == step_before_epoch:
                log.warning("training loader produced no optimizer steps; stopping")
                break
            train_eval_loss = self.evaluate_train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            metrics = {
                "avg_loss": avg_loss,
                "train_loss_online": avg_loss,
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
            }
            metrics.update(
                {
                    f"train_{key}": value
                    for key, value in self.last_train_epoch_metrics.items()
                }
            )
            if train_eval_loss is not None:
                metrics.update(
                    {
                        f"train_eval_{key}": value
                        for key, value in self.last_train_eval_epoch_metrics.items()
                    }
                )
            if val_loss is not None:
                metrics.update(
                    {
                        f"val_{key}": value
                        for key, value in self.last_val_epoch_metrics.items()
                    }
                )

            if self.scheduler is not None:
                name = self.scheduler_config.get("name", "none")
                if name == "reduce_on_plateau":
                    monitor_loss = metrics.get(self.scheduler_monitor_key)
                    if monitor_loss is None:
                        monitor_loss = val_loss if val_loss is not None else avg_loss
                    self.scheduler.step(monitor_loss)
                elif not self.step_based_training:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            val_mae_nm = metrics.get("val_mae_nm")
            physical_metric_text = ""
            if val_mae_nm is not None:
                joint_mae_nm = [
                    value
                    for key, value in sorted(
                        self.last_val_epoch_metrics.items(),
                        key=lambda item: item[0],
                    )
                    if key.startswith("mae_nm_j")
                ]
                joint_metric_text = ",".join(
                    f"{value:.4f}" for value in joint_mae_nm
                )
                physical_metric_text = (
                    f" val_mae_nm={val_mae_nm:.6f}"
                    f" val_joint_mae_nm=[{joint_metric_text}]"
                )
            train_eval_text = (
                f" train_eval_loss={train_eval_loss:.6f}"
                if train_eval_loss is not None
                else ""
            )
            if val_loss is None:
                log.info(
                    "epoch=%d train_loss_online=%.6f%s lr=%.2e",
                    epoch,
                    avg_loss,
                    train_eval_text,
                    current_lr,
                )
            else:
                log.info(
                    "epoch=%d train_loss_online=%.6f%s val_loss=%.6f%s lr=%.2e",
                    epoch,
                    avg_loss,
                    train_eval_text,
                    val_loss,
                    physical_metric_text,
                    current_lr,
                )

            history_item = {
                "epoch": epoch,
                "global_step": self.global_step,
                **metrics,
            }
            self.loss_history.append(history_item)

            self.save_loss_plot()

            epoch_metrics = {
                "epoch": epoch,
                "train/epoch_loss": avg_loss,
                "train/loss_online": avg_loss,
                "train/learning_rate": current_lr,
            }
            epoch_metrics.update(
                {
                    f"train/epoch_{key}": value
                    for key, value in self.last_train_epoch_metrics.items()
                }
            )
            if train_eval_loss is not None:
                epoch_metrics["train_eval/loss"] = train_eval_loss
                epoch_metrics.update(
                    {
                        f"train_eval/{key}": value
                        for key, value in self.last_train_eval_epoch_metrics.items()
                    }
                )
            if val_loss is not None:
                epoch_metrics["val/loss"] = val_loss
                epoch_metrics.update(
                    {
                        f"val/{key}": value
                        for key, value in self.last_val_epoch_metrics.items()
                    }
                )
            self.log_wandb(epoch_metrics, step=self.global_step)

            if self.step_based_training:
                if self.global_step % self.checkpoint_every_steps == 0:
                    self.save_step_checkpoint(epoch, metrics)
            else:
                self.save_topk_checkpoint(epoch, metrics)

            if self.should_stop_early(epoch, metrics):
                log.info(
                    "early stopping at epoch=%d: monitor=%s best=%.6f patience=%d",
                    epoch,
                    self.early_stopping_monitor_key,
                    self.early_stopping_best,
                    self.early_stopping_patience,
                )
                break

            epoch += 1

        if (
            self.step_based_training
            and self.global_step > 0
            and self._last_step_checkpoint != self.global_step
        ):
            final_metrics = self.loss_history[-1] if self.loss_history else {}
            self.save_step_checkpoint(epoch, final_metrics)

        self.last_summary = {
            "num_epochs": len(self.loss_history),
            "global_step": self.global_step,
            "last_loss": self.loss_history[-1]["avg_loss"] if self.loss_history else None,
            "last_val_loss": self.loss_history[-1]["val_loss"] if self.loss_history else None,
            "last_train_eval_loss": (
                self.loss_history[-1].get("train_eval_loss")
                if self.loss_history
                else None
            ),
            "last_train_metrics": dict(self.last_train_epoch_metrics),
            "last_train_eval_metrics": dict(self.last_train_eval_epoch_metrics),
            "last_val_metrics": dict(self.last_val_epoch_metrics),
            "split": self.current_split_metadata,
            "monitor_key": self.monitor_key,
            "scheduler_monitor_key": self.scheduler_monitor_key,
            "early_stopping_monitor_key": self.early_stopping_monitor_key,
            "top_k": self.top_k,
            "best_checkpoints": self.best_checkpoints,
            "max_train_steps": self.max_optimizer_steps,
            "max_optimizer_steps": self.max_optimizer_steps,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "step_based_training": self.step_based_training,
            "stopped_early": (
                self.step_based_training
                and self.max_optimizer_steps is not None
                and self.global_step < self.max_optimizer_steps
            ),
            "output_dir": self.output_dir,
            "ckpt_dir": self.ckpt_dir,
            "ema_enabled": self.ema_enabled,
            "wandb_run_id": (
                self.wandb_run.id if self.wandb_run is not None else None
            ),
        }

        self.finish_wandb(exit_code=0)
        return self.last_summary

    def should_stop_early(self, epoch, metrics):
        if not self.early_stopping_enabled:
            return False

        score = metrics.get(self.early_stopping_monitor_key)
        if score is None:
            return False

        if score < self.early_stopping_best - self.early_stopping_min_delta:
            self.early_stopping_best = score
            self.early_stopping_bad_epochs = 0
            return False

        if epoch + 1 < self.early_stopping_warmup:
            return False

        self.early_stopping_bad_epochs += 1
        return self.early_stopping_bad_epochs >= self.early_stopping_patience
        
    
    def save_loss_plot(self):
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)

        steps = [item["global_step"] for item in self.loss_history]
        epochs = [item["epoch"] for item in self.loss_history]
        losses = [item["avg_loss"] for item in self.loss_history]
        train_eval_losses = [
            item.get("train_eval_loss") for item in self.loss_history
        ]
        val_losses = [item.get("val_loss") for item in self.loss_history]
        has_train_eval = any(loss is not None for loss in train_eval_losses)
        has_val = any(loss is not None for loss in val_losses)

        plt.figure()
        plt.plot(epochs, losses, label="train online")
        if has_train_eval:
            plt.plot(epochs, train_eval_losses, label="train eval")
        plt.xlabel("epoch")
        plt.ylabel("avg_loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / "loss_epoch.png"
        # plt.savefig(path)
        plt.close()

        if has_val:
            plt.figure()
            plt.plot(epochs, val_losses, label="val", color="tab:orange")
            plt.xlabel("epoch")
            plt.ylabel("val_loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            path = self.output_dir / "val_loss_epoch.png"
            # plt.savefig(path)
            plt.close()

        plt.figure()
        plt.plot(steps, losses, label="train online")
        if has_train_eval:
            plt.plot(steps, train_eval_losses, label="train eval")
        if has_val:
            plt.plot(steps, val_losses, label="val")
        plt.xlabel("steps")
        plt.ylabel("avg_loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / "loss_steps.png"
        # plt.savefig(path)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, losses, label="train online")
        if has_train_eval:
            axes[0].plot(epochs, train_eval_losses, label="train eval")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("avg_loss")
        axes[0].set_title("Train Loss / Epoch")
        axes[0].legend()
        axes[0].grid(True)

        if has_val:
            axes[1].plot(epochs, val_losses, label="val", color="tab:orange")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val_loss")
        axes[1].set_title("Val Loss / Epoch")
        if has_val:
            axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(steps, losses, label="train online")
        if has_train_eval:
            axes[2].plot(steps, train_eval_losses, label="train eval")
        if has_val:
            axes[2].plot(steps, val_losses, label="val")
        axes[2].set_xlabel("steps")
        axes[2].set_ylabel("loss")
        axes[2].set_title("Loss / Steps")
        axes[2].legend()
        axes[2].grid(True)

        fig.tight_layout()
        path = self.output_dir / "loss_summary.png"
        fig.savefig(path)
        plt.close(fig)

        # log.info(f"saved loss plot: {path}")

    def save_step_checkpoint(self, epoch, metrics):
        """Save a step checkpoint and retain only the newest ``top_k`` files."""

        if self.global_step <= 0:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        step = int(self.global_step)
        metrics = dict(metrics or {})
        dataloader_config = self.config.get("dataloader") or {}
        normalizer = getattr(self.dataset, "normalizer", None)
        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        checkpoint = {
            "checkpoint_type": "optimizer_step",
            "epoch": int(epoch),
            "global_step": step,
            "metrics": metrics,
            "avg_loss": metrics.get("avg_loss"),
            "val_loss": metrics.get("val_loss"),
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": getattr(normalizer, "stats", {}),
                "eps": getattr(normalizer, "eps", 1.0e-6),
                "normalize_mode": dataloader_config.get("normalize_mode"),
                "normalize_lowdim_keys": dataloader_config.get(
                    "normalize_lowdim_keys"
                ),
            },
        }
        path = self.ckpt_dir / f"step_{step:08d}.pt"
        torch.save(checkpoint, path)

        self.best_checkpoints.append(
            {
                "score": None,
                "epoch": int(epoch),
                "global_step": step,
                "path": path,
                "metrics": metrics,
            }
        )
        self.best_checkpoints.sort(key=lambda item: item["global_step"])
        while len(self.best_checkpoints) > self.top_k:
            removed = self.best_checkpoints.pop(0)
            removed_path = Path(removed["path"])
            if removed_path.exists():
                removed_path.unlink()

        if self.save_latest_checkpoint:
            torch.save(checkpoint, self.ckpt_dir / "latest.pt")
        self._last_step_checkpoint = step
        log.info(
            "saved optimizer-step checkpoint: step=%d path=%s retained=%s%s",
            step,
            path,
            [item["global_step"] for item in self.best_checkpoints],
            " latest.pt" if self.save_latest_checkpoint else "",
        )
        
    def save_checkpoint(self, epoch, avg_loss, val_loss=None):

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        ckpt = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_loss": val_loss,
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": self.dataset.normalizer.stats,
                "eps": self.dataset.normalizer.eps,
                "normalize_mode": self.config["dataloader"].get("normalize_mode"),
                "normalize_lowdim_keys": self.config["dataloader"].get("normalize_lowdim_keys"),
            }
        }

        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)
        # log.info(f"saved checkpoint: {path}")
        
    def save_topk_checkpoint(self, epoch, metrics, checkpoint_label=None):
        score = metrics.get(self.monitor_key)

        if score is None:
            log.warning(f"monitor_key={self.monitor_key} is None, skip checkpoint")
            return

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        label = (
            f"epoch_{epoch:03d}"
            if checkpoint_label is None
            else str(checkpoint_label)
        )
        path = self.ckpt_dir / f"{label}_{self.monitor_key}_{score:.6f}.pt"

        model_state = (
            self.ema.model.state_dict() if self.ema is not None
            else self.model.state_dict()
        )
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "monitor_key": self.monitor_key,
            "monitor_score": score,
            "metrics": metrics,
            "model": model_state,
            "model_raw": self.model.state_dict() if self.ema is not None else None,
            "ema": self._ema_checkpoint_metadata(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config,
            "dataloader_filters": getattr(self.dataset, "filter_config", {}),
            "sample_rate_hz": getattr(self.dataset, "sample_rate_hz", None),
            "derived_target_config": getattr(
                self, "derived_target_config", {"enabled": False}
            ),
            "normalizer": {
                "stats": self.dataset.normalizer.stats,
                "eps": self.dataset.normalizer.eps,
                "normalize_mode": self.config["dataloader"].get("normalize_mode"),
                "normalize_lowdim_keys": self.config["dataloader"].get("normalize_lowdim_keys"),
            }
        }

        torch.save(ckpt, path)
        # log.info(f"saved checkpoint: {path}")

        self.best_checkpoints.append({
            "score": score,
            "path": path,
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": dict(metrics),
        })

        self.best_checkpoints.sort(key=lambda item: item["score"])

        while len(self.best_checkpoints) > self.top_k:
            removed = self.best_checkpoints.pop(-1)
            removed_path = removed["path"]
            if removed_path.exists():
                removed_path.unlink()
                # log.info(f"removed checkpoint: {removed_path}")

    def _ema_checkpoint_metadata(self):
        if self.ema is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "decay": self.ema.decay,
            "update_after_step": self.ema.update_after_step,
            "update_every": self.ema.update_every,
            "global_step": self.global_step,
        }

    def format_summary(self, summary):
        lines = []
        lines.append("Training finished")
        lines.append(f"num_epochs: {summary['num_epochs']}")
        lines.append(f"global_step: {summary['global_step']}")
        if summary.get("max_train_steps") is not None:
            lines.append(
                f"max_train_steps: {summary['max_train_steps']}"
            )
            lines.append(f"stopped_early: {summary['stopped_early']}")
        lines.append(f"last_loss: {summary['last_loss']}")
        lines.append(f"last_train_eval_loss: {summary.get('last_train_eval_loss')}")
        lines.append(f"last_val_loss: {summary['last_val_loss']}")
        if summary.get("last_val_metrics", {}).get("mae_nm") is not None:
            lines.append(
                f"last_val_mae_nm: {summary['last_val_metrics']['mae_nm']}"
            )
            joint_mae = [
                summary["last_val_metrics"][key]
                for key in sorted(summary["last_val_metrics"])
                if key.startswith("mae_nm_j")
            ]
            lines.append(f"last_val_joint_mae_nm: {joint_mae}")
        lines.append(f"monitor_key: {summary['monitor_key']}")
        lines.append(f"output_dir: {summary['output_dir']}")
        lines.append(f"ckpt_dir: {summary['ckpt_dir']}")
        lines.append(f"ema_enabled: {summary['ema_enabled']}")
        lines.append(f"wandb_run_id: {summary['wandb_run_id']}")
        lines.append("best_checkpoints:")
    
        for item in summary["best_checkpoints"]:
            step_text = (
                f" step={item['global_step']}"
                if "global_step" in item
                else ""
            )
            score = item.get("score")
            score_text = f" score={score:.6f}" if score is not None else ""
            lines.append(
                f"  epoch={item['epoch']}{step_text}{score_text} "
                f"path={item['path']}"
            )
    
        return "\n".join(lines)
