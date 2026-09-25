"""Standalone deterministic WM training, regression loss, and CPU smoke entry."""

from __future__ import annotations

import argparse
import copy
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.deterministic_world_model import DeterministicRobotStateWorldModel, _streams
from train.base_trainer import BaseTrainer, ModelEMA


log = logging.getLogger(__name__)


class DeterministicWorldModelLoss:
    """Per-stream normalized MSE and supervised contact CE, reduced per sample."""

    def __init__(self, config):
        model = config.get("model") or {}
        loss = config.get("loss") or {}
        contact = config.get("contact_gate") or {}
        self.predicted_state_streams = _streams(model.get("outputs", ("q", "tau")), "outputs")
        self.contact_state_count = int(model.get("contact_state_count", 3))
        weights = loss.get("stream_weights") or {}
        self.stream_weights = {key: float(weights.get(key, 1.0)) for key in self.predicted_state_streams}
        self.contact_weight = float(loss.get("contact_weight", 0.1))
        if any(not math.isfinite(v) or v < 0 for v in (*self.stream_weights.values(), self.contact_weight)):
            raise ValueError("loss weights must be finite and non-negative")
        self.use_importance_weight = bool(loss.get("use_importance_weight", True))
        configured = contact.get("class_weights", loss.get("contact_class_weights", "auto"))
        self.contact_class_weights_is_auto = isinstance(configured, str) and configured.lower() == "auto"
        self.contact_class_weights = None
        if configured is not None and not self.contact_class_weights_is_auto:
            self.set_contact_class_weights(configured)

    def set_contact_class_weights(self, values):
        values = tuple(float(v) for v in values)
        if len(values) != self.contact_state_count or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("contact class weights must contain one finite positive value per class")
        self.contact_class_weights = values
        self.contact_class_weights_is_auto = False

    def __call__(self, out, batch):
        # Forward carries exactly the temporal grid used by its predictions.
        batch = out.get("_prepared_batch", batch)
        reference = out[f"{self.predicted_state_streams[0]}_pred"]
        batch_size = reference.shape[0]
        weights = reference.new_ones(batch_size, dtype=torch.float32)
        if self.use_importance_weight and "importance_weight" in batch:
            weights = torch.as_tensor(batch["importance_weight"], device=reference.device, dtype=torch.float32)
            if weights.shape != (batch_size,) or not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise ValueError("importance_weight must be finite, non-negative and have shape [B]")
        per_sample = reference.new_zeros(batch_size, dtype=torch.float32)
        metrics = {}
        for key in self.predicted_state_streams:
            prediction, target = out[f"{key}_pred"], batch[f"{key}_future"]
            if prediction.shape != target.shape:
                raise ValueError(f"{key} prediction and target shapes must match")
            error = prediction.float() - target.float()
            mse = error.square().flatten(1).mean(1)
            per_sample = per_sample + self.stream_weights[key] * mse
            metrics[f"{key}_mse"] = mse.mean().detach()
            metrics[f"{key}_mae"] = error.abs().mean().detach()
        state_loss = (per_sample * weights).mean()
        logits = out["contact_logits"].float()
        target = batch["contact_future"]
        if target.shape != (*logits.shape[:2], 1) or logits.shape[-1] != self.contact_state_count:
            raise ValueError("contact_future must be [B, H, 1], aligned with contact_logits")
        labels = target[..., 0]
        if (not torch.isfinite(labels).all() or torch.any(labels != labels.round())
                or torch.any(labels < 0) or torch.any(labels >= self.contact_state_count)):
            raise ValueError("contact_future must contain valid integer phase labels")
        labels = labels.long()
        ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        metrics["contact_ce"] = ce.mean().detach()  # Unweighted NLL for evaluation.
        if self.contact_class_weights is not None:
            class_weights = logits.new_tensor(self.contact_class_weights)
            ce = ce * class_weights[labels]
        contact_loss = (ce.mean(1) * weights).mean()
        total = state_loss + self.contact_weight * contact_loss
        metrics.update(state_loss=state_loss.detach(), contact_loss=contact_loss.detach(), total_loss=total.detach())
        return total, metrics


class DeterministicWorldModelTrainer(BaseTrainer):
    """Use the shared training infrastructure without inheriting the flow trainer."""

    def __init__(self, config):
        config = copy.deepcopy(config)
        config.setdefault("train", {}).setdefault("output_dir", "outputs/deterministic_wm_baseline")
        super().__init__(config)
        self.loss_calculator = DeterministicWorldModelLoss(config)
        if self.device_batch_keys is not None:
            inputs = _streams((config.get("model") or {}).get("inputs", ("q", "dq", "delta_q", "tau")), "inputs")
            self.device_batch_keys.update(inputs)
            self.device_batch_keys.update(f"{key}_future" for key in self.loss_calculator.predicted_state_streams)
            self.device_batch_keys.update({"action", "action_mask", "contact_future", "importance_weight"})

    def build_dataset(self):
        return ContactWorldModelDataset(self.config, compute_normalizer=False)

    def build_model(self):
        return DeterministicRobotStateWorldModel(self.config)

    # Dataset normalization and phase-sampler methods below are local copies of
    # the WM policy. Only train-subset indices fit statistics or sampling weights.

    @staticmethod
    def _sample_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            return list(dataset.indices)
        return list(range(len(dataset)))

    def fit_dataset_normalizer(self, train_dataset):
        sample_indices = self._sample_indices(train_dataset)
        self.dataset.fit_normalizer(sample_indices)
        self._fit_contact_weight(sample_indices)

    def _fit_contact_weight(self, sample_indices):
        if not (self.config.get("contact_gate") or {}).get("enabled", False):
            return
        if self.loss_calculator.contact_class_weights_is_auto:
            frame_indices = self.dataset.covered_raw_indices(sample_indices)
            labels = self.dataset.contact.index_select(
                0,
                torch.as_tensor(
                    frame_indices,
                    dtype=torch.long,
                    device=self.dataset.contact.device,
                ),
            ).reshape(-1).round().to(dtype=torch.long)
            class_count = self.loss_calculator.contact_state_count
            counts = torch.bincount(labels, minlength=class_count).to(dtype=torch.float64)
            if torch.any(counts <= 0):
                raise ValueError(
                    "automatic three-phase weighting requires all contact phases "
                    "to be present in the training episodes"
                )
            total = counts.sum()
            weights = total / (float(class_count) * counts)
            max_weight = float(
                (self.config.get("contact_gate") or {}).get(
                    "max_class_weight", 20.0
                )
            )
            weights = weights.clamp(max=max_weight)
            self.loss_calculator.set_contact_class_weights(weights.tolist())
            log.info(
                "contact phases: counts=%s "
                "class_weights=%s",
                [int(value) for value in counts],
                [round(float(value), 4) for value in weights],
            )
            return

    def build_train_sampler(self, train_dataset):
        """Optionally stratify windows by their stored contact-phase labels.

        A fixed ``max`` aggregation over the existing future phase sequence
        assigns each window to one of the model's configured contact states.
        """
        sampling = self.train_config.get("contact_sampling") or {}
        self.dataset.importance_weight_by_sample_index = {}
        if not bool(sampling.get("enabled", False)):
            return None
        if not hasattr(self.dataset, "contact"):
            raise ValueError("contact_sampling requires dataset contact labels")
        base_indices = self._sample_indices(train_dataset)
        if not base_indices:
            raise ValueError("contact_sampling requires a non-empty train dataset")
        class_count = self.loss_calculator.contact_state_count
        phase_weights = sampling.get("phase_weights", [1.0] * class_count)
        phase_reduction = str(sampling.get("future_phase_reduction", "max")).lower()
        if phase_reduction != "max":
            raise ValueError(
                "train.contact_sampling.future_phase_reduction must be 'max'"
            )

        def sampling_phases():
            labels = []
            for sample_idx in base_indices:
                high_idx = self.dataset.valid_indices[int(sample_idx)]
                labels.append(
                    self.dataset.future_contact_phase(
                        high_idx, reduction=phase_reduction
                    )
                )
            return torch.as_tensor(labels, dtype=torch.long)

        if isinstance(phase_weights, str):
            if phase_weights.lower() == "auto":
                labels = sampling_phases()
                counts = torch.bincount(labels, minlength=class_count).float()
                phase_weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1.0), torch.zeros_like(counts))
                phase_weights = phase_weights.tolist()
            else:
                raise ValueError("train.contact_sampling.phase_weights must be a list or 'auto'")
        phase_weights = [float(value) for value in phase_weights]
        if len(phase_weights) != class_count or any(not math.isfinite(value) or value < 0 for value in phase_weights):
            raise ValueError(
                "train.contact_sampling.phase_weights must match "
                "model.contact_state_count and contain non-negative values"
            )
        labels = sampling_phases()
        if torch.any(labels < 0) or torch.any(labels >= class_count):
            raise ValueError(
                "contact sampling phase lies outside model.contact_state_count"
            )
        weights = torch.as_tensor(phase_weights, dtype=torch.double).index_select(0, labels)
        if torch.any(weights <= 0):
            raise ValueError(
                "every observed contact phase must have a positive sampling weight"
            )
        importance = weights.sum() / (float(len(weights)) * weights)
        for subset_index, correction in zip(base_indices, importance.tolist()):
            high_idx = self.dataset.valid_indices[int(subset_index)]
            self.dataset.importance_weight_by_sample_index[int(high_idx)] = float(
                correction
            )
        sampled_probability = weights / weights.sum()
        expected_phase_ratio = [
            float(sampled_probability[labels == phase].sum().item())
            for phase in range(class_count)
        ]
        log.info(
            "contact-phase sampler: source_counts=%s expected_sample_ratio=%s "
            "continuous_importance_mean=%.6f",
            torch.bincount(labels, minlength=class_count).tolist(),
            [round(value, 4) for value in expected_phase_ratio],
            float(importance.mean().item()),
        )
        return torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=int(sampling.get("num_samples", len(base_indices))),
            replacement=bool(sampling.get("replacement", True)),
        )

    def compute_loss(self, batch):
        out = self.model(batch)
        loss, out["loss_dict"] = self.loss_calculator(out, batch)
        return loss, out

    @staticmethod
    def _contact_metrics(confusion):
        counts = confusion.float()
        true_positive = counts.diagonal()
        denominator = counts.sum(0) + counts.sum(1)
        f1 = 2 * true_positive / denominator.clamp_min(1)
        return {
            "contact_accuracy": (true_positive.sum() / counts.sum().clamp_min(1)).item(),
            "contact_macro_f1": f1.mean().item(),
        }

    @torch.no_grad()
    def evaluate_loader(self, loader, epoch, description):
        training_model = self.model
        evaluation_model = self.ema.model if self.ema is not None and self.ema_use_for_validation else self.model
        previous_mode = evaluation_model.training
        self.model = evaluation_model
        self.model.eval()
        metric_sums, metric_counts = {}, {}
        class_count = self.loss_calculator.contact_state_count
        confusion = torch.zeros(class_count, class_count, dtype=torch.long, device=self.device)
        try:
            for batch in tqdm(loader, desc=f"{description} epoch {epoch}", leave=False):
                batch = self.batch_to_device(batch)
                with self.autocast_context():
                    _, out = self.compute_loss(batch)
                self._accumulate_scalar_metrics(
                    metric_sums, metric_counts, out["loss_dict"], self._batch_size(batch),
                    defer_device_sync=self.defer_metric_sync,
                )
                labels = out["_prepared_batch"]["contact_future"][..., 0].long()
                prediction = out["contact_logits"].argmax(-1)
                confusion += torch.bincount(
                    (labels * class_count + prediction).reshape(-1), minlength=class_count ** 2,
                ).reshape(class_count, class_count)
        finally:
            evaluation_model.train(previous_mode)
            self.model = training_model
        metrics = self._average_scalar_metrics(metric_sums, metric_counts)
        if not metrics:
            raise ValueError("validation loader must contain at least one sample")
        metrics.update(self._contact_metrics(confusion))
        return metrics["total_loss"], metrics

    def _model_checkpoint_metadata(self):
        return {
            **super()._model_checkpoint_metadata(),
            "deterministic_wm_contract": self.model.baseline_contract(),
            "deterministic_loss": {
                "stream_weights": self.loss_calculator.stream_weights,
                "contact_weight": self.loss_calculator.contact_weight,
                "use_importance_weight": self.loss_calculator.use_importance_weight,
                "contact_class_weights": self.loss_calculator.contact_class_weights,
            },
        }

    def _load_resume_checkpoint(self):
        if self.resume_from is None:
            return
        path = self.resolve_resume_checkpoint(self.resume_from)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.validate_checkpoint(checkpoint)
        saved_loss = checkpoint.get("deterministic_loss") or {}
        current_loss = self._model_checkpoint_metadata()["deterministic_loss"]
        if saved_loss != current_loss:
            raise ValueError("deterministic loss configuration/class weights differ from resume checkpoint")
        # The shared loader restores raw/EMA weights and all training state.
        super()._load_resume_checkpoint()


def run_smoke_test(config):
    """One synthetic optimizer step plus validation and an EMA checkpoint.

    This requires no LeRobot installation or recorded data. Dataset integration
    is exercised separately by tests with synthetic LeRobot columns.
    """
    trainer = DeterministicWorldModelTrainer(config)
    trainer.set_seed()
    trainer.model = trainer.build_model().to(trainer.device)
    model = trainer.model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.lr, weight_decay=trainer.weight_decay)
    if trainer.ema_enabled:
        trainer.ema = ModelEMA(model, trainer.ema_decay, trainer.ema_update_after_step, trainer.ema_update_every)
    trainer.dataset = SimpleNamespace(normalizer=None, filter_config={}, sample_rate_hz=model.external_state_rate_hz)
    count = 2
    batch = {key: torch.randn(count, model.external_history_horizon, model.joint_dim) for key in model.inputs}
    batch.update({f"{key}_future": torch.randn(count, model.external_future_horizon, model.joint_dim) for key in model.outputs})
    batch.update(
        action=torch.randn(count, model.action_condition_horizon, model.action_dim),
        action_mask=torch.ones(count, model.action_condition_horizon, dtype=torch.bool),
        contact_future=(torch.arange(count * model.external_future_horizon) % model.contact_state_count).reshape(count, -1, 1).float(),
        importance_weight=torch.tensor([0.5, 1.5]),
    )
    trainer.loader = [batch]
    trainer.val_loader = [batch]
    trainer.max_optimizer_steps = 1
    trainer.checkpoint_every_steps = 0
    trainer.gradient_every = 1
    before = model.state_head[-1].weight.detach().clone()
    train_loss = trainer.train_one_epoch(0)
    if trainer.global_step != 1 or torch.equal(before, model.state_head[-1].weight):
        raise RuntimeError("smoke test did not update model parameters")
    if not math.isfinite(train_loss):
        raise RuntimeError("smoke test produced non-finite loss")
    val_loss = trainer.validate_one_epoch(0)
    model.eval()
    with torch.no_grad():
        public = model.predict(trainer.batch_to_device(batch))
    trainer.save_step_checkpoint(0, {"avg_loss": train_loss, "val_loss": val_loss})
    report = {
        "optimizer_steps": trainer.global_step, "train_loss": train_loss,
        "validation": trainer.last_val_epoch_metrics,
        "parameters": sum(p.numel() for p in model.parameters()),
        "external_shapes": {key: list(public[key].shape) for key in (*[f"{k}_pred" for k in model.outputs], "contact_logits")},
        "checkpoint": str(trainer.ckpt_dir / "step_00000001.pt"),
    }
    log.info("Synthetic smoke test passed: %s", report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the deterministic robot-state world model.")
    parser.add_argument("--config", "-c", type=Path, default=Path("config/train_cfg/deterministic_wm_baseline.yaml"))
    parser.add_argument("--resume", type=Path, help="Checkpoint file or output/checkpoint directory")
    parser.add_argument("--device", help="cpu, cuda:0, or npu:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="One synthetic CPU step; no recorded data required")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    train = config.setdefault("train", {})
    if args.smoke_test:
        if args.resume:
            raise ValueError("--smoke-test starts a synthetic run; use --resume with recorded-data training")
        train.update(device="cpu", dataset_on_device=False, num_workers=0, val_num_workers=0,
                     persistent_workers=False, val_persistent_workers=False, max_optimizer_steps=1,
                     wandb={"enabled": False}, output_dir="outputs/deterministic_wm_baseline_smoke")
        train.pop("resume_from", None)
        train.pop("resume_checkpoint", None)
        train.pop("resume", None)
        torch.set_num_threads(1)
    if args.device:
        train["device"] = args.device
    if str(train.get("device", "")).split(":")[0] == "cpu":
        train["amp"] = {**(train.get("amp") or {}), "enabled": False}
        train["dataset_on_device"] = False
    if args.resume:
        train["resume_from"] = str(args.resume)
    if args.output_dir:
        train["output_dir"] = str(args.output_dir)
    if args.max_optimizer_steps is not None:
        train["max_optimizer_steps"] = args.max_optimizer_steps
    if args.smoke_test:
        run_smoke_test(config)
    else:
        trainer = DeterministicWorldModelTrainer(config)
        summary = trainer.train()
        log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
