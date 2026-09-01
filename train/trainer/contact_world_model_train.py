"""Training entry point for the Contact World Model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel, PREDICTED_STATE_STREAMS
from train.base_trainer import BaseTrainer
from train.contact_world_model_loss import ContactWorldModelLoss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the q/dq/delta_q/tau Contact World Model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/contact_world_model.yaml"),
    )
    return parser.parse_args()


class ContactWorldModelTrainer(BaseTrainer):
    """Episode-safe trainer for direct state/contact supervision."""

    def __init__(self, config):
        super().__init__(config)
        self.loss_calculator = ContactWorldModelLoss(config)
        self.validation_flow_time = float(
            self.train_config.get("validation_flow_time", 0.5)
        )
        if not 0.0 <= self.validation_flow_time <= 1.0:
            raise ValueError("train.validation_flow_time must be in [0, 1]")

        # Rollout validation is deliberately separate from the flow-matching
        # objective.  It can be used as the validation/checkpoint monitor
        # without introducing an expensive, non-differentiable training path.
        rollout_config = self.train_config.get("rollout_validation") or {}
        self.rollout_validation_enabled = bool(rollout_config.get("enabled", False))
        self.rollout_replace_val_loss = bool(
            rollout_config.get("replace_val_loss", False)
        )
        self.rollout_steps = int(
            rollout_config.get(
                "steps", (self.config.get("model") or {}).get(
                    "flow_inference_steps", 8
                )
            )
        )
        self.rollout_solver = str(
            rollout_config.get(
                "solver", (self.config.get("model") or {}).get(
                    "flow_solver", "heun"
                )
            )
        ).lower()
        self.rollout_source_seed = int(rollout_config.get("source_seed", 1234))
        self.rollout_max_batches = int(rollout_config.get("max_batches", 0))
        self.free_running_steps = int(
            rollout_config.get("free_running_steps", 8)
        )
        self.free_running_max_batches = int(
            rollout_config.get(
                "free_running_max_batches", self.rollout_max_batches
            )
        )
        configured_horizons = rollout_config.get(
            "horizons", [1, 4, 8, 16, 32]
        )
        self.rollout_horizons = tuple(
            sorted({int(value) for value in configured_horizons if int(value) > 0})
        )
        self.rollout_divergence_threshold = rollout_config.get(
            "divergence_abs_threshold", 100.0
        )
        if self.rollout_divergence_threshold is not None:
            self.rollout_divergence_threshold = float(
                self.rollout_divergence_threshold
            )
            if self.rollout_divergence_threshold <= 0.0:
                raise ValueError(
                    "train.rollout_validation.divergence_abs_threshold must be "
                    "positive or null"
                )
        if self.rollout_steps <= 0:
            raise ValueError("train.rollout_validation.steps must be positive")
        if self.rollout_solver not in {"euler", "heun"}:
            raise ValueError(
                "train.rollout_validation.solver must be 'euler' or 'heun'"
            )
        if self.rollout_max_batches < 0 or self.free_running_max_batches < 0:
            raise ValueError(
                "train.rollout_validation max_batches values must be non-negative"
            )
        if self.free_running_steps < 0:
            raise ValueError(
                "train.rollout_validation.free_running_steps must be non-negative"
            )
        if not self.rollout_horizons:
            raise ValueError(
                "train.rollout_validation.horizons must contain a positive value"
            )

    def build_dataset(self):
        return ContactWorldModelDataset(self.config, compute_normalizer=False)

    def build_model(self):
        return ContactWorldModel(self.config)

    @staticmethod
    def _sample_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            return list(dataset.indices)
        return list(range(len(dataset)))

    def fit_dataset_normalizer(self, train_dataset):
        sample_indices = self._sample_indices(train_dataset)
        self.dataset.fit_normalizer(sample_indices)
        self.loss_calculator.set_normalizer(self.dataset.normalizer)
        self._fit_contact_weight(sample_indices)

    def _fit_contact_weight(self, sample_indices):
        if not (self.config.get("contact_gate") or {}).get("enabled", False):
            return
        if self.loss_calculator.contact_class_weights_is_auto:
            frame_indices = self.dataset.covered_raw_indices(sample_indices)
            labels = self.dataset.contact.index_select(
                0, torch.as_tensor(frame_indices, dtype=torch.long)
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
        """Optionally sample windows by the phase reached in the future target.

        Sampling by the anchor's current phase only changes the history
        distribution. Contact prediction should learn imminent contact, so each
        sample is weighted by the most advanced phase present in its future
        supervision window.
        """
        sampling = self.train_config.get("contact_sampling") or {}
        if not bool(sampling.get("enabled", False)):
            return None
        if not hasattr(self.dataset, "contact"):
            raise ValueError("contact_sampling requires dataset contact labels")
        base_indices = self._sample_indices(train_dataset)
        if not base_indices:
            raise ValueError("contact_sampling requires a non-empty train dataset")
        phase_weights = sampling.get("phase_weights", [1.0, 1.0, 1.0])
        phase_reduction = str(sampling.get("future_phase_reduction", "max")).lower()
        if phase_reduction != "max":
            raise ValueError(
                "train.contact_sampling.future_phase_reduction must be 'max'"
            )

        def future_labels():
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
                labels = future_labels()
                counts = torch.bincount(labels, minlength=self.loss_calculator.contact_state_count).float()
                phase_weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1.0), torch.zeros_like(counts))
                phase_weights = phase_weights.tolist()
            else:
                raise ValueError("train.contact_sampling.phase_weights must be a list or 'auto'")
        phase_weights = [float(value) for value in phase_weights]
        if len(phase_weights) != self.loss_calculator.contact_state_count or any(value < 0 for value in phase_weights):
            raise ValueError("train.contact_sampling.phase_weights has invalid length/values")
        labels = future_labels()
        weights = torch.as_tensor(phase_weights, dtype=torch.double).index_select(0, labels)
        if torch.any(weights <= 0):
            raise ValueError("every observed phase must have a positive sampling weight")
        return torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=int(sampling.get("num_samples", len(base_indices))),
            replacement=bool(sampling.get("replacement", True)),
        )

    def compute_loss(self, batch):
        self.loss_calculator.set_global_step(self.global_step)
        flow_time = None if self.model.training else self.validation_flow_time
        out = self.model(batch, flow_time=flow_time)
        loss, loss_dict = self.loss_calculator(out, batch)
        out["loss_dict"] = loss_dict
        return loss, out

    @staticmethod
    def _metric_add(accumulator, name, value, count):
        """Accumulate a finite tensor error without retaining validation graphs."""
        value = value.detach().float()
        finite = torch.isfinite(value)
        if finite.any():
            accumulator[name + "_sum"] += float(value[finite].sum().item())
            accumulator[name + "_count"] += int(finite.sum().item())
        accumulator[name + "_nonfinite"] += int((~finite).sum().item())

    @staticmethod
    def _metric_finalize(accumulator):
        metrics = {}
        prefixes = {
            key[:-4]
            for key in accumulator
            if key.endswith("_sum")
        }
        for name in prefixes:
            count = int(accumulator.get(name + "_count", 0))
            total = float(accumulator.get(name + "_sum", 0.0))
            metrics[name] = total / count if count else float("inf")
            nonfinite = int(accumulator.get(name + "_nonfinite", 0))
            if nonfinite:
                metrics[name + "_nonfinite"] = nonfinite
        return metrics

    def _fixed_source_noise(self, batch, batch_index, *, step=0):
        """Generate reproducible Gaussian sources for comparable checkpoints."""
        reference = batch[self.model.inputs[0]]
        shape = (
            reference.shape[0],
            self.model.future_horizon,
            self.model.flow_dim,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.rollout_source_seed + int(batch_index) * 1009 + int(step) * 9176
        )
        source = torch.randn(shape, generator=generator, dtype=torch.float32)
        return source.to(device=reference.device, dtype=reference.dtype)

    def _denormalize(self, key, value):
        normalizer = getattr(self.dataset, "normalizer", None)
        if normalizer is None:
            return None
        mode = str(
            (self.config.get("dataloader") or {}).get(
                "normalize_mode", "gaussian"
            )
        ).lower()
        function = getattr(normalizer, mode + "_denormalize", None)
        if function is None:
            return None
        return function(key, value.float())

    def _accumulate_rollout_state_metrics(
        self, accumulator, predictions, targets, *, prefix, horizons
    ):
        """Collect normalized aggregate and per-stream horizon errors."""
        state_keys = self.model.predicted_state_streams
        reference_key = state_keys[0]
        usable_horizons = [
            horizon for horizon in horizons if horizon <= predictions[reference_key].shape[1]
        ]
        for horizon in usable_horizons:
            state_errors = []
            for key in state_keys:
                error = predictions[key][:, :horizon] - targets[key][:, :horizon]
                state_errors.append(error)
                error_float = error.float()
                self._metric_add(
                    accumulator,
                    f"{prefix}_{key}_mse_h{horizon}",
                    error_float.square(),
                    error_float.numel(),
                )
                self._metric_add(
                    accumulator,
                    f"{prefix}_{key}_mae_h{horizon}",
                    error_float.abs(),
                    error_float.numel(),
                )
            aggregate = torch.cat(
                [error.reshape(error.shape[0], -1) for error in state_errors],
                dim=1,
            )
            self._metric_add(
                accumulator,
                f"{prefix}_mse_h{horizon}",
                aggregate.float().square(),
                aggregate.numel(),
            )
            self._metric_add(
                accumulator,
                f"{prefix}_mae_h{horizon}",
                aggregate.float().abs(),
                aggregate.numel(),
            )

    def _accumulate_physical_metrics(
        self, accumulator, predictions, batch, *, prefix, horizons
    ):
        state_keys = self.model.predicted_state_streams
        physical_predictions = {}
        physical_targets = {}
        for key in state_keys:
            prediction = self._denormalize(key, predictions[key])
            raw_target = batch.get(f"{key}_future_raw")
            if prediction is None or raw_target is None:
                return
            physical_predictions[key] = prediction
            physical_targets[key] = raw_target.to(
                device=prediction.device, dtype=prediction.dtype
            )
        reference_key = state_keys[0]
        for horizon in horizons:
            if horizon > physical_predictions[reference_key].shape[1]:
                continue
            for key in state_keys:
                error = (
                    physical_predictions[key][:, :horizon]
                    - physical_targets[key][:, :horizon]
                )
                self._metric_add(
                    accumulator,
                    f"{prefix}_{key}_physical_mse_h{horizon}",
                    error.square(),
                    error.numel(),
                )
                self._metric_add(
                    accumulator,
                    f"{prefix}_{key}_physical_mae_h{horizon}",
                    error.abs(),
                    error.numel(),
                )
                for joint in range(error.shape[-1]):
                    joint_error = error[..., joint]
                    self._metric_add(
                        accumulator,
                        f"{prefix}_{key}_physical_mae_h{horizon}_j{joint}",
                        joint_error.abs(),
                        joint_error.numel(),
                    )
                    self._metric_add(
                        accumulator,
                        f"{prefix}_{key}_physical_mse_h{horizon}_j{joint}",
                        joint_error.square(),
                        joint_error.numel(),
                    )

    @staticmethod
    def _contact_metrics(accumulator, prediction, target, *, prefix, count):
        if prediction is None or target is None:
            return
        prediction = prediction.reshape(-1).to(dtype=torch.long)
        target = target.reshape(-1).to(device=prediction.device, dtype=torch.long)
        valid = (target >= 0) & (prediction >= 0)
        if not valid.any():
            return
        prediction = prediction[valid]
        target = target[valid]
        classes = max(
            int(prediction.max().item()) + 1,
            int(target.max().item()) + 1,
            3,
        )
        confusion = torch.bincount(
            target * classes + prediction, minlength=classes * classes
        ).reshape(classes, classes)
        previous = accumulator.get(prefix + "_confusion")
        accumulator[prefix + "_confusion"] = (
            confusion.cpu()
            if previous is None
            else previous + confusion.cpu()
        )

    @staticmethod
    def _finalize_contact_metrics(accumulator, *, prefix):
        confusion = accumulator.pop(prefix + "_confusion", None)
        if confusion is None:
            return {}
        true_positive = torch.diag(confusion).float()
        support = confusion.sum(dim=1).float()
        predicted = confusion.sum(dim=0).float()
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / support.clamp_min(1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
        result = {
            prefix + "_accuracy": float(true_positive.sum().item() / confusion.sum().clamp_min(1).item()),
            prefix + "_f1_macro": float(f1.mean().item()),
        }
        for phase, value in enumerate(f1.tolist()):
            result[f"{prefix}_f1_phase{phase}"] = float(value)
        return result

    def _run_rollout_validation(self, epoch):
        if self.val_loader is None:
            return {}
        accumulator = defaultdict(float)
        contact_prefixes = []
        training_model = self.model
        if self.ema is not None and self.ema_use_for_validation:
            self.model = self.ema.model
        self.model.eval()
        try:
            for batch_index, raw_batch in enumerate(self.val_loader):
                if self.rollout_max_batches and batch_index >= self.rollout_max_batches:
                    break
                batch = self.batch_to_device(raw_batch)
                source_noise = self._fixed_source_noise(batch, batch_index)
                with self.autocast_context():
                    integrated = self.model.predict(
                        batch,
                        steps=self.rollout_steps,
                        solver=self.rollout_solver,
                        source_noise=source_noise,
                    )
                predictions = {
                    key: integrated[f"{key}_pred"]
                    for key in self.model.predicted_state_streams
                }
                targets = {
                    key: batch[f"{key}_future"]
                    for key in self.model.predicted_state_streams
                }
                self._accumulate_rollout_state_metrics(
                    accumulator,
                    predictions,
                    targets,
                    prefix="rollout",
                    horizons=self.rollout_horizons,
                )
                self._accumulate_physical_metrics(
                    accumulator,
                    predictions,
                    batch,
                    prefix="rollout",
                    horizons=self.rollout_horizons,
                )
                contact_prediction = integrated.get("contact_state_pred")
                contact_target = batch.get("contact_future")
                self._contact_metrics(
                    accumulator,
                    contact_prediction,
                    contact_target,
                    prefix="rollout_contact",
                    count=batch[self.model.inputs[0]].shape[0],
                )
                if "rollout_contact_confusion" not in contact_prefixes:
                    contact_prefixes.append("rollout_contact")
                accumulator["rollout_batches"] += 1
                accumulator["rollout_samples"] += batch[self.model.inputs[0]].shape[0]
                prediction_values = [predictions[key] for key in predictions]
                prediction_values.append(integrated.get("flow_state_pred"))
                finite = all(
                    value is not None and bool(torch.isfinite(value).all().item())
                    for value in prediction_values
                )
                accumulator["rollout_nonfinite_batches"] += int(not finite)
                max_abs = max(
                    float(value.detach().float().abs().nan_to_num(posinf=float("inf")).max().item())
                    for value in prediction_values
                    if value is not None
                )
                accumulator["rollout_max_abs"] = max(
                    accumulator["rollout_max_abs"], max_abs
                )
                if (
                    self.rollout_divergence_threshold is not None
                    and (not finite or max_abs > self.rollout_divergence_threshold)
                ):
                    accumulator["rollout_diverged_batches"] += 1

                if self.free_running_steps <= 0:
                    continue
                if (
                    self.free_running_max_batches
                    and batch_index >= self.free_running_max_batches
                ):
                    continue
                horizon = min(self.free_running_steps, self.model.future_horizon)
                history = {
                    key: batch[key].clone() for key in self.model.inputs
                }
                running_batch = dict(batch)
                free_predictions = {key: [] for key in self.model.predicted_state_streams}
                free_contacts = []
                for step in range(horizon):
                    running_batch.update(history)
                    if "action_rollout" in batch:
                        if step >= batch["action_rollout"].shape[1]:
                            raise ValueError(
                                "action_rollout does not cover free-running validation"
                            )
                        running_batch["action"] = batch["action_rollout"][:, step]
                        if "action_rollout_mask" in batch:
                            running_batch["action_mask"] = batch[
                                "action_rollout_mask"
                            ][:, step]
                    step_noise = self._fixed_source_noise(
                        running_batch, batch_index, step=step + 1
                    )
                    with self.autocast_context():
                        step_output = self.model.predict(
                            running_batch,
                            steps=self.rollout_steps,
                            solver=self.rollout_solver,
                            source_noise=step_noise,
                        )
                    for key in free_predictions:
                        first = step_output[f"{key}_pred"][:, :1]
                        free_predictions[key].append(first)
                        history[key] = torch.cat(
                            (history[key][:, 1:], first), dim=1
                        )
                    if step_output.get("contact_state_pred") is not None:
                        free_contacts.append(
                            step_output["contact_state_pred"][:, :1]
                        )
                free_predictions = {
                    key: torch.cat(values, dim=1)
                    for key, values in free_predictions.items()
                }
                free_prediction_values = list(free_predictions.values())
                free_finite = all(
                    bool(torch.isfinite(value).all().item())
                    for value in free_prediction_values
                )
                free_max_abs = max(
                    float(
                        value.detach()
                        .float()
                        .abs()
                        .nan_to_num(posinf=float("inf"))
                        .max()
                        .item()
                    )
                    for value in free_prediction_values
                )
                accumulator["free_running_nonfinite_batches"] += int(
                    not free_finite
                )
                accumulator["free_running_max_abs"] = max(
                    accumulator["free_running_max_abs"], free_max_abs
                )
                if (
                    self.rollout_divergence_threshold is not None
                    and (
                        not free_finite
                        or free_max_abs > self.rollout_divergence_threshold
                    )
                ):
                    accumulator["free_running_diverged_batches"] += 1
                free_targets = {
                    key: batch[f"{key}_future"][:, :horizon]
                    for key in free_predictions
                }
                self._accumulate_rollout_state_metrics(
                    accumulator,
                    free_predictions,
                    free_targets,
                    prefix="free_running",
                    horizons=self.rollout_horizons,
                )
                self._accumulate_physical_metrics(
                    accumulator,
                    free_predictions,
                    batch,
                    prefix="free_running",
                    horizons=self.rollout_horizons,
                )
                if free_contacts:
                    free_contact_prediction = torch.cat(free_contacts, dim=1)
                    free_contact_target = batch["contact_future"][:, :horizon]
                    self._contact_metrics(
                        accumulator,
                        free_contact_prediction,
                        free_contact_target,
                        prefix="free_running_contact",
                        count=batch[self.model.inputs[0]].shape[0],
                    )
                    if "free_running_contact_confusion" not in contact_prefixes:
                        contact_prefixes.append("free_running_contact")
                accumulator["free_running_batches"] += 1
        finally:
            self.model = training_model

        metrics = self._metric_finalize(accumulator)
        for prefix in contact_prefixes:
            metrics.update(self._finalize_contact_metrics(accumulator, prefix=prefix))
        metrics["rollout_batches"] = int(accumulator.get("rollout_batches", 0))
        metrics["rollout_samples"] = int(accumulator.get("rollout_samples", 0))
        metrics["rollout_nonfinite_batches"] = int(
            accumulator.get("rollout_nonfinite_batches", 0)
        )
        metrics["rollout_diverged_batches"] = int(
            accumulator.get("rollout_diverged_batches", 0)
        )
        metrics["rollout_max_abs"] = float(
            accumulator.get("rollout_max_abs", 0.0)
        )
        metrics["free_running_batches"] = int(
            accumulator.get("free_running_batches", 0)
        )
        metrics["free_running_nonfinite_batches"] = int(
            accumulator.get("free_running_nonfinite_batches", 0)
        )
        metrics["free_running_diverged_batches"] = int(
            accumulator.get("free_running_diverged_batches", 0)
        )
        metrics["free_running_max_abs"] = float(
            accumulator.get("free_running_max_abs", 0.0)
        )
        metrics["rollout_finite"] = float(
            metrics.get("rollout_nonfinite_batches", 0.0) == 0.0
        )
        metrics["rollout_diverged"] = float(
            metrics.get("rollout_diverged_batches", 0.0) > 0.0
        )
        metrics["free_running_finite"] = float(
            metrics.get("free_running_nonfinite_batches", 0.0) == 0.0
        )
        metrics["free_running_diverged"] = float(
            metrics.get("free_running_diverged_batches", 0.0) > 0.0
        )
        metrics["rollout_epoch"] = int(epoch)
        if "rollout_mse_h32" in metrics:
            metrics["rollout_loss"] = metrics["rollout_mse_h32"]
        elif "rollout_mse_h1" in metrics:
            metrics["rollout_loss"] = metrics["rollout_mse_h1"]
        return metrics

    def validate_one_epoch(self, epoch):
        # First retain the ordinary flow/direct/physics validation diagnostics.
        val_loss = super().validate_one_epoch(epoch)
        if not self.rollout_validation_enabled or self.val_loader is None:
            return val_loss
        rollout_metrics = self._run_rollout_validation(epoch)
        self.last_val_epoch_metrics.update(rollout_metrics)
        if self.rollout_replace_val_loss:
            replacement = rollout_metrics.get("rollout_mse_h32")
            if replacement is None:
                replacement = rollout_metrics.get("rollout_mse_h1")
            if replacement is not None:
                val_loss = float(replacement)
        return val_loss


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    log.info("Contact World Model config: %s", config)
    trainer = ContactWorldModelTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
