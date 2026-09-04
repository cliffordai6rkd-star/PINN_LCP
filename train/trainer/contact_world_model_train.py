"""Training entry point for the Contact World Model."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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
from train.carswm_checkpoint_viz import render_checkpoint_summary
from train.carswm_metrics import distribution_metrics
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
        self.rollout_validation_steps = int(
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
        self.feedback_reconditioned_enabled = bool(
            rollout_config.get("feedback_reconditioned", True)
        )
        configured_update_intervals = rollout_config.get(
            "measurement_update_intervals", [1, 4, 8, 32]
        )
        if not isinstance(configured_update_intervals, (list, tuple)):
            raise ValueError(
                "train.rollout_validation.measurement_update_intervals must be "
                "a sequence"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in configured_update_intervals
        ):
            raise ValueError(
                "train.rollout_validation.measurement_update_intervals must "
                "contain positive integers"
            )
        self.feedback_measurement_update_intervals = (
            tuple(sorted({int(value) for value in configured_update_intervals}))
            if self.feedback_reconditioned_enabled
            else ()
        )
        self.feedback_num_samples = int(rollout_config.get("num_samples", 8))
        self.feedback_max_batches = int(
            rollout_config.get("feedback_max_batches", self.rollout_max_batches)
        )
        default_feedback_metric = (
            f"feedback_u{max(self.feedback_measurement_update_intervals)}_energy_score"
            if self.feedback_measurement_update_intervals
            else "rollout_mse_h32"
        )
        self.rollout_replace_val_loss_metric = str(
            rollout_config.get("replace_val_loss_metric", default_feedback_metric)
        ).strip()
        if not self.rollout_replace_val_loss_metric:
            raise ValueError(
                "train.rollout_validation.replace_val_loss_metric must not be empty"
            )
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
        if self.rollout_validation_steps <= 0:
            raise ValueError("train.rollout_validation.steps must be positive")
        if self.rollout_solver not in {"euler", "heun"}:
            raise ValueError(
                "train.rollout_validation.solver must be 'euler' or 'heun'"
            )
        if (
            self.rollout_max_batches < 0
            or self.free_running_max_batches < 0
            or self.feedback_max_batches < 0
        ):
            raise ValueError(
                "train.rollout_validation max_batches values must be non-negative"
            )
        if self.feedback_num_samples < 1:
            raise ValueError(
                "train.rollout_validation.num_samples must be positive"
            )
        if self.free_running_steps < 0:
            raise ValueError(
                "train.rollout_validation.free_running_steps must be non-negative"
            )
        if not self.rollout_horizons:
            raise ValueError(
                "train.rollout_validation.horizons must contain a positive value"
            )
        if (
            self.rollout_validation_enabled
            and self.feedback_measurement_update_intervals
        ):
            future_horizon = int(
                (self.config.get("dataloader") or {}).get(
                    "prediction_horizon", 32
                )
            )
            required_action_rollout = max(
                ((future_horizon - 1) // interval) * interval + 1
                for interval in self.feedback_measurement_update_intervals
            )
            configured_action_rollout = int(
                (self.config.get("dataloader") or {}).get(
                    "action_rollout_horizon", 0
                )
            )
            if configured_action_rollout < required_action_rollout:
                raise ValueError(
                    "dataloader.action_rollout_horizon must be at least "
                    f"{required_action_rollout} for feedback-reconditioned "
                    "validation"
                )
        probability_config = self.train_config.get("probabilistic_validation") or {}
        self.probabilistic_validation_enabled = bool(
            probability_config.get("enabled", True)
        )
        self.probabilistic_num_samples = int(
            probability_config.get("num_samples", 8)
        )
        self.probabilistic_max_batches = int(
            probability_config.get("max_batches", 8)
        )
        self.probabilistic_every = int(probability_config.get("every", 1))
        self.probabilistic_replace_val_loss = bool(
            probability_config.get("replace_val_loss", True)
        )
        if self.probabilistic_num_samples < 1:
            raise ValueError("probabilistic_validation.num_samples must be positive")
        if self.probabilistic_max_batches < 0 or self.probabilistic_every < 1:
            raise ValueError(
                "probabilistic_validation max_batches/every must be non-negative/positive"
            )
        viz_config = self.train_config.get("checkpoint_visualization") or {}
        self.checkpoint_visualization_enabled = bool(
            viz_config.get("enabled", False)
        )
        self.checkpoint_visualization_every_saved = bool(
            viz_config.get("every_saved_checkpoint", True)
        )
        self.checkpoint_visualization_indices = dict(
            viz_config.get("fixed_validation_indices") or {}
        )
        self.checkpoint_visualization_paired_indices = dict(
            viz_config.get("paired_validation_indices") or {}
        )
        self.checkpoint_visualization_num_samples = int(
            viz_config.get("num_samples", 32)
        )
        self.checkpoint_visualization_seed = int(viz_config.get("seed", 2027))
        self.checkpoint_visualization_use_ema = bool(viz_config.get("use_ema", True))
        self.checkpoint_visualization_denormalize = bool(
            viz_config.get("denormalize_for_plot", True)
        )
        self.checkpoint_visualization_wrist_joint = int(
            viz_config.get("wrist_joint_index", 5)
        )
        self.checkpoint_visualization_flow_steps = int(
            viz_config.get(
                "flow_steps",
                (config.get("model") or {}).get("flow_inference_steps", 8),
            )
        )
        self.checkpoint_visualization_flow_solver = str(
            viz_config.get(
                "flow_solver",
                (config.get("model") or {}).get("flow_solver", "heun"),
            )
        ).lower()
        plot_ranges = viz_config.get("plot_ranges") or {}
        self.checkpoint_visualization_plot_ranges = {
            key: tuple(float(value) for value in plot_ranges[key])
            for key in ("q", "tau")
            if key in plot_ranges and plot_ranges[key] is not None
        }
        self._checkpoint_visualization_records = []
        self._checkpoint_visualization_noise = {}
        if self.checkpoint_visualization_enabled:
            if self.checkpoint_visualization_num_samples < 2:
                raise ValueError("checkpoint_visualization.num_samples must be at least 2")
            if not 0 <= self.checkpoint_visualization_wrist_joint < self.loss_calculator.joint_dim:
                raise ValueError("checkpoint_visualization.wrist_joint_index is out of range")
            if self.checkpoint_visualization_flow_steps < 1:
                raise ValueError("checkpoint_visualization.flow_steps must be positive")
            if self.checkpoint_visualization_flow_solver not in {"euler", "heun"}:
                raise ValueError(
                    "checkpoint_visualization.flow_solver must be euler or heun"
                )
            configured_streams = set(self.loss_calculator.predicted_state_streams)
            if not {"q", "tau"}.issubset(configured_streams):
                raise ValueError(
                    "six-panel checkpoint visualization requires q and tau in "
                    "model.outputs; disable it for other valid output selections"
                )
            for key, limits in self.checkpoint_visualization_plot_ranges.items():
                if len(limits) != 2 or not limits[0] < limits[1]:
                    raise ValueError(
                        f"checkpoint_visualization.plot_ranges.{key} must be [min, max]"
                    )

    def build_dataset(self):
        return ContactWorldModelDataset(self.config, compute_normalizer=False)

    def build_model(self):
        return ContactWorldModel(self.config)

    def setup(self):
        super().setup()
        self._prepare_checkpoint_visualization()

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
        if len(phase_weights) != class_count or any(value < 0 for value in phase_weights):
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
        total_steps = getattr(self, "max_optimizer_steps", None)
        if total_steps is None and self.loader is not None:
            total_steps = self.num_epochs * len(self.loader)
        self.loss_calculator.set_global_step(self.global_step, total_steps)
        flow_time = None if self.model.training else self.validation_flow_time
        out = self.model(batch, flow_time=flow_time)
        loss, loss_dict = self.loss_calculator(out, batch)
        out["loss_dict"] = loss_dict
        return loss, out

    @staticmethod
    def _phase_names(class_count):
        if int(class_count) == 3:
            return ("free", "pre_contact", "contact")
        return tuple(f"phase_{index}" for index in range(int(class_count)))

    def _validation_base_indices(self):
        if self.val_loader is None:
            return []
        dataset = self.val_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            return [int(value) for value in dataset.indices]
        return list(range(len(dataset)))

    def _prepare_checkpoint_visualization(self):
        if not self.checkpoint_visualization_enabled:
            return
        base_indices = self._validation_base_indices()
        if not base_indices:
            raise ValueError(
                "checkpoint_visualization requires a non-empty validation split"
            )
        phase_names = self._phase_names(self.loss_calculator.contact_state_count)
        selected = {
            str(name): int(index)
            for name, index in self.checkpoint_visualization_indices.items()
        }
        if not selected:
            for index in base_indices:
                high_idx = self.dataset.valid_indices[index]
                phase = int(self.dataset.future_contact_phase(high_idx, reduction="max"))
                if 0 <= phase < len(phase_names) and phase_names[phase] not in selected:
                    selected[phase_names[phase]] = index
                if len(selected) == len(phase_names):
                    break
        selected.update(
            {
                str(name): int(index)
                for name, index in self.checkpoint_visualization_paired_indices.items()
            }
        )
        invalid = {
            name: index for name, index in selected.items() if index not in base_indices
        }
        if invalid:
            raise ValueError(
                "checkpoint_visualization indices must be base dataset indices "
                f"inside the fixed validation split: {invalid}"
            )
        if not selected:
            raise ValueError("checkpoint_visualization could not select an existing phase")

        # Cache exact inputs once. Dataset augmentation, when enabled, can no
        # longer change a visualized anchor across checkpoints.
        from torch.utils.data._utils.collate import default_collate

        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.checkpoint_visualization_seed)
            records = []
            for name, index in selected.items():
                sample = self.dataset[index]
                records.append(
                    {
                        "name": name,
                        "index": index,
                        "raw_index": int(self.dataset.valid_indices[index]),
                        "batch": default_collate([sample]),
                    }
                )
        self._checkpoint_visualization_records = records
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.checkpoint_visualization_seed)
        for record in records:
            self._checkpoint_visualization_noise[record["name"]] = torch.randn(
                1,
                self.checkpoint_visualization_num_samples,
                self.model.future_horizon,
                self.model.flow_dim,
                generator=generator,
                dtype=torch.float32,
            )
        viz_dir = self.output_dir / "checkpoint_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "phase_aggregation": "max_existing_future_contact_phase",
            "fixed_validation_indices": {
                record["name"]: record["index"] for record in records
            },
            "raw_anchor_indices": {
                record["name"]: record["raw_index"] for record in records
            },
            "num_samples": self.checkpoint_visualization_num_samples,
            "seed": self.checkpoint_visualization_seed,
        }
        (viz_dir / "fixed_samples.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _checkpoint_visualization_physical(self, key, value):
        if not self.checkpoint_visualization_denormalize:
            return value.float()
        physical = self._denormalize(key, value)
        return value.float() if physical is None else physical.float()

    @staticmethod
    def _mean_metric_dict(values):
        return {
            key: float(torch.cat(items).mean().item())
            for key, items in values.items()
            if items
        }

    @staticmethod
    def _plot_limits(values):
        array = torch.cat([value.reshape(-1).float().cpu() for value in values])
        minimum = float(array.min().item())
        maximum = float(array.max().item())
        width = max(maximum - minimum, 1.0e-3)
        return [minimum - 0.05 * width, maximum + 0.05 * width]

    @torch.no_grad()
    def _save_checkpoint_visualization(self, epoch, checkpoint_path):
        if not self._checkpoint_visualization_records:
            return
        visualization_model = (
            self.ema.model
            if self.checkpoint_visualization_use_ema and self.ema is not None
            else self.model
        )
        was_training = visualization_model.training
        visualization_model.eval()
        plot_records = []
        metric_values = defaultdict(list)
        scale_values = {"q": [], "tau": []}
        try:
            for record in self._checkpoint_visualization_records:
                batch = self.batch_to_device(record["batch"])
                reference = batch[visualization_model.inputs[0]]
                source_noise = self._checkpoint_visualization_noise[record["name"]].to(
                    device=reference.device, dtype=reference.dtype
                )
                samples = visualization_model.sample(
                    batch,
                    num_samples=self.checkpoint_visualization_num_samples,
                    steps=self.checkpoint_visualization_flow_steps,
                    solver=self.checkpoint_visualization_flow_solver,
                    source_noise=source_noise,
                )
                normalized_samples = {
                    key: samples[f"{key}_pred"].float()
                    for key in visualization_model.predicted_state_streams
                }
                normalized_targets = {
                    key: batch[f"{key}_future"].float()
                    for key in visualization_model.predicted_state_streams
                }
                metrics = distribution_metrics(
                    normalized_samples,
                    normalized_targets,
                    samples.get("contact_probability"),
                    batch.get("contact_future"),
                )
                for key, value in metrics.items():
                    metric_values[key].append(value.detach().float().reshape(-1).cpu())
                physical_samples = {
                    key: self._checkpoint_visualization_physical(key, value)[0].cpu()
                    for key, value in normalized_samples.items()
                }
                physical_targets = {
                    key: self._checkpoint_visualization_physical(key, value)[0].cpu()
                    for key, value in normalized_targets.items()
                }
                if "q" not in physical_samples or "tau" not in physical_samples:
                    raise ValueError(
                        "checkpoint visualization panels require q and tau in model.outputs"
                    )
                scale_values["q"].extend((physical_samples["q"], physical_targets["q"]))
                scale_values["tau"].extend((physical_samples["tau"], physical_targets["tau"]))
                plot_records.append(
                    {
                        "name": record["name"],
                        "samples": {
                            key: value.numpy() for key, value in physical_samples.items()
                        },
                        "targets": {
                            key: value.numpy() for key, value in physical_targets.items()
                        },
                        "contact_probability": samples["contact_probability"][0].float().cpu().numpy(),
                        "contact_target": batch["contact_future"][0].float().cpu().numpy(),
                    }
                )
        finally:
            visualization_model.train(was_training)

        metrics = self._mean_metric_dict(metric_values)
        metrics.update(
            {
                "step": int(self.global_step),
                "epoch": int(epoch),
                "checkpoint": Path(checkpoint_path).name,
                "model_version": getattr(self.model, "MODEL_VERSION", None),
                "num_samples": self.checkpoint_visualization_num_samples,
                "visualization_seed": self.checkpoint_visualization_seed,
                "flow_steps": self.checkpoint_visualization_flow_steps,
                "nfe": self.checkpoint_visualization_flow_steps
                * (2 if self.checkpoint_visualization_flow_solver == "heun" else 1),
                "phase_aggregation": "max_existing_future_contact_phase",
            }
        )
        viz_dir = self.output_dir / "checkpoint_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        scale_path = viz_dir / "plot_scales.json"
        if scale_path.exists():
            scales = json.loads(scale_path.read_text(encoding="utf-8"))
        else:
            scales = dict(self.checkpoint_visualization_plot_ranges)
            for key in ("q", "tau"):
                scales.setdefault(key, self._plot_limits(scale_values[key]))
            scale_path.write_text(
                json.dumps(scales, indent=2, sort_keys=True), encoding="utf-8"
            )
        existing_metrics = []
        for path in sorted(viz_dir.glob("step_*_metrics.json")):
            if path.name != f"step_{self.global_step:08d}_metrics.json":
                existing_metrics.append(json.loads(path.read_text(encoding="utf-8")))
        image_path = viz_dir / f"step_{self.global_step:08d}_summary.png"
        render_checkpoint_summary(
            image_path,
            plot_records,
            existing_metrics,
            metrics,
            scales,
            wrist_joint_index=self.checkpoint_visualization_wrist_joint,
            contact_names=self._phase_names(self.loss_calculator.contact_state_count),
        )
        metrics_path = viz_dir / f"step_{self.global_step:08d}_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        shutil.copyfile(image_path, viz_dir / "latest_summary.png")

    def save_step_checkpoint(self, epoch, metrics):
        super().save_step_checkpoint(epoch, metrics)
        if (
            self.checkpoint_visualization_enabled
            and self.checkpoint_visualization_every_saved
            and self._last_step_checkpoint == self.global_step
        ):
            self._save_checkpoint_visualization(
                epoch, self.ckpt_dir / f"step_{self.global_step:08d}.pt"
            )

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

    def _fixed_source_noise(
        self, batch, batch_index, *, step=0, num_samples=None
    ):
        """Generate reproducible Gaussian sources for comparable checkpoints."""
        reference = batch[self.model.inputs[0]]
        shape = (reference.shape[0],) + (
            self.model.future_horizon,
            self.model.flow_dim,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.rollout_source_seed + int(batch_index) * 1009 + int(step) * 9176
        )
        if num_samples is None:
            source = torch.randn(
                shape, generator=generator, dtype=torch.float32
            )
        else:
            # Sample-major generation makes draw zero bitwise identical to the
            # deterministic open-loop source for every batch element.
            source = torch.randn(
                (int(num_samples),) + shape,
                generator=generator,
                dtype=torch.float32,
            ).permute(1, 0, 2, 3).contiguous()
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

    @staticmethod
    def _feedback_phase_names(class_count):
        if int(class_count) == 3:
            return ("free", "transition", "contact")
        return tuple(f"phase_{index}" for index in range(int(class_count)))

    def _feedback_condition_batch(self, batch, history, anchor):
        """Build only the causal conditions available at one feedback anchor."""

        action_rollout = batch.get("action_rollout")
        action_rollout_mask = batch.get("action_rollout_mask")
        if action_rollout is None or action_rollout_mask is None:
            raise KeyError(
                "feedback-reconditioned validation requires action_rollout and "
                "action_rollout_mask"
            )
        if action_rollout.ndim != 4:
            raise ValueError(
                "action_rollout must have shape [B, R, A, Da] for "
                "feedback-reconditioned validation"
            )
        if action_rollout_mask.ndim != 3:
            raise ValueError(
                "action_rollout_mask must have shape [B, R, A] for "
                "feedback-reconditioned validation"
            )
        if tuple(action_rollout_mask.shape) != tuple(action_rollout.shape[:-1]):
            raise ValueError(
                "action_rollout_mask shape must match action_rollout [B, R, A]"
            )
        if int(anchor) >= action_rollout.shape[1]:
            raise ValueError(
                "action_rollout does not cover feedback anchor "
                f"{int(anchor)}"
            )
        condition = {key: history[key] for key in self.model.inputs}
        condition["action"] = action_rollout[:, int(anchor)]
        condition["action_mask"] = action_rollout_mask[:, int(anchor)]
        return condition

    def _append_feedback_measurements(self, history, batch, start, stop):
        """Advance history exclusively with recorded measurements in [start, stop)."""

        updated = {}
        for key in self.model.inputs:
            future_key = f"{key}_future"
            recorded = batch.get(future_key)
            if recorded is None:
                raise KeyError(
                    "feedback-reconditioned validation requires recorded future "
                    f"measurements in {future_key!r}"
                )
            if int(stop) > recorded.shape[1]:
                raise ValueError(
                    f"{future_key} does not cover feedback update ending at {stop}"
                )
            history_length = history[key].shape[1]
            measured = recorded[:, int(start) : int(stop)]
            updated[key] = torch.cat((history[key], measured), dim=1)[
                :, -history_length:
            ]
        return updated

    @torch.no_grad()
    def _feedback_reconditioned_samples(self, batch, batch_index, interval):
        """Sample one measurement-updated offline trajectory.

        The model sees only state history through the current anchor. Predictions
        are retained for scoring but are never inserted into a later history.
        """

        interval = int(interval)
        if interval <= 0:
            raise ValueError("feedback measurement interval must be positive")
        horizon = int(self.model.future_horizon)
        history = {key: batch[key].clone() for key in self.model.inputs}
        sample_chunks = {
            key: [] for key in self.model.predicted_state_streams
        }
        target_chunks = {
            key: [] for key in self.model.predicted_state_streams
        }
        contact_probability_chunks = []
        contact_target = batch.get("contact_future")
        if contact_target is None or contact_target.shape[1] < horizon:
            raise ValueError(
                "contact_future must cover the complete feedback validation horizon"
            )
        segments = []
        anchors = []
        for anchor in range(0, horizon, interval):
            stop = min(anchor + interval, horizon)
            condition = self._feedback_condition_batch(batch, history, anchor)
            source_noise = self._fixed_source_noise(
                condition,
                batch_index,
                step=anchor,
                num_samples=self.feedback_num_samples,
            )
            with self.autocast_context():
                sampled = self.model.sample(
                    condition,
                    num_samples=self.feedback_num_samples,
                    steps=self.rollout_validation_steps,
                    solver=self.rollout_solver,
                    source_noise=source_noise,
                )
            segment_samples = {}
            segment_targets = {}
            for key in self.model.predicted_state_streams:
                future_key = f"{key}_future"
                target = batch.get(future_key)
                if target is None or target.shape[1] < horizon:
                    raise ValueError(
                        f"{future_key} must cover the complete feedback "
                        "validation horizon"
                    )
                segment_samples[key] = sampled[f"{key}_pred"][
                    :, :, : stop - anchor
                ].float()
                segment_targets[key] = target[:, anchor:stop].float()
                sample_chunks[key].append(segment_samples[key])
                target_chunks[key].append(segment_targets[key])

            segment_contact_probability = sampled.get("contact_probability")
            if segment_contact_probability is not None:
                segment_contact_probability = segment_contact_probability[
                    :, :, : stop - anchor
                ].float()
                contact_probability_chunks.append(segment_contact_probability)
            segment_contact_target = contact_target[:, anchor:stop]
            segment_metrics = distribution_metrics(
                segment_samples,
                segment_targets,
                segment_contact_probability,
                segment_contact_target,
            )
            segment_phase = (
                segment_contact_target.reshape(
                    segment_contact_target.shape[0], -1
                )
                .round()
                .long()
                .max(dim=1)
                .values
            )
            segments.append(
                {
                    "anchor": anchor,
                    "stop": stop,
                    "phase": segment_phase,
                    "metrics": segment_metrics,
                }
            )
            anchors.append(anchor)
            history = self._append_feedback_measurements(
                history, batch, anchor, stop
            )

        samples = {
            key: torch.cat(chunks, dim=2)
            for key, chunks in sample_chunks.items()
        }
        targets = {
            key: torch.cat(chunks, dim=1)
            for key, chunks in target_chunks.items()
        }
        contact_probability = (
            torch.cat(contact_probability_chunks, dim=2)
            if contact_probability_chunks
            else None
        )
        return {
            "anchors": tuple(anchors),
            "samples": samples,
            "targets": targets,
            "contact_probability": contact_probability,
            "contact_target": contact_target[:, :horizon],
            "segments": segments,
        }

    def _accumulate_feedback_interval(
        self, accumulator, batch, batch_index, interval
    ):
        prefix = f"feedback_u{int(interval)}"
        result = self._feedback_reconditioned_samples(
            batch, batch_index, interval
        )
        values = distribution_metrics(
            result["samples"],
            result["targets"],
            result["contact_probability"],
            result["contact_target"],
        )
        for name, value in values.items():
            self._metric_add(
                accumulator,
                f"{prefix}_{name}",
                value,
                value.numel(),
            )

        phase_names = self._feedback_phase_names(
            self.loss_calculator.contact_state_count
        )
        for segment in result["segments"]:
            phase = segment["phase"]
            for phase_index, phase_name in enumerate(phase_names):
                mask = phase == phase_index
                if not mask.any():
                    continue
                for name, value in segment["metrics"].items():
                    self._metric_add(
                        accumulator,
                        f"{prefix}_phase_{phase_name}_{name}",
                        value[mask],
                        int(mask.sum().item()),
                    )

        sample_values = list(result["samples"].values())
        finite = all(
            bool(torch.isfinite(value).all().item()) for value in sample_values
        )
        max_abs = max(
            float(
                value.detach()
                .abs()
                .nan_to_num(posinf=float("inf"))
                .max()
                .item()
            )
            for value in sample_values
        )
        accumulator[f"{prefix}_batches"] += 1
        accumulator[f"{prefix}_evaluated_samples"] += batch[
            self.model.inputs[0]
        ].shape[0]
        accumulator[f"{prefix}_segments"] += len(result["segments"])
        accumulator[f"{prefix}_nonfinite_batches"] += int(not finite)
        accumulator[f"{prefix}_max_abs"] = max(
            accumulator[f"{prefix}_max_abs"], max_abs
        )
        if (
            self.rollout_divergence_threshold is not None
            and (not finite or max_abs > self.rollout_divergence_threshold)
        ):
            accumulator[f"{prefix}_diverged_batches"] += 1

    @torch.no_grad()
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
                        steps=self.rollout_validation_steps,
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

                if (
                    self.feedback_measurement_update_intervals
                    and (
                        not self.feedback_max_batches
                        or batch_index < self.feedback_max_batches
                    )
                ):
                    for interval in self.feedback_measurement_update_intervals:
                        self._accumulate_feedback_interval(
                            accumulator, batch, batch_index, interval
                        )

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
                            steps=self.rollout_validation_steps,
                            solver=self.rollout_solver,
                            source_noise=step_noise,
                        )
                    for key in free_predictions:
                        first = step_output[f"{key}_pred"][:, :1]
                        free_predictions[key].append(first)
                        # Only predicted streams that are also conditions can
                        # be fed back into the next free-running window.
                        if key in history:
                            history[key] = torch.cat(
                                (history[key][:, 1:], first), dim=1
                            )
                    # An input stream omitted from model.outputs is advanced
                    # with its recorded future observation.  This keeps the
                    # teacher validation well-defined without inventing a
                    # prediction for that conditioning modality.
                    for key in self.model.inputs:
                        if key in self.model.predicted_state_streams:
                            continue
                        recorded = batch.get(f"{key}_future")
                        if recorded is None or step >= recorded.shape[1]:
                            raise ValueError(
                                "free-running validation requires recorded future "
                                f"values for unpredicted input stream {key!r}"
                            )
                        history[key] = torch.cat(
                            (history[key][:, 1:], recorded[:, step : step + 1]),
                            dim=1,
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
        for interval in self.feedback_measurement_update_intervals:
            prefix = f"feedback_u{interval}"
            metrics[f"{prefix}_batches"] = int(
                accumulator.get(f"{prefix}_batches", 0)
            )
            metrics[f"{prefix}_evaluated_samples"] = int(
                accumulator.get(f"{prefix}_evaluated_samples", 0)
            )
            metrics[f"{prefix}_segments"] = int(
                accumulator.get(f"{prefix}_segments", 0)
            )
            metrics[f"{prefix}_num_samples"] = self.feedback_num_samples
            metrics[f"{prefix}_nonfinite_batches"] = int(
                accumulator.get(f"{prefix}_nonfinite_batches", 0)
            )
            metrics[f"{prefix}_diverged_batches"] = int(
                accumulator.get(f"{prefix}_diverged_batches", 0)
            )
            metrics[f"{prefix}_max_abs"] = float(
                accumulator.get(f"{prefix}_max_abs", 0.0)
            )
            metrics[f"{prefix}_finite"] = float(
                metrics[f"{prefix}_nonfinite_batches"] == 0
            )
            metrics[f"{prefix}_diverged"] = float(
                metrics[f"{prefix}_diverged_batches"] > 0
            )
            metrics[f"{prefix}_epoch"] = int(epoch)
        metrics["rollout_epoch"] = int(epoch)
        if "rollout_mse_h32" in metrics:
            metrics["rollout_loss"] = metrics["rollout_mse_h32"]
        elif "rollout_mse_h1" in metrics:
            metrics["rollout_loss"] = metrics["rollout_mse_h1"]
        return metrics

    @torch.no_grad()
    def _run_probabilistic_validation(self, epoch):
        if self.val_loader is None:
            return {}
        accumulator = defaultdict(float)
        training_model = self.model
        if self.ema is not None and self.ema_use_for_validation:
            self.model = self.ema.model
        self.model.eval()
        try:
            for batch_index, raw_batch in enumerate(self.val_loader):
                if self.probabilistic_max_batches and batch_index >= self.probabilistic_max_batches:
                    break
                batch = self.batch_to_device(raw_batch)
                reference = batch[self.model.inputs[0]]
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    self.rollout_source_seed + int(batch_index) * 1009
                )
                source_noise = torch.randn(
                    reference.shape[0],
                    self.probabilistic_num_samples,
                    self.model.future_horizon,
                    self.model.flow_dim,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=reference.device, dtype=reference.dtype)
                with self.autocast_context():
                    samples = self.model.sample(
                        batch,
                        num_samples=self.probabilistic_num_samples,
                        steps=self.rollout_validation_steps,
                        solver=self.rollout_solver,
                        source_noise=source_noise,
                    )
                sample_streams = {
                    key: samples[f"{key}_pred"].float()
                    for key in self.model.predicted_state_streams
                }
                targets = {
                    key: batch[f"{key}_future"].float()
                    for key in self.model.predicted_state_streams
                }
                values = distribution_metrics(
                    sample_streams,
                    targets,
                    samples.get("contact_probability"),
                    batch.get("contact_future"),
                )
                state_samples = torch.cat(list(sample_streams.values()), dim=-1)
                state_target = torch.cat(list(targets.values()), dim=-1)
                mean_future = state_samples.mean(dim=1)
                values["deterministic_mean_mse"] = (
                    mean_future - state_target
                ).square().flatten(1).mean(dim=1)
                flat = state_samples.flatten(2)
                medoid_index = torch.cdist(flat, flat).sum(dim=-1).argmin(dim=1)
                medoid = state_samples[
                    torch.arange(state_samples.shape[0], device=state_samples.device),
                    medoid_index,
                ]
                values["deterministic_medoid_mse"] = (
                    medoid - state_target
                ).square().flatten(1).mean(dim=1)
                for name, value in values.items():
                    self._metric_add(
                        accumulator, name, value, value.numel()
                    )

                future_phase = raw_batch.get("future_phase")
                if future_phase is not None:
                    names = (
                        ("free", "precontact", "contact")
                        if self.loss_calculator.contact_state_count == 3
                        else tuple(
                            f"phase_{index}"
                            for index in range(
                                self.loss_calculator.contact_state_count
                            )
                        )
                    )
                    future_phase = future_phase.reshape(-1).round().long()
                    for phase, group in enumerate(names):
                        cpu_mask = future_phase == phase
                        mask = cpu_mask.to(device=state_samples.device)
                        if not mask.any():
                            continue
                        for name in (
                            "energy_score",
                            "min_ade",
                            "sample_spread",
                            "contact_entropy",
                        ):
                            value = values.get(name)
                            if value is not None:
                                self._metric_add(
                                    accumulator,
                                    f"phase_{group}_{name}",
                                    value[mask],
                                    int(mask.sum().item()),
                                )
                accumulator["probabilistic_batches"] += 1
        finally:
            self.model = training_model
        metrics = self._metric_finalize(accumulator)
        metrics["probabilistic_batches"] = int(
            accumulator.get("probabilistic_batches", 0)
        )
        metrics["probabilistic_num_samples"] = self.probabilistic_num_samples
        metrics["probabilistic_epoch"] = int(epoch)
        return metrics

    def validate_one_epoch(self, epoch):
        # First retain the ordinary flow/direct/physics validation diagnostics.
        val_loss = super().validate_one_epoch(epoch)
        if self.rollout_validation_enabled and self.val_loader is not None:
            rollout_metrics = self._run_rollout_validation(epoch)
            self.last_val_epoch_metrics.update(rollout_metrics)
            if self.rollout_replace_val_loss:
                replacement = rollout_metrics.get(
                    self.rollout_replace_val_loss_metric
                )
                if replacement is None:
                    raise ValueError(
                        "rollout_validation.replace_val_loss_metric was not "
                        "produced: "
                        f"{self.rollout_replace_val_loss_metric!r}"
                    )
                val_loss = float(replacement)
        if (
            self.probabilistic_validation_enabled
            and self.val_loader is not None
            and int(epoch) % self.probabilistic_every == 0
        ):
            probability_metrics = self._run_probabilistic_validation(epoch)
            self.last_val_epoch_metrics.update(probability_metrics)
            if self.probabilistic_replace_val_loss and "energy_score" in probability_metrics:
                val_loss = float(probability_metrics["energy_score"])
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
