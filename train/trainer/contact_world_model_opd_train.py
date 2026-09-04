"""Progressive flow distillation for the Contact World Model.

The Teacher is a frozen multi-step Flow model. The Student matches the Teacher
endpoint and then matches it again after its own predicted state is written
back into history. Action chunks stay fixed; an optional frozen tau-free chain
supplies physics-derived rollout contact supervision.
"""

from __future__ import annotations

import argparse
import copy
import logging
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import yaml

from model.pinn_model.contact_world_model import ContactWorldModel, PREDICTED_STATE_STREAMS
from model.pinn_model.contact_gate import (
    ContactGateConfig,
    batched_hysteresis_three_phase_mask,
)
from physics.nero_dynamics import load_tau_other_predictor
from train.trainer.contact_world_model_train import ContactWorldModelTrainer


log = logging.getLogger(__name__)

class ContactWorldModelOPDTrainer(ContactWorldModelTrainer):
    """Train a few-step Student against a frozen multi-step Teacher."""

    def __init__(self, config: Mapping):
        super().__init__(config)
        distill = config.get("distillation") or {}
        self.distill_enabled = bool(distill.get("enabled", True))
        self.teacher_steps = int(
            distill.get(
                "teacher_steps",
                (config.get("model") or {}).get("flow_inference_steps", 8),
            )
        )
        self.student_steps = int(
            distill.get("student_steps", (config.get("model") or {}).get("flow_inference_steps", 2))
        )
        self.distill_weight = float(distill.get("weight", 1.0))
        self.rollout_weight = float(distill.get("rollout_weight", 1.0))
        self.rollout_steps = int(distill.get("rollout_steps", 4))
        self.supervised_depths_per_batch = int(
            distill.get("supervised_depths_per_batch", 1)
        )
        self.contact_distill_temperature = float(
            distill.get("contact_temperature", 2.0)
        )
        self.contact_distill_weight = float(
            distill.get("contact_distill_weight", 1.0)
        )
        self.contact_hard_weight = float(distill.get("contact_hard_weight", 1.0))
        teacher_amp = distill.get("teacher_amp", False)
        self.teacher_amp = False if teacher_amp is False else str(teacher_amp).lower()
        rollout_history = distill.get("rollout_history") or {}
        self.rollout_history_mode = str(
            rollout_history.get("mode", "mixed")
        ).lower()
        self.history_beta_start = float(rollout_history.get("beta_start", 0.0))
        self.history_beta_end = float(rollout_history.get("beta_end", 1.0))
        self.history_curriculum_fraction = float(
            rollout_history.get("curriculum_fraction", 1.0)
        )
        self.depth_sample_counts = defaultdict(int)
        curriculum = distill.get("curriculum") or {}
        configured_rollout_schedule = curriculum.get(
            "rollout_steps_schedule",
            distill.get("rollout_steps_schedule"),
        )
        if configured_rollout_schedule is None:
            configured_rollout_schedule = [self.rollout_steps]
        self.rollout_steps_schedule = tuple(
            int(value) for value in configured_rollout_schedule
        )
        self.curriculum_enabled = bool(curriculum.get("enabled", True))
        configured_curriculum_steps = curriculum.get("optimizer_steps")
        self.curriculum_optimizer_steps = (
            None
            if configured_curriculum_steps is None
            else int(configured_curriculum_steps)
        )
        # Compatibility only for old configs. New CARS-WM configs express the
        # curriculum directly in optimizer updates.
        self.curriculum_epochs = int(curriculum.get("epochs", 100))
        self.teacher_weight_start = float(
            curriculum.get("teacher_weight_start", self.distill_weight)
        )
        self.teacher_weight_end = float(
            curriculum.get("teacher_weight_end", 0.25 * self.distill_weight)
        )
        self.rollout_weight_start = float(
            curriculum.get("rollout_weight_start", 0.0)
        )
        self.rollout_weight_end = float(
            curriculum.get("rollout_weight_end", self.rollout_weight)
        )
        rollout_contact = distill.get("rollout_contact") or {}
        self.rollout_contact_enabled = bool(rollout_contact.get("enabled", False))
        self.rollout_contact_weight = float(
            rollout_contact.get("weight", 0.2)
        )
        self.rollout_contact_physical_weight = float(
            rollout_contact.get("physical_consistency_weight", 0.02)
        )
        self.rollout_contact_horizon = int(
            rollout_contact.get(
                "horizon",
                (config.get("dataloader") or {}).get(
                    "prediction_horizon", 32
                ),
            )
        )
        self.rollout_contact_backfill = bool(
            rollout_contact.get("backfill", False)
        )
        self.rollout_contact_temperature = float(
            rollout_contact.get("temperature", 0.05)
        )
        self.rollout_contact_max_model_batch_size = int(
            rollout_contact.get("max_model_batch_size", 1024)
        )
        self.tau_free_checkpoint_path = rollout_contact.get(
            "tau_free_checkpoint_path"
        )
        self.rollout_contact_gate = ContactGateConfig.from_config(config)
        self.tau_free_predictor = None
        self.teacher_checkpoint_path = distill.get("teacher_checkpoint_path")
        self.teacher = None
        self._validate_distillation_config()

    def _validate_distillation_config(self):
        if not self.distill_enabled:
            return
        if self.teacher_steps <= 0 or self.student_steps <= 0:
            raise ValueError("distillation teacher_steps/student_steps must be positive")
        if self.student_steps >= self.teacher_steps:
            raise ValueError("distillation.student_steps must be smaller than teacher_steps")
        if (
            self.distill_weight < 0.0
            or self.rollout_weight < 0.0
        ):
            raise ValueError("distillation weights must be non-negative")
        if self.rollout_steps < 0:
            raise ValueError("distillation.rollout_steps must be non-negative")
        if self.supervised_depths_per_batch < 1:
            raise ValueError("distillation.supervised_depths_per_batch must be positive")
        if self.contact_distill_temperature <= 0.0:
            raise ValueError("distillation.contact_temperature must be positive")
        if min(self.contact_distill_weight, self.contact_hard_weight) < 0.0:
            raise ValueError("contact distillation weights must be non-negative")
        if self.teacher_amp not in {False, "bf16", "bfloat16"}:
            raise ValueError("distillation.teacher_amp must be false or bf16")
        if self.rollout_history_mode not in {"recorded", "student", "mixed"}:
            raise ValueError(
                "distillation.rollout_history.mode must be recorded, student, or mixed"
            )
        if not 0.0 <= self.history_beta_start <= 1.0 or not 0.0 <= self.history_beta_end <= 1.0:
            raise ValueError("rollout history beta values must be in [0, 1]")
        if not 0.0 < self.history_curriculum_fraction <= 1.0:
            raise ValueError("rollout_history.curriculum_fraction must be in (0, 1]")
        if not self.rollout_steps_schedule or any(
            value < 0 for value in self.rollout_steps_schedule
        ):
            raise ValueError(
                "distillation.curriculum.rollout_steps_schedule must contain "
                "non-negative integers"
            )
        if any(
            current > following
            for current, following in zip(
                self.rollout_steps_schedule, self.rollout_steps_schedule[1:]
            )
        ):
            raise ValueError(
                "distillation.curriculum.rollout_steps_schedule must be "
                "non-decreasing"
            )
        configured_future = int(
            (self.config.get("dataloader") or {}).get(
                "prediction_horizon",
                (self.config.get("dataloader") or {}).get("future_horizon", 40),
            )
        )
        if max(self.rollout_steps, *self.rollout_steps_schedule) > configured_future:
            raise ValueError(
                "distillation rollout steps must not exceed prediction_horizon"
            )
        configured_action_rollout = (self.config.get("dataloader") or {}).get(
            "action_rollout_horizon"
        )
        if (
            configured_action_rollout is not None
            and int(configured_action_rollout) <= max(self.rollout_steps_schedule)
        ):
            raise ValueError(
                "dataloader.action_rollout_horizon must cover the rollout "
                "depth plus its re-anchored supervised action condition"
            )
        if (
            self.curriculum_optimizer_steps is not None
            and self.curriculum_optimizer_steps <= 0
        ):
            raise ValueError(
                "distillation.curriculum.optimizer_steps must be positive"
            )
        if self.curriculum_optimizer_steps is None and self.curriculum_epochs <= 0:
            raise ValueError(
                "distillation.curriculum optimizer_steps/epochs must be positive"
            )
        if min(
            self.teacher_weight_start,
            self.teacher_weight_end,
            self.rollout_weight_start,
            self.rollout_weight_end,
        ) < 0.0:
            raise ValueError("distillation curriculum weights must be non-negative")
        if not self.teacher_checkpoint_path:
            raise ValueError("distillation.teacher_checkpoint_path is required")
        if self.rollout_contact_weight < 0.0 or self.rollout_contact_physical_weight < 0.0:
            raise ValueError("distillation.rollout_contact weights must be non-negative")
        if self.rollout_contact_horizon <= 0 or self.rollout_contact_horizon > configured_future:
            raise ValueError(
                "distillation.rollout_contact.horizon must be in "
                f"[1, {configured_future}]"
            )
        if self.rollout_contact_temperature <= 0.0:
            raise ValueError(
                "distillation.rollout_contact.temperature must be positive"
            )
        if self.rollout_contact_max_model_batch_size <= 0:
            raise ValueError(
                "distillation.rollout_contact.max_model_batch_size must be positive"
            )
        if self.rollout_contact_enabled:
            required_streams = {"q", "dq", "delta_q", "tau"}
            configured_streams = {
                str(value).lower()
                for value in (self.config.get("model") or {}).get("inputs", ())
            }
            if not required_streams.issubset(configured_streams):
                raise ValueError(
                    "rollout contact supervision requires model.inputs to contain "
                    "q, dq, delta_q, and tau"
                )
            configured_contact_states = int(
                (self.config.get("model") or {}).get("contact_state_count", 3)
            )
            if configured_contact_states != 3:
                raise ValueError(
                    "rollout contact supervision requires contact_state_count=3"
                )
            if not self.rollout_contact_gate.enabled:
                raise ValueError(
                    "rollout contact supervision requires contact_gate.enabled=true"
                )
            if self.rollout_contact_gate.label_mode != "three_phase":
                raise ValueError(
                    "rollout contact supervision requires contact_gate.label_mode=three_phase"
                )
            if self.rollout_contact_gate.metric not in {"tau_ext_l1", "tau_ext_l2"}:
                raise ValueError(
                    "rollout contact supervision requires contact_gate.metric="
                    "tau_ext_l1 or tau_ext_l2"
                )
            if not self.tau_free_checkpoint_path:
                raise ValueError(
                    "distillation.rollout_contact.tau_free_checkpoint_path is required "
                    "when rollout contact supervision is enabled"
                )
            if (
                self.rollout_contact_weight == 0.0
                and self.rollout_contact_physical_weight == 0.0
            ):
                raise ValueError(
                    "rollout contact supervision needs a positive CE or physical weight"
                )

    def build_model(self):
        student_config = copy.deepcopy(self.config)
        student_config.setdefault("model", {})["flow_inference_steps"] = self.student_steps
        return self._student_model_from_config(student_config)

    @staticmethod
    def _resolve_teacher_checkpoint(configured_path):
        """Resolve a Teacher file or checkpoint directory before loading it.

        Training configs are normally run from the repository root, but a
        rendered config may be launched from another working directory.  Keep
        relative paths useful in both cases and report the missing prerequisite
        before ``torch.load`` emits a low-level file error.
        """

        raw_path = Path(configured_path).expanduser()
        candidates = [raw_path]
        if not raw_path.is_absolute():
            repository_root = Path(__file__).resolve().parents[2]
            candidates.append(repository_root / raw_path)

        checkpoint_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            None,
        )
        if checkpoint_path is None:
            expected = raw_path if raw_path.is_absolute() else candidates[0]
            raise FileNotFoundError(
                "Teacher checkpoint path does not exist: "
                f"{expected}. Train the Teacher first or set "
                "distillation.teacher_checkpoint_path to an existing .pt "
                "file/directory. The standard workflow is "
                "`bash scripts/train_contact_wm_opd_sweep.sh`."
            )

        if checkpoint_path.is_file():
            return checkpoint_path

        candidates = sorted((checkpoint_path / "checkpoints").glob("epoch_*.pt"))
        if not candidates:
            candidates = sorted(checkpoint_path.glob("epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                "Teacher checkpoint directory contains no .pt files: "
                f"{checkpoint_path}. Train the Teacher first or set "
                "distillation.teacher_checkpoint_path to an existing .pt "
                "file/directory."
            )

        return max(candidates, key=lambda item: item.name)

    @staticmethod
    def _student_model_from_config(config):
        from model.pinn_model.contact_world_model import ContactWorldModel

        return ContactWorldModel(config)

    def setup(self):
        super().setup()
        if not self.distill_enabled:
            return
        checkpoint_path = self._resolve_teacher_checkpoint(self.teacher_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        teacher_config = checkpoint.get("config", self.config)
        teacher_config = copy.deepcopy(dict(teacher_config))
        if checkpoint.get("model_version") != ContactWorldModel.MODEL_VERSION:
            raise ValueError(
                "Teacher checkpoint is not a compatible Contact World Model checkpoint; "
                "retrain the Teacher with the canonical model"
            )
        self._validate_teacher_contract(teacher_config, checkpoint)
        ContactWorldModel(teacher_config).validate_checkpoint_contract(
            checkpoint.get("carswm_contract")
        )
        teacher_config.setdefault("model", {})["flow_inference_steps"] = self.teacher_steps
        self.teacher = self._student_model_from_config(teacher_config).to(self.device)
        state_dict = checkpoint.get("model") or checkpoint.get("model_raw")
        if not isinstance(state_dict, Mapping):
            raise KeyError("Teacher checkpoint does not contain model weights")
        if not any(str(key).startswith("state_encoders.") for key in state_dict):
            raise ValueError(
                "Teacher checkpoint does not contain independent state encoders; "
                "retrain it with ContactWorldModel"
            )
        self.teacher.load_state_dict(state_dict, strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        if self.teacher.flow_dim != self.model.flow_dim:
            raise ValueError("Teacher and Student output contracts differ")
        log.info(
            "loaded frozen Teacher=%s (%d steps), Student=%d steps",
            checkpoint_path,
            self.teacher_steps,
            self.student_steps,
        )
        if self.rollout_contact_enabled:
            tau_free_path = self._resolve_teacher_checkpoint(
                self.tau_free_checkpoint_path
            )
            self.tau_free_predictor = load_tau_other_predictor(
                tau_free_path,
                device=self.device,
                max_model_batch_size=self.rollout_contact_max_model_batch_size,
            )
            expected_inputs = ("q", "dq", "delta_q")
            if tuple(self.tau_free_predictor.active_inputs) != expected_inputs:
                raise ValueError(
                    "rollout contact tau-free checkpoint must use model.inputs="
                    f"{list(expected_inputs)}, got "
                    f"{list(self.tau_free_predictor.active_inputs)}"
                )
            if self.tau_free_predictor.history_horizon > self.model.history_horizon:
                raise ValueError(
                    "rollout contact tau-free history horizon exceeds model history: "
                    f"{self.tau_free_predictor.history_horizon} > "
                    f"{self.model.history_horizon}"
                )
            log.info(
                "loaded frozen tau-free rollout model=%s (history=%d)",
                tau_free_path,
                self.tau_free_predictor.history_horizon,
            )

    def _validate_teacher_contract(self, teacher_config, checkpoint):
        student_model = self.config.get("model") or {}
        teacher_model = teacher_config.get("model") or {}
        student_data = self.config.get("dataloader") or {}
        teacher_data = teacher_config.get("dataloader") or {}
        fields = {
            "model.inputs": (tuple(teacher_model.get("inputs", ())), tuple(student_model.get("inputs", ()))),
            "model.joint_dim": (teacher_model.get("joint_dim"), student_model.get("joint_dim")),
            "model.action_dim": (teacher_model.get("action_dim"), student_model.get("action_dim")),
            "model.contact_state_count": (teacher_model.get("contact_state_count"), student_model.get("contact_state_count")),
            "model.flow_solver": (teacher_model.get("flow_solver", "heun"), student_model.get("flow_solver", "heun")),
            "model.flow_source_mode": (teacher_model.get("flow_source_mode", "gaussian"), student_model.get("flow_source_mode", "gaussian")),
            "dataloader.action_key": (teacher_data.get("action_key"), student_data.get("action_key")),
            "dataloader.action_condition_horizon": (teacher_data.get("action_condition_horizon"), student_data.get("action_condition_horizon")),
            # The offset is part of the temporal action contract: changing it
            # from 0 to 1 shifts every condition window by one expert token.
            "dataloader.action_start_offset": (teacher_data.get("action_start_offset", 1), student_data.get("action_start_offset", 1)),
            "train_data.action_alignment": ((teacher_config.get("train_data") or {}).get("action_alignment"), (self.config.get("train_data") or {}).get("action_alignment")),
            "dataloader.normalize_mode": (teacher_data.get("normalize_mode"), student_data.get("normalize_mode")),
        }
        mismatches = {key: value for key, value in fields.items() if value[0] != value[1]}
        normalizer = checkpoint.get("normalizer")
        if normalizer is None:
            mismatches["checkpoint.normalizer"] = ("present", "missing")
        else:
            teacher_stats = normalizer.get("stats") if isinstance(normalizer, Mapping) else None
            student_stats = getattr(getattr(self, "dataset", None), "normalizer", None)
            student_stats = getattr(student_stats, "stats", None)
            if not isinstance(teacher_stats, Mapping) or not isinstance(student_stats, Mapping):
                mismatches["normalizer.stats"] = (type(teacher_stats).__name__, type(student_stats).__name__)
            else:
                if set(teacher_stats) != set(student_stats):
                    mismatches["normalizer.stats.keys"] = (sorted(teacher_stats), sorted(student_stats))
                else:
                    for key in teacher_stats:
                        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
                            if statistic not in teacher_stats[key] or statistic not in student_stats[key]:
                                mismatches[f"normalizer.stats.{key}.{statistic}"] = "missing"
                                break
                            left = torch.as_tensor(teacher_stats[key][statistic])
                            right = torch.as_tensor(student_stats[key][statistic]).cpu()
                            if left.shape != right.shape or not torch.allclose(left, right, rtol=1.0e-5, atol=1.0e-6):
                                mismatches[f"normalizer.stats.{key}.{statistic}"] = "different"
                                break
        if mismatches:
            raise ValueError(f"Teacher/Student action/state contract mismatch: {mismatches}")

    @staticmethod
    def _state_keys(model):
        """Return state streams that can be committed to model history."""
        if model is None:
            return list(PREDICTED_STATE_STREAMS)
        return list(model.predicted_state_streams)

    def _distillation_terms(self):
        """Return continuous state endpoint terms for OPD.

        Contact logits are handled separately with temperature-scaled KL.
        """

        terms = [(key, f"{key}_pred", 1.0) for key in self.model.predicted_state_streams]
        return tuple(terms)

    def _rollout_contact_signal(self, tau_ext):
        metric = self.rollout_contact_gate.metric
        if metric == "tau_ext_l2":
            return torch.linalg.vector_norm(tau_ext, dim=-1)
        if metric == "tau_ext_l1":
            return tau_ext.abs().sum(dim=-1)
        raise RuntimeError(
            "rollout contact signal requires tau_ext_l1 or tau_ext_l2"
        )

    def _rollout_contact_labels(self, signal):
        """Apply the configured hysteresis to each predicted signal window.

        Hysteresis labels are intentionally detached pseudo-labels.  The
        differentiable physical consistency term below is what sends a
        gradient back into the predicted state trajectory.
        """

        return batched_hysteresis_three_phase_mask(
            signal.detach().to(dtype=torch.float32),
            on_threshold=self.rollout_contact_gate.on_threshold,
            off_threshold=self.rollout_contact_gate.off_threshold,
            consecutive_frames=self.rollout_contact_gate.consecutive_frames,
            backfill=self.rollout_contact_backfill,
        )

    def _rollout_contact_loss(self, batch, student_out):
        """Generate physics-derived contact supervision for one Student rollout.

        The frozen tau-free model remains differentiable with respect to its
        q/dq/delta_q inputs.  Its parameters are frozen, but this method is
        deliberately not wrapped in ``no_grad`` so the optional physical term
        can constrain the Student state trajectory.
        """

        if not self.rollout_contact_enabled:
            zero = student_out[f"{self.model.predicted_state_streams[0]}_pred"].new_zeros(())
            return zero, {
                "rollout_contact_ce": zero.detach(),
                "rollout_contact_physical": zero.detach(),
                "rollout_tau_ext_norm": zero.detach(),
            }
        if self.tau_free_predictor is None:
            raise RuntimeError(
                "rollout contact is enabled but tau-free predictor is not loaded"
            )

        horizon = self.rollout_contact_horizon

        def physical(key, value):
            # The tau-free checkpoint is a FP32 model.  Converting here keeps
            # its input contract stable under Student AMP while preserving the
            # autograd path from the physical loss to Student predictions.
            return self.loss_calculator._physical(key, value).to(dtype=torch.float32)

        history = {
            key: physical(key, batch[key])
            for key in ("q", "dq", "delta_q")
        }
        future = {
            key: physical(key, student_out[f"{key}_pred"][:, :horizon])
            for key in ("q", "dq", "delta_q")
        }
        tau_pred = physical("tau", student_out["tau_pred"][:, :horizon])
        autocast_context = getattr(self, "autocast_context", None)
        context = (
            autocast_context(enabled=False)
            if autocast_context is not None
            else nullcontext()
        )
        with context:
            tau_free = self.tau_free_predictor(history, future)
        tau_ext = tau_pred - tau_free
        signal = self._rollout_contact_signal(tau_ext)
        labels = self._rollout_contact_labels(signal)
        logits = student_out["contact_logits"][:, :horizon]
        if logits.shape[-1] != 3:
            raise ValueError(
                "rollout contact supervision expects three contact logits"
            )
        class_weight = self.loss_calculator.contact_class_weights
        weight = None
        if class_weight is not None:
            weight = torch.as_tensor(
                class_weight, device=logits.device, dtype=logits.dtype
            )
        rollout_ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1).long(), weight=weight
        )
        # Detach the contact head target so this term updates the continuous
        # state trajectory without creating a self-reinforcing head/state loop.
        # A margin form is used instead of only matching saturated sigmoids:
        # it continues to provide a useful gradient when a predicted signal is
        # far outside the configured free/contact bands.
        head_probabilities = torch.softmax(logits, dim=-1).detach().to(
            dtype=signal.dtype
        )
        threshold_gap = signal.new_tensor(
            max(
                self.rollout_contact_gate.on_threshold
                - self.rollout_contact_gate.off_threshold,
                1.0e-6,
            )
        )
        temperature = signal.new_tensor(self.rollout_contact_temperature)
        smooth_scale = temperature / threshold_gap
        log_two = signal.new_tensor(0.6931471805599453)

        def smooth_hinge(value):
            return smooth_scale * torch.relu(
                torch.nn.functional.softplus(value / temperature) - log_two
            )

        free_violation = smooth_hinge(
            signal - self.rollout_contact_gate.off_threshold
        )
        contact_violation = smooth_hinge(
            self.rollout_contact_gate.on_threshold - signal
        )
        align_violation = smooth_scale * (
            torch.relu(
                torch.nn.functional.softplus(
                    (self.rollout_contact_gate.off_threshold - signal)
                    / temperature
                )
                - log_two
            )
            + torch.relu(
                torch.nn.functional.softplus(
                    (signal - self.rollout_contact_gate.on_threshold)
                    / temperature
                )
                - log_two
            )
        )
        physical_consistency = (
            head_probabilities[..., 0] * free_violation.square()
            + head_probabilities[..., 1] * align_violation.square()
            + head_probabilities[..., 2] * contact_violation.square()
        ).mean()
        total = (
            self.rollout_contact_weight * rollout_ce
            + self.rollout_contact_physical_weight * physical_consistency
        )
        metrics = {
            "rollout_contact_ce": rollout_ce.detach(),
            "rollout_contact_physical": physical_consistency.detach(),
            "rollout_tau_ext_norm": signal.detach().mean(),
        }
        return total, metrics

    def _sample_source_noise(self, batch):
        reference = batch[self.model.inputs[0]]
        return torch.randn(
            reference.shape[0],
            self.model.future_horizon,
            self.model.flow_dim,
            device=reference.device,
            dtype=reference.dtype,
        )

    def _teacher_predict(self, batch, *, source_noise=None):
        """Generate one frozen Teacher target at the requested condition."""

        # Keep Teacher targets in the original FP32 path even when the Student
        # uses autocast.  This preserves the distillation target exactly while
        # still allowing the trainable model to use mixed precision.
        autocast_context = getattr(self, "autocast_context", None)
        use_amp = bool(getattr(self, "teacher_amp", False)) and str(
            getattr(self, "device", "cpu")
        ).startswith("cuda")
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else (
                autocast_context(enabled=False)
                if autocast_context is not None
                else nullcontext()
            )
        )
        with context:
            with torch.no_grad():
                return self.teacher.predict(
                    batch,
                    steps=self.teacher_steps,
                    solver=self.teacher.flow_solver,
                    source_noise=source_noise,
                )

    def _endpoint_distill(
        self,
        student_batch,
        teacher_batch=None,
        *,
        teacher_out=None,
        source_noise=None,
    ):
        if teacher_batch is None:
            teacher_batch = student_batch
        if source_noise is None:
            source_noise = self._sample_source_noise(student_batch)
        if teacher_out is None:
            teacher_out = self._teacher_predict(
                teacher_batch, source_noise=source_noise
            )
        student_out = self.model.predict_differentiable(
            student_batch,
            steps=self.student_steps,
            solver=self.model.flow_solver,
            source_noise=source_noise,
        )
        losses = []
        weights = []
        importance = student_batch.get("importance_weight")
        for _, output_key, weight in self._distillation_terms():
            if weight == 0.0:
                continue
            per_sample = self.loss_calculator._per_sample_mean(
                (
                    student_out[output_key]
                    - teacher_out[output_key].detach().to(
                        dtype=student_out[output_key].dtype
                    )
                ).square()
            )
            losses.append(
                weight
                * self.loss_calculator._weighted_mean(per_sample, importance)
            )
            weights.append(weight)
        reference = student_out[f"{self.model.predicted_state_streams[0]}_pred"]
        continuous_loss = (
            torch.stack(losses).sum() / sum(weights)
            if losses
            else reference.new_zeros(())
        )
        temperature = float(getattr(self, "contact_distill_temperature", 2.0))
        student_logits = student_out["contact_logits"]
        teacher_logits = teacher_out["contact_logits"].detach().to(
            dtype=student_logits.dtype
        )
        teacher_probability = torch.softmax(teacher_logits / temperature, dim=-1)
        contact_kl = torch.nn.functional.kl_div(
            torch.log_softmax(student_logits / temperature, dim=-1),
            teacher_probability,
            reduction="batchmean",
        ) * (temperature * temperature) / student_logits.shape[1]
        hard_ce = reference.new_zeros(())
        if "contact_future" in student_batch:
            labels = student_batch["contact_future"].squeeze(-1).round().long()
            hard_ce = torch.nn.functional.cross_entropy(
                student_logits.reshape(-1, student_logits.shape[-1]),
                labels.reshape(-1),
            )
        total = (
            continuous_loss
            + float(getattr(self, "contact_distill_weight", 1.0)) * contact_kl
            + float(getattr(self, "contact_hard_weight", 1.0)) * hard_ce
        )
        student_out["distill_metrics"] = {
            "continuous_distill_loss": continuous_loss.detach(),
            "contact_distill_kl": contact_kl.detach(),
            "contact_distill_hard_ce": hard_ce.detach(),
        }
        return total, student_out

    def _write_back(
        self,
        batch,
        student_out,
        rollout_step=0,
        *,
        advance_action=True,
        recorded_batch=None,
        history_beta=1.0,
    ):
        next_batch = dict(batch)
        for key in self._state_keys(self.model):
            if key not in batch or f"{key}_pred" not in student_out:
                continue
            history = batch[key]
            prediction = student_out[f"{key}_pred"]
            # One high-rate state is committed per rollout iteration. Detach
            # keeps memory bounded while still exposing Student-state errors.
            committed = prediction[:, :1].detach()
            if recorded_batch is not None and f"{key}_future" in recorded_batch:
                depth = int(rollout_step)
                recorded = recorded_batch[f"{key}_future"][:, depth : depth + 1]
                if recorded.shape[1] != 1:
                    raise ValueError("recorded future does not cover rollout depth")
                committed = (
                    (1.0 - float(history_beta)) * recorded
                    + float(history_beta) * committed
                )
            next_batch[key] = torch.cat((history[:, 1:], committed), dim=1)
        contact_prediction = student_out.get("contact_state_pred")
        if contact_prediction is None and "contact_logits" in student_out:
            contact_prediction = student_out["contact_logits"].argmax(
                dim=-1, keepdim=True
            ).to(dtype=batch["contact"].dtype) if "contact" in batch else None
        if "contact" in batch and contact_prediction is not None:
            next_batch["contact"] = torch.cat(
                (batch["contact"][:, 1:], contact_prediction[:, :1].detach()),
                dim=1,
            )
        for target_key in (
            *(f"{key}_future" for key in self._state_keys(self.model)),
            "contact_future",
        ):
            target = batch.get(target_key)
            if torch.is_tensor(target) and target.ndim >= 2 and target.shape[1] > 0:
                next_batch[target_key] = torch.cat(
                    (target[:, 1:], target[:, -1:]), dim=1
                )
        # Direct actions are low-rate chunks, but each OPD state rollout must
        # use the chunk re-anchored at its new high-rate state.  The dataset
        # supplies these chunks as [B, R, A, D], where entry zero is the
        # current condition and entry ``rollout_step + 1`` is the next one.
        if advance_action and "action_rollout" in batch:
            next_step = int(rollout_step) + 1
            action_rollout = batch["action_rollout"]
            if action_rollout.ndim != 4 or next_step >= action_rollout.shape[1]:
                raise ValueError(
                    "action_rollout does not contain the next OPD state chunk"
                )
            next_batch["action"] = action_rollout[:, next_step]
            if "action_rollout_mask" in batch:
                next_batch["action_mask"] = batch["action_rollout_mask"][:, next_step]
        return next_batch

    def _rollout_distill(
        self,
        batch,
        rollout_steps=None,
        sampled_depth=None,
        history_beta=1.0,
    ):
        if rollout_steps is None:
            rollout_steps = self.rollout_steps
        rollout_steps = int(rollout_steps)
        if rollout_steps == 0:
            zero = batch[self.model.predicted_state_streams[0]].new_zeros(())
            return zero, {
                "rollout_contact_ce": zero.detach(),
                "rollout_contact_physical": zero.detach(),
                "rollout_tau_ext_norm": zero.detach(),
            }
        depth = (
            int(sampled_depth)
            if sampled_depth is not None
            else int(
                torch.randint(1, rollout_steps + 1, (), device="cpu").item()
            )
        )
        if not 1 <= depth <= rollout_steps:
            raise ValueError("sampled rollout depth must be in [1, rollout_steps]")
        student_current = dict(batch)
        for step in range(depth):
            # Intermediate states create the Student-induced condition only.
            # They are detached, and no Teacher/contact pseudo-label is called.
            student_out = self.model.predict(
                student_current,
                steps=self.student_steps,
                solver=self.model.flow_solver,
            )
            student_current = self._write_back(
                student_current,
                student_out,
                rollout_step=step,
                advance_action=True,
                recorded_batch=batch,
                history_beta=history_beta,
            )
        source_noise = self._sample_source_noise(student_current)
        loss, supervised_out = self._endpoint_distill(
            student_current,
            teacher_batch=student_current,
            source_noise=source_noise,
        )
        contact_loss, metrics = self._rollout_contact_loss(
            student_current, supervised_out
        )
        metrics.update(supervised_out.get("distill_metrics") or {})
        metrics["sampled_depth"] = float(depth)
        if not hasattr(self, "depth_sample_counts"):
            self.depth_sample_counts = defaultdict(int)
        self.depth_sample_counts[depth] += 1
        return loss + contact_loss, metrics

    def _curriculum_state(self):
        if not self.curriculum_enabled:
            progress = 1.0
        else:
            if self.curriculum_optimizer_steps is not None:
                total_steps = self.curriculum_optimizer_steps
            else:
                batches_per_epoch = max(len(self.loader), 1) if self.loader is not None else 1
                total_steps = max(self.curriculum_epochs * batches_per_epoch, 1)
            progress = min(max(self.global_step / total_steps, 0.0), 1.0)
        teacher_weight = (
            (1.0 - progress) * self.teacher_weight_start
            + progress * self.teacher_weight_end
        )
        rollout_weight = (
            (1.0 - progress) * self.rollout_weight_start
            + progress * self.rollout_weight_end
        )
        return progress, teacher_weight, rollout_weight

    def _rollout_steps_for_progress(self, progress):
        """Select the configured non-decreasing rollout-depth stage."""

        schedule = self.rollout_steps_schedule
        if not self.curriculum_enabled:
            return int(self.rollout_steps)
        if len(schedule) == 1:
            return int(schedule[-1])
        stage = min(int(float(progress) * len(schedule)), len(schedule) - 1)
        return int(schedule[stage])

    def _history_beta(self, progress):
        if self.rollout_history_mode == "recorded":
            return 0.0
        if self.rollout_history_mode == "student":
            return 1.0
        fraction = min(float(progress) / self.history_curriculum_fraction, 1.0)
        return (
            (1.0 - fraction) * self.history_beta_start
            + fraction * self.history_beta_end
        )

    def _sample_supervision_depths(self, maximum_depth):
        maximum_depth = int(maximum_depth)
        if maximum_depth <= 0:
            return ()
        count = min(self.supervised_depths_per_batch, maximum_depth)
        return tuple(
            int(value) + 1
            for value in torch.randperm(maximum_depth, device="cpu")[:count].tolist()
        )

    def compute_loss(self, batch):
        # Keep ordinary future-label supervision as an anchor for the first
        # stage; OPD then supplies the frozen Teacher and on-policy terms.
        base_loss, out = super().compute_loss(batch)
        if not self.distill_enabled or not self.model.training:
            return base_loss, out
        progress, teacher_weight, rollout_weight = self._curriculum_state()
        rollout_steps = self._rollout_steps_for_progress(progress)
        history_beta = self._history_beta(progress)
        # Depth zero is computed exactly once on the recorded condition.
        source_noise = self._sample_source_noise(batch)
        teacher_out = self._teacher_predict(batch, source_noise=source_noise)
        distill_loss, direct_out = self._endpoint_distill(
            batch,
            teacher_out=teacher_out,
            source_noise=source_noise,
        )
        depths = self._sample_supervision_depths(rollout_steps)
        rollout_losses = []
        rollout_metrics = []
        for depth in depths:
            current_loss, current_metrics = self._rollout_distill(
                batch,
                rollout_steps=rollout_steps,
                sampled_depth=depth,
                history_beta=history_beta,
            )
            rollout_losses.append(current_loss)
            rollout_metrics.append(current_metrics)
        rollout_loss = (
            torch.stack(rollout_losses).mean()
            if rollout_losses
            else base_loss.new_zeros(())
        )
        total = (
            base_loss
            + teacher_weight * distill_loss
            + rollout_weight * rollout_loss
        )
        aggregated_rollout_metrics = {}
        metric_keys = {
            key for item in rollout_metrics for key in item
        }
        for key in metric_keys:
            values = [item[key] for item in rollout_metrics if key in item]
            tensors = [
                value
                if torch.is_tensor(value)
                else base_loss.new_tensor(float(value))
                for value in values
            ]
            aggregated_rollout_metrics[key] = torch.stack(tensors).mean().detach()
        out["loss_dict"].update(
            {
                "distill_loss": distill_loss.detach(),
                "rollout_distill_loss": rollout_loss.detach(),
                **direct_out.get("distill_metrics", {}),
                **aggregated_rollout_metrics,
                "opd_curriculum_progress": progress,
                "opd_teacher_weight": teacher_weight,
                "opd_rollout_weight": rollout_weight,
                "opd_rollout_steps": float(rollout_steps),
                "opd_history_beta": float(history_beta),
                "opd_teacher_calls": float(1 + len(depths)),
                "opd_total_loss": total.detach(),
            }
        )
        for depth, count in self.depth_sample_counts.items():
            out["loss_dict"][f"opd_depth_{depth}_sample_count"] = float(count)
        return total, out


def main():
    parser = argparse.ArgumentParser(description="Train a few-step OPD Student")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/contact_world_model_opd.yaml"),
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    trainer = ContactWorldModelOPDTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
