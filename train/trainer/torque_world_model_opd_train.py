"""Progressive ODE/Flow distillation for the torque world model.

The Teacher is a frozen multi-step Flow model. The Student matches the Teacher
endpoint and then matches it again after its own predicted state is written
back into history. Action chunks stay fixed; an optional frozen tau-free chain
supplies physics-derived rollout contact supervision.
"""

from __future__ import annotations

import argparse
import copy
import logging
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping

import torch
import yaml

from model.pinn_model.contact_gate import (
    ContactGateConfig,
    batched_hysteresis_three_phase_mask,
)
from physics.nero_dynamics import load_tau_other_predictor
from train.trainer.torque_world_model_train import TorqueWorldModelTrainer


log = logging.getLogger(__name__)

_CHECKPOINT_SCORE = re.compile(
    r"_(?:val_loss|train_eval_loss|loss)_"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.pt$"
)
_CHECKPOINT_STEP = re.compile(r"^step_(\d+)\.pt$")
_CHECKPOINT_EPOCH = re.compile(r"^epoch_(\d+)\.pt$")


class TorqueWorldModelOPDTrainer(TorqueWorldModelTrainer):
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
            and int(configured_action_rollout) < max(self.rollout_steps_schedule)
        ):
            raise ValueError(
                "dataloader.action_rollout_horizon must cover the rollout "
                "steps schedule"
            )
        if self.curriculum_epochs <= 0:
            raise ValueError("distillation.curriculum.epochs must be positive")
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

        candidates = sorted((checkpoint_path / "checkpoints").glob("*.pt"))
        if not candidates:
            candidates = sorted(checkpoint_path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                "Teacher checkpoint directory contains no .pt files: "
                f"{checkpoint_path}. Train the Teacher first or set "
                "distillation.teacher_checkpoint_path to an existing .pt "
                "file/directory."
            )

        scored = []
        stepped = []
        scheduled_epochs = []
        for candidate in candidates:
            match = _CHECKPOINT_SCORE.search(candidate.name)
            if match is not None:
                scored.append((float(match.group(1)), candidate))
            step_match = _CHECKPOINT_STEP.match(candidate.name)
            if step_match is not None:
                stepped.append((int(step_match.group(1)), candidate))
            epoch_match = _CHECKPOINT_EPOCH.match(candidate.name)
            if epoch_match is not None:
                scheduled_epochs.append((int(epoch_match.group(1)), candidate))
        if scheduled_epochs:
            return max(scheduled_epochs, key=lambda item: (item[0], item[1].name))[1]
        if stepped:
            return max(stepped, key=lambda item: item[0])[1]
        if scored:
            return min(scored, key=lambda item: item[0])[1]
        return candidates[0]

    @staticmethod
    def _student_model_from_config(config):
        from model.pinn_model.torque_world_model import TorqueWorldModel

        return TorqueWorldModel(config)

    def setup(self):
        super().setup()
        if not self.distill_enabled:
            return
        checkpoint_path = self._resolve_teacher_checkpoint(self.teacher_checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        teacher_config = checkpoint.get("config", self.config)
        teacher_config = copy.deepcopy(dict(teacher_config))
        teacher_config.setdefault("model", {})["flow_inference_steps"] = self.teacher_steps
        self.teacher = self._student_model_from_config(teacher_config).to(self.device)
        state_dict = checkpoint.get("model") or checkpoint.get("model_raw")
        if not isinstance(state_dict, Mapping):
            raise KeyError("Teacher checkpoint does not contain model weights")
        self.teacher.load_state_dict(state_dict, strict=True)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        if (
            self.teacher.flow_dim != self.model.flow_dim
            or self.teacher.joint_dim != self.model.joint_dim
            or self.teacher.action_dim != self.model.action_dim
            or tuple(self.teacher.inputs) != tuple(self.model.inputs)
            or self.teacher.contact_state_count != self.model.contact_state_count
        ):
            raise ValueError(
                "Teacher and Student state contracts differ: "
                f"teacher=(joint={self.teacher.joint_dim}, inputs={self.teacher.inputs}) "
                f"student=(joint={self.model.joint_dim}, inputs={self.model.inputs}, "
                f"action={self.model.action_dim})"
            )
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
                    "rollout contact tau-free history horizon exceeds SWM history: "
                    f"{self.tau_free_predictor.history_horizon} > "
                    f"{self.model.history_horizon}"
                )
            log.info(
                "loaded frozen tau-free rollout model=%s (history=%d)",
                tau_free_path,
                self.tau_free_predictor.history_horizon,
            )

    @staticmethod
    def _state_keys(model):
        """Return state streams that can be committed to model history."""
        del model
        return ["q", "dq", "delta_q", "tau"]

    def _distillation_terms(self):
        """Return continuous state endpoint terms for OPD.

        Contact is intentionally absent: its categorical logits are trained
        from recorded phase labels by the ordinary data loss, not from a
        Teacher/Student regression target.
        """

        terms = [
            ("q", "q_pred", 1.0),
            ("dq", "dq_pred", 1.0),
            ("delta_q", "delta_q_pred", 1.0),
            ("tau", "tau_pred", 1.0),
        ]
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
            zero = student_out["q_pred"].new_zeros(())
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
        """Generate a frozen Teacher target without inheriting Student AMP."""

        # Keep Teacher targets in the original FP32 path even when the Student
        # uses autocast.  This preserves the distillation target exactly while
        # still allowing the trainable model to use mixed precision.
        autocast_context = getattr(self, "autocast_context", None)
        context = (
            autocast_context(enabled=False)
            if autocast_context is not None
            else nullcontext()
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
        for _, output_key, weight in self._distillation_terms():
            if weight == 0.0:
                continue
            losses.append(
                weight
                * torch.nn.functional.mse_loss(
                    student_out[output_key],
                    teacher_out[output_key].detach(),
                )
            )
            weights.append(weight)
        if not losses:
            return student_out["q_pred"].new_zeros(()), student_out
        return torch.stack(losses).sum() / sum(weights), student_out

    def _write_back(
        self,
        batch,
        student_out,
        rollout_step=0,
        *,
        advance_action=True,
    ):
        next_batch = dict(batch)
        for key in self._state_keys(self.model):
            if key not in batch or f"{key}_pred" not in student_out:
                continue
            history = batch[key]
            prediction = student_out[f"{key}_pred"]
            # One high-rate state is committed per rollout iteration. Detach
            # keeps memory bounded while still exposing Student-state errors.
            next_batch[key] = torch.cat(
                (history[:, 1:], prediction[:, :1].detach()), dim=1
            )
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

    def _write_back_real(self, batch, reference_batch, step):
        """Advance a legacy teacher-forced history with recorded state data.

        Guided OPD no longer calls this helper for distillation; it remains for
        compatibility with diagnostics and still advances the action contract
        when an ``action_rollout`` field is available.
        """

        next_batch = dict(batch)
        for key in self._state_keys(self.model):
            future_key = f"{key}_future"
            if key not in batch or future_key not in reference_batch:
                continue
            history = batch[key]
            recorded = reference_batch[future_key][:, step : step + 1].detach()
            next_batch[key] = torch.cat((history[:, 1:], recorded), dim=1)
        if "contact" in batch and "contact_future" in reference_batch:
            next_batch["contact"] = torch.cat(
                (
                    batch["contact"][:, 1:],
                    reference_batch["contact_future"][:, step : step + 1].detach(),
                ),
                dim=1,
            )
        if "action_rollout" in batch:
            next_step = int(step) + 1
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
        initial_teacher_out=None,
        initial_source_noise=None,
        rollout_steps=None,
    ):
        if rollout_steps is None:
            rollout_steps = self.rollout_steps
        rollout_steps = int(rollout_steps)
        if rollout_steps == 0:
            zero = batch["q"].new_zeros(())
            return zero, {
                "rollout_contact_ce": zero.detach(),
                "rollout_contact_physical": zero.detach(),
                "rollout_tau_ext_norm": zero.detach(),
            }
        student_current = dict(batch)
        losses = []
        contact_metrics = []
        for step in range(rollout_steps):
            teacher_out = initial_teacher_out if step == 0 else None
            source_noise = (
                initial_source_noise
                if step == 0 and initial_source_noise is not None
                else self._sample_source_noise(student_current)
            )
            loss, student_out = self._endpoint_distill(
                student_current,
                # Guided OPD relabels the frozen Teacher on the same
                # Student-induced history.  Student write-back is detached,
                # so this changes the state distribution without introducing
                # full-horizon BPTT.
                student_current,
                teacher_out=teacher_out,
                source_noise=source_noise,
            )
            contact_loss, current_contact_metrics = self._rollout_contact_loss(
                student_current, student_out
            )
            losses.append(loss + contact_loss)
            contact_metrics.append(current_contact_metrics)
            student_current = self._write_back(
                student_current,
                student_out,
                rollout_step=step,
                advance_action=step + 1 < rollout_steps,
            )
        metrics = {}
        for key in (
            "rollout_contact_ce",
            "rollout_contact_physical",
            "rollout_tau_ext_norm",
        ):
            metrics[key] = torch.stack([item[key] for item in contact_metrics]).mean()
        return torch.stack(losses).mean(), metrics

    def _curriculum_state(self):
        if not self.curriculum_enabled:
            progress = 1.0
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

    def compute_loss(self, batch):
        # Keep ordinary future-label supervision as an anchor for the first
        # stage; OPD then supplies the frozen Teacher and on-policy terms.
        base_loss, out = super().compute_loss(batch)
        if not self.distill_enabled:
            return base_loss, out
        progress, teacher_weight, rollout_weight = self._curriculum_state()
        rollout_steps = self._rollout_steps_for_progress(progress)
        # The first rollout Teacher context is identical to the direct
        # endpoint context.  Reuse its frozen target, while recomputing the
        # Student path so training-time dropout behavior remains unchanged.
        source_noise = self._sample_source_noise(batch)
        teacher_out = self._teacher_predict(batch, source_noise=source_noise)
        distill_loss, _ = self._endpoint_distill(
            batch,
            teacher_out=teacher_out,
            source_noise=source_noise,
        )
        rollout_loss, rollout_contact_metrics = self._rollout_distill(
            batch,
            initial_teacher_out=teacher_out,
            initial_source_noise=source_noise,
            rollout_steps=rollout_steps,
        )
        total = (
            base_loss
            + teacher_weight * distill_loss
            + rollout_weight * rollout_loss
        )
        out["loss_dict"].update(
            {
                "distill_loss": distill_loss.detach(),
                "rollout_distill_loss": rollout_loss.detach(),
                **{
                    key: value.detach()
                    for key, value in rollout_contact_metrics.items()
                },
                "opd_curriculum_progress": progress,
                "opd_teacher_weight": teacher_weight,
                "opd_rollout_weight": rollout_weight,
                "opd_rollout_steps": float(rollout_steps),
                "opd_total_loss": total.detach(),
            }
        )
        return total, out


def main():
    parser = argparse.ArgumentParser(description="Train a few-step OPD Student")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/torque_world_model_opd.yaml"),
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    trainer = TorqueWorldModelOPDTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
