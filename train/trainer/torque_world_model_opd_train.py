"""Progressive ODE/Flow distillation for the torque world model.

The Teacher is a frozen multi-step Flow model.  The Student is trained with
the same condition and source trajectory, first matching the Teacher endpoint
and then matching the endpoint again after its own predicted state is written
back into the history.  RNEA is deliberately not part of this trainer unless
``physics.rnea.enabled`` is explicitly set.
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

from data_process.world_model_dataset import compose_action_condition
from physics.nero_dynamics import PinocchioDynamics
from train.trainer.torque_world_model_train import TorqueWorldModelTrainer


log = logging.getLogger(__name__)

_CHECKPOINT_SCORE = re.compile(
    r"_(?:val_loss|train_eval_loss|loss)_"
    r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\.pt$"
)
_CHECKPOINT_STEP = re.compile(r"^step_(\d+)\.pt$")


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
        self.contact_distill_weight = float(
            distill.get("contact_weight", self.distill_weight)
        )
        self.rollout_weight = float(distill.get("rollout_weight", 1.0))
        self.rollout_steps = int(distill.get("rollout_steps", 4))
        curriculum = distill.get("curriculum") or {}
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
        self.teacher_checkpoint_path = distill.get("teacher_checkpoint_path")
        data_config = config.get("dataloader") or {}
        self.action_condition_features = tuple(
            str(value)
            for value in data_config.get("action_condition_features", ())
        )
        self.pose_dynamics = None
        if {"current_ee_pose", "relative_pose"} & set(
            self.action_condition_features
        ):
            self.pose_dynamics = PinocchioDynamics(config)
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
            or self.contact_distill_weight < 0.0
            or self.rollout_weight < 0.0
        ):
            raise ValueError("distillation weights must be non-negative")
        if self.rollout_steps < 0:
            raise ValueError("distillation.rollout_steps must be non-negative")
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
        for candidate in candidates:
            match = _CHECKPOINT_SCORE.search(candidate.name)
            if match is not None:
                scored.append((float(match.group(1)), candidate))
            step_match = _CHECKPOINT_STEP.match(candidate.name)
            if step_match is not None:
                stepped.append((int(step_match.group(1)), candidate))
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
            or self.teacher.wrench_dim != self.model.wrench_dim
            or self.teacher.action_dim != self.model.action_dim
        ):
            raise ValueError(
                "Teacher and Student state contracts differ: "
                f"teacher=(joint={self.teacher.joint_dim}, wrench={self.teacher.wrench_dim}) "
                f"student=(joint={self.model.joint_dim}, wrench={self.model.wrench_dim}, "
                f"action={self.model.action_dim})"
            )
        log.info(
            "loaded frozen Teacher=%s (%d steps), Student=%d steps",
            checkpoint_path,
            self.teacher_steps,
            self.student_steps,
        )

    @staticmethod
    def _state_keys(model):
        """Return state streams that can be committed to model history."""

        keys = ["q", "tau"] if model.q_tau_contact_contract else ["q", "dq", "tau"]
        if model.wrench_dim:
            keys.append("wrench")
        return keys

    def _distillation_terms(self):
        """Return ``(name, output_key, weight)`` terms for either contract."""

        if self.model.q_tau_contact_contract:
            return (
                ("q", "q_pred", 1.0),
                ("tau", "tau_pred", 1.0),
                ("contact", "contact_logits", self.contact_distill_weight),
            )
        terms = [
            ("q", "q_pred", 1.0),
            ("dq", "dq_pred", 1.0),
            ("tau", "tau_pred", 1.0),
        ]
        if self.model.wrench_dim:
            terms.append(("wrench", "wrench_pred", 1.0))
        return tuple(terms)

    def _teacher_predict(self, batch):
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
                )

    def _endpoint_distill(
        self,
        student_batch,
        teacher_batch=None,
        *,
        teacher_out=None,
    ):
        if teacher_batch is None:
            teacher_batch = student_batch
        if teacher_out is None:
            teacher_out = self._teacher_predict(teacher_batch)
        student_out = self.model.predict_differentiable(
            student_batch,
            steps=self.student_steps,
            solver=self.model.flow_solver,
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

    def _refresh_action_condition(self, batch, q_state=None, current_pose=None):
        """Update pose-derived action tokens after a state-history writeback."""

        if self.pose_dynamics is None or "target_pose_abs" not in batch:
            return batch

        if current_pose is None:
            if q_state is None:
                raise ValueError("q_state or current_pose is required")
            # Only the committed first prediction affects the next history
            # condition.  Avoid FK for the remaining prediction horizon.
            if q_state.ndim >= 3:
                q_state = q_state[:, :1]
            q_physical = self.loss_calculator._physical("q", q_state.float())
            current_pose = self.pose_dynamics.frame_poses(q_physical)

        if current_pose.ndim == 3:
            current_pose = current_pose[:, 0]
        if current_pose.ndim != 2 or current_pose.shape[-1] != 7:
            raise ValueError(
                "current_pose must have shape [B, 7] or [B, 1, 7], got "
                f"{tuple(current_pose.shape)}"
            )
        condition_raw = compose_action_condition(
            current_pose,
            batch["target_pose_abs"],
            self.action_condition_features,
        )
        batch["target_relative_pose"] = self.dataset._normalize(
            "target_relative_pose", condition_raw
        )
        if "current_ee_pose" in batch:
            batch["current_ee_pose"] = torch.cat(
                (
                    batch["current_ee_pose"][:, 1:],
                    current_pose[:, None, :].detach(),
                ),
                dim=1,
            )
        return batch

    def _write_back(self, batch, student_out):
        next_batch = dict(batch)
        for key in self._state_keys(self.model):
            history = batch[key]
            prediction = student_out[f"{key}_pred"]
            # One high-rate state is committed per rollout iteration. Detach
            # keeps memory bounded while still exposing Student-state errors.
            next_batch[key] = torch.cat(
                (history[:, 1:], prediction[:, :1].detach()), dim=1
            )
        return self._refresh_action_condition(next_batch, student_out["q_pred"][:, :1])

    def _write_back_real(self, batch, reference_batch, step):
        """Advance Teacher history with the corresponding recorded future state."""

        next_batch = dict(batch)
        for key in self._state_keys(self.model):
            future_key = f"{key}_future"
            if future_key not in reference_batch:
                raise KeyError(
                    f"OPD Teacher rollout requires recorded {future_key!r}"
                )
            history = batch[key]
            recorded = reference_batch[future_key][:, step : step + 1].detach()
            next_batch[key] = torch.cat((history[:, 1:], recorded), dim=1)
        current_pose = None
        if "current_ee_pose_future" in reference_batch:
            # The real future q is static dataset data and its FK was already
            # computed when TorqueWorldModelDataset was initialized.
            current_pose = reference_batch["current_ee_pose_future"][
                :, step : step + 1
            ]
        return self._refresh_action_condition(
            next_batch,
            next_batch["q"][:, -1:],
            current_pose=current_pose,
        )

    def _rollout_distill(self, batch, initial_teacher_out=None):
        if self.rollout_steps == 0:
            return batch["q"].new_zeros(())
        student_current = dict(batch)
        teacher_current = dict(batch)
        losses = []
        for step in range(self.rollout_steps):
            teacher_out = initial_teacher_out if step == 0 else None
            loss, student_out = self._endpoint_distill(
                student_current,
                teacher_current,
                teacher_out=teacher_out,
            )
            losses.append(loss)
            student_current = self._write_back(student_current, student_out)
            teacher_current = self._write_back_real(
                teacher_current,
                batch,
                step,
            )
        return torch.stack(losses).mean()

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

    def compute_loss(self, batch):
        # Keep ordinary future-label supervision as an anchor for the first
        # stage; OPD then supplies the frozen Teacher and on-policy terms.
        base_loss, out = super().compute_loss(batch)
        if not self.distill_enabled:
            return base_loss, out
        # The first rollout Teacher context is identical to the direct
        # endpoint context.  Reuse its frozen target, while recomputing the
        # Student path so training-time dropout behavior remains unchanged.
        teacher_out = self._teacher_predict(batch)
        distill_loss, _ = self._endpoint_distill(batch, teacher_out=teacher_out)
        rollout_loss = self._rollout_distill(
            batch,
            initial_teacher_out=teacher_out,
        )
        progress, teacher_weight, rollout_weight = self._curriculum_state()
        total = (
            base_loss
            + teacher_weight * distill_loss
            + rollout_weight * rollout_loss
        )
        out["loss_dict"].update(
            {
                "distill_loss": distill_loss.detach(),
                "rollout_distill_loss": rollout_loss.detach(),
                "opd_curriculum_progress": progress,
                "opd_teacher_weight": teacher_weight,
                "opd_rollout_weight": rollout_weight,
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
