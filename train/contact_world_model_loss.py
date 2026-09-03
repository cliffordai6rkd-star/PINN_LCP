"""Data losses for the configurable state/contact world model.

``delta_q`` is a real
dataset channel and is supervised directly; it is never reconstructed from a
predicted q trajectory.  The optional kinematic and velocity-smoothness terms
are soft regularizers only.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from model.pinn_model.contact_world_model import PREDICTED_STATE_STREAMS


class ContactWorldModelLoss:
    """Combine flow matching, direct state MSE and contact CE."""

    def __init__(self, config: Mapping):
        self.config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        loss_config = config.get("loss") or {}
        self.joint_dim = int(model_config.get("joint_dim", 7))
        configured_inputs = model_config.get("inputs", PREDICTED_STATE_STREAMS)
        if isinstance(configured_inputs, str):
            configured_inputs = [configured_inputs]
        self.predicted_state_streams = tuple(str(value).lower() for value in configured_inputs)
        self.contact_state_count = int(model_config.get("contact_state_count", 3))
        if self.contact_state_count != 3:
            raise ValueError("model.contact_state_count must be exactly 3")
        self.normalize_mode = data_config.get("normalize_mode")
        self.flow_weight = float(loss_config.get("flow_weight", 1.0))
        self.flow_q_weight = float(loss_config.get("flow_q_weight", 1.0))
        self.flow_dq_weight = float(loss_config.get("flow_dq_weight", 1.0))
        self.flow_delta_q_weight = float(loss_config.get("flow_delta_q_weight", 1.0))
        self.flow_tau_weight = float(loss_config.get("flow_tau_weight", 1.0))
        self.q_weight = float(loss_config.get("q_weight", 1.0))
        self.dq_weight = float(loss_config.get("dq_weight", 1.0))
        self.delta_q_weight = float(loss_config.get("delta_q_weight", 1.0))
        self.tau_weight = float(loss_config.get("tau_weight", 1.0))
        self.contact_weight = float(loss_config.get("contact_weight", 1.0))
        endpoint_config = loss_config.get("endpoint_loss") or {}
        self.endpoint_enabled = bool(endpoint_config.get("enabled", True))
        self.endpoint_initial_weight = float(
            endpoint_config.get("initial_weight", 0.1)
        )
        self.endpoint_final_weight = float(
            endpoint_config.get("final_weight", 0.0)
        )
        self.endpoint_decay_fraction = float(
            endpoint_config.get("decay_fraction", 0.3)
        )
        self.endpoint_schedule = str(
            endpoint_config.get("schedule", "linear")
        ).lower()
        self.kinematic_consistency_weight = float(
            loss_config.get("kinematic_consistency_weight", 0.0)
        )
        configured_joint_scales = loss_config.get("kinematic_joint_scales")
        self.kinematic_joint_scales = (
            None
            if configured_joint_scales is None
            else tuple(float(value) for value in configured_joint_scales)
        )
        self.delta_q_consistency_weight = float(
            loss_config.get("delta_q_consistency_weight", 0.0)
        )
        self.torque_contact_weight = float(
            loss_config.get("torque_contact_weight", 0.0)
        )
        self.ddq_smoothness_weight = float(loss_config.get("ddq_smoothness_weight", 0.0))
        self.ddq_smoothness_warmup_steps = int(
            loss_config.get("ddq_smoothness_warmup_steps", 1000)
        )
        self.ddq_smoothness_huber_delta = float(
            loss_config.get("ddq_smoothness_huber_delta", 1.0)
        )
        self.ddq_smoothness_normalize = bool(
            loss_config.get("ddq_smoothness_normalize", True)
        )
        self.emit_physical_diagnostics = bool(loss_config.get("emit_physical_diagnostics", False))
        self._global_step = 0
        self._total_steps = None
        self._ddq_smoothness_factor = 0.0 if self.ddq_smoothness_warmup_steps > 0 else 1.0
        self.dt = float(
            loss_config.get(
                "dt",
                data_config.get(
                    "state_dt",
                    1.0 / float(data_config.get("high_fps", 100.0)),
                ),
            )
        )
        self.normalizer = None

        contact_config = config.get("contact_gate") or {}
        configured_weights = contact_config.get(
            "class_weights", loss_config.get("contact_class_weights", "auto")
        )
        self.contact_class_weights_is_auto = isinstance(configured_weights, str) and configured_weights.lower() == "auto"
        self.contact_class_weights = None
        if not self.contact_class_weights_is_auto and configured_weights is not None:
            values = tuple(float(value) for value in configured_weights)
            if len(values) != self.contact_state_count:
                raise ValueError(
                    f"contact class weights must contain {self.contact_state_count} values"
                )
            self.contact_class_weights = values
        self._validate()

    def _validate(self):
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise ValueError("loss.dt must be finite and positive")
        if self.ddq_smoothness_warmup_steps < 0:
            raise ValueError("loss.ddq_smoothness_warmup_steps must be non-negative")
        if self.ddq_smoothness_huber_delta <= 0.0 or not math.isfinite(self.ddq_smoothness_huber_delta):
            raise ValueError("loss.ddq_smoothness_huber_delta must be finite and positive")
        weights = {
            name: value
            for name, value in vars(self).items()
            if name.endswith("_weight") and isinstance(value, (int, float))
        }
        invalid = [name for name, value in weights.items() if value < 0.0]
        if invalid:
            raise ValueError(f"loss weights must be non-negative: {invalid}")
        if self.contact_class_weights is not None and any(
            not math.isfinite(value) or value <= 0.0 for value in self.contact_class_weights
        ):
            raise ValueError("contact class weights must be finite and positive")
        if self.kinematic_joint_scales is not None and (
            len(self.kinematic_joint_scales) != self.joint_dim
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.kinematic_joint_scales
            )
        ):
            raise ValueError(
                f"loss.kinematic_joint_scales must contain {self.joint_dim} "
                "finite positive values"
            )
        if not 0.0 < self.endpoint_decay_fraction <= 1.0:
            raise ValueError("endpoint_loss.decay_fraction must be in (0, 1]")
        if self.endpoint_schedule not in {"linear", "cosine"}:
            raise ValueError("endpoint_loss.schedule must be 'linear' or 'cosine'")

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_contact_class_weights(self, values):
        values = tuple(float(value) for value in values)
        if len(values) != self.contact_state_count or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(
                f"contact class weights must contain {self.contact_state_count} positive values"
            )
        self.contact_class_weights = values
        self.contact_class_weights_is_auto = False

    def set_global_step(self, global_step: int, total_steps: int | None = None):
        self._global_step = max(int(global_step), 0)
        if total_steps is not None:
            self._total_steps = max(int(total_steps), 1)
        if self.ddq_smoothness_warmup_steps <= 0:
            self._ddq_smoothness_factor = 1.0
        else:
            self._ddq_smoothness_factor = min(
                1.0, self._global_step / float(self.ddq_smoothness_warmup_steps)
            )

    @property
    def endpoint_weight(self):
        if not self.endpoint_enabled:
            return 0.0
        progress = (
            0.0
            if self._total_steps is None
            else min(self._global_step / float(self._total_steps), 1.0)
        )
        decay_progress = min(progress / self.endpoint_decay_fraction, 1.0)
        if self.endpoint_schedule == "cosine":
            decay_progress = 0.5 - 0.5 * math.cos(math.pi * decay_progress)
        return (
            (1.0 - decay_progress) * self.endpoint_initial_weight
            + decay_progress * self.endpoint_final_weight
        )

    @staticmethod
    def _per_sample_mean(value):
        if value.ndim < 1:
            raise ValueError("per-sample loss input must include a batch dimension")
        return value.reshape(value.shape[0], -1).mean(dim=1)

    @staticmethod
    def _weighted_mean(value, importance_weight=None):
        if value.ndim != 1:
            raise ValueError("weighted loss must have shape [B]")
        if importance_weight is None:
            return value.mean()
        weight = torch.as_tensor(
            importance_weight, device=value.device, dtype=value.dtype
        ).reshape(-1)
        if weight.shape != value.shape:
            raise ValueError("importance_weight must have shape [B]")
        if torch.any(weight < 0) or not torch.isfinite(weight).all():
            raise ValueError("importance_weight must be finite and non-negative")
        return (value * weight).mean()

    @staticmethod
    def _required(batch, key, ndim=3):
        if key not in batch:
            raise KeyError(f"contact world-model loss requires {key!r}")
        value = batch[key]
        if not torch.is_tensor(value) or value.ndim != ndim:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value)
            raise ValueError(f"{key!r} must have {ndim} dimensions, got {shape}")
        return value

    def _physical(self, key, value):
        """Denormalize a tensor for regularizers/diagnostics."""
        if self.normalize_mode is None or self.normalizer is None:
            return value
        functions = {
            "gaussian": self.normalizer.gaussian_denormalize,
            "limit": self.normalizer.limit_denormalize,
            "quantile": self.normalizer.quantile_denormalize,
        }
        if self.normalize_mode not in functions:
            raise ValueError(f"unsupported normalize_mode {self.normalize_mode!r}")
        return functions[self.normalize_mode](key, value)

    def _flow_slices(self):
        slices = {}
        offset = 0
        for key in self.predicted_state_streams:
            slices[key] = slice(offset, offset + self.joint_dim)
            offset += self.joint_dim
        return slices

    def flow_loss_components(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Flow prediction and target shapes differ")
        slices = self._flow_slices()
        losses = {
            key: self._per_sample_mean((prediction[..., sl] - target[..., sl]).square())
            for key, sl in slices.items()
            if key in self.predicted_state_streams
        }
        zero = prediction.new_zeros(prediction.shape[0])
        stream_weights = {
            "q": self.flow_q_weight,
            "dq": self.flow_dq_weight,
            "delta_q": self.flow_delta_q_weight,
            "tau": self.flow_tau_weight,
        }
        total = sum(
            (stream_weights[key] * losses[key] for key in self.predicted_state_streams),
            zero,
        )
        # Contact logits are categorical and are trained only by the data CE
        # below.  They intentionally do not enter continuous flow MSE.
        metric_losses = tuple(losses.get(key, zero) for key in PREDICTED_STATE_STREAMS)
        return total, *metric_losses

    def _direct_losses(self, out, batch):
        result = {}
        for key in self.predicted_state_streams:
            prediction = self._required(out, f"{key}_pred")
            target = self._required(batch, f"{key}_future").to(device=prediction.device, dtype=prediction.dtype)
            if prediction.shape != target.shape:
                raise ValueError(f"{key}_pred and {key}_future shapes differ")
            result[key] = self._per_sample_mean((prediction - target).square())
        return result

    def _contact_loss(self, out, batch, reference):
        if "contact_logits" not in out:
            return reference.new_zeros(reference.shape[0])
        target = self._required(batch, "contact_future").to(device=reference.device)
        labels = target.squeeze(-1).round().long().clamp(0, self.contact_state_count - 1)
        logits = out["contact_logits"]
        if logits.shape[:2] != labels.shape:
            raise ValueError("contact logits and labels have incompatible shapes")
        weight = None
        if self.contact_class_weights is not None:
            weight = torch.as_tensor(self.contact_class_weights, device=logits.device, dtype=logits.dtype)
        frame_loss = F.cross_entropy(
            logits.reshape(-1, self.contact_state_count),
            labels.reshape(-1),
            weight=weight,
            reduction="none",
        ).reshape(logits.shape[:2])
        return frame_loss.mean(dim=1)

    def _kinematic_consistency(self, out, batch):
        if self.kinematic_consistency_weight <= 0.0 or not {"q", "dq"}.issubset(self.predicted_state_streams) or "dq" not in batch:
            reference = out[f"{self.predicted_state_streams[0]}_pred"]
            return reference.new_zeros(reference.shape[0])
        q_future = self._physical("q", self._required(out, "q_pred"))
        dq_future = self._physical("dq", self._required(out, "dq_pred"))
        q_history = self._physical("q", self._required(batch, "q"))
        dq_history = self._physical("dq", self._required(batch, "dq"))
        increments = torch.cat(
            (q_future[:, :1] - q_history[:, -1:], q_future[:, 1:] - q_future[:, :-1]),
            dim=1,
        )
        # Use trapezoidal integration.  The first future interval bridges the
        # last measured velocity and the first predicted velocity; subsequent
        # intervals use adjacent predicted velocities.
        velocity_integral = torch.cat(
            (
                0.5 * self.dt * (dq_history[:, -1:] + dq_future[:, :1]),
                0.5 * self.dt * (dq_future[:, :-1] + dq_future[:, 1:]),
            ),
            dim=1,
        )
        scale = self.kinematic_joint_scales
        if scale is None and self.normalizer is not None:
            stats = getattr(self.normalizer, "stats", {})
            q_stats = stats.get("q") if isinstance(stats, Mapping) else None
            if isinstance(q_stats, Mapping):
                scale = q_stats.get("std")
        if scale is None:
            scale = torch.ones(self.joint_dim, device=q_future.device, dtype=q_future.dtype)
        scale = torch.as_tensor(scale, device=q_future.device, dtype=q_future.dtype)
        scale = scale.reshape(1, 1, -1).clamp_min(1.0e-6)
        residual = (increments - velocity_integral) / scale
        return self._per_sample_mean(residual.square())

    def _ddq_smoothness(self, out):
        if self.ddq_smoothness_weight <= 0.0 or "dq" not in self.predicted_state_streams:
            reference_key = self.predicted_state_streams[0]
            reference = out[f"{reference_key}_pred"]
            return reference.new_zeros(reference.shape[0])
        dq_future = self._physical("dq", self._required(out, "dq_pred"))
        if dq_future.shape[1] < 2:
            return dq_future.new_zeros(dq_future.shape[0])
        ddq = torch.diff(dq_future, dim=1) / self.dt
        # Smoothness is a change-of-acceleration (jerk) penalty.  Penalizing
        # ddq^2 itself would bias the model toward zero acceleration and can
        # erase legitimate high-speed motion; the optional kinematic term is
        # the place where q/dq consistency is enforced.
        if ddq.shape[1] < 2:
            return ddq.new_zeros(ddq.shape[0])
        jerk = torch.diff(ddq, dim=1) / self.dt
        if self.ddq_smoothness_normalize:
            scale = None
            if self.normalizer is not None:
                stats = getattr(self.normalizer, "stats", {})
                dq_stats = stats.get("dq") if isinstance(stats, Mapping) else None
                if isinstance(dq_stats, Mapping):
                    scale = dq_stats.get("std")
            if scale is None:
                scale = dq_future.detach().std(dim=(0, 1), unbiased=False)
            scale = torch.as_tensor(scale, device=jerk.device, dtype=jerk.dtype)
            scale = scale.reshape(1, 1, -1).clamp_min(1.0e-6)
            # jerk has units dq / dt^2.  Scaling by dq_std / dt^2 keeps the
            # regularizer numerically comparable across joints and datasets.
            jerk = jerk * (self.dt * self.dt) / scale
        loss = F.huber_loss(
            jerk,
            torch.zeros_like(jerk),
            delta=self.ddq_smoothness_huber_delta,
            reduction="none",
        )
        return self._per_sample_mean(loss)

    def _delta_q_consistency(self, out, batch):
        reference = out[f"{self.predicted_state_streams[0]}_pred"]
        if self.delta_q_consistency_weight <= 0.0:
            return reference.new_zeros(reference.shape[0])
        if "q_cmd_future" not in batch:
            raise ValueError(
                "delta_q_consistency_weight requires an explicitly aligned "
                "q_cmd_future; expert action is not a joint command"
            )
        q_cmd = self._required(batch, "q_cmd_future").to(
            device=reference.device, dtype=reference.dtype
        )
        expected = q_cmd - self._required(out, "q_pred")
        return self._per_sample_mean(
            (self._required(out, "delta_q_pred") - expected).square()
        )

    def _torque_contact_consistency(self, out, batch):
        reference = out[f"{self.predicted_state_streams[0]}_pred"]
        if self.torque_contact_weight <= 0.0:
            return reference.new_zeros(reference.shape[0])
        if "tau_free_future" not in batch or "tau" not in self.predicted_state_streams:
            raise ValueError(
                "torque_contact_weight requires aligned tau_free_future and tau output"
            )
        tau = self._physical("tau", self._required(out, "tau_pred"))
        tau_free = self._required(batch, "tau_free_future").to(
            device=tau.device, dtype=tau.dtype
        )
        signal = (tau - tau_free).abs().sum(dim=-1)
        labels = self._required(batch, "contact_future").squeeze(-1).round().long()
        gate = self.config.get("contact_gate") or {}
        thresholds = (gate.get("thresholds") or {}).get(
            str(gate.get("metric", "tau_ext_l1")).lower(), {}
        )
        off = float(thresholds.get("off", thresholds.get(False, 0.0)))
        on = float(thresholds.get("on", thresholds.get(True, off)))
        free_loss = torch.relu(signal - off).square()
        contact_loss = torch.relu(on - signal).square()
        loss = torch.where(labels == 0, free_loss, torch.zeros_like(signal))
        loss = torch.where(labels == 2, contact_loss, loss)
        return loss.mean(dim=1)

    def __call__(self, out, batch):
        flow_prediction = out.get("flow_velocity_pred")
        flow_target = out.get("flow_velocity_target")
        if flow_prediction is None or flow_target is None:
            raise KeyError("model output must contain flow velocity prediction and target")
        flow_loss_ps, flow_q_ps, flow_dq_ps, flow_delta_q_ps, flow_tau_ps = self.flow_loss_components(flow_prediction, flow_target)
        direct = self._direct_losses(out, batch)
        contact_loss_ps = self._contact_loss(
            out,
            batch,
            out[f"{self.predicted_state_streams[0]}_pred"],
        )
        kinematic_ps = self._kinematic_consistency(out, batch)
        smoothness_ps = self._ddq_smoothness(out)
        delta_consistency_ps = self._delta_q_consistency(out, batch)
        torque_contact_ps = self._torque_contact_consistency(out, batch)
        importance_weight = batch.get("importance_weight")
        flow_loss = self._weighted_mean(flow_loss_ps, importance_weight)
        direct_loss = sum(
            (
                {"q": self.q_weight, "dq": self.dq_weight,
                 "delta_q": self.delta_q_weight, "tau": self.tau_weight}[key]
                * direct[key]
                for key in self.predicted_state_streams
            ),
            flow_loss_ps.new_zeros(flow_loss_ps.shape),
        )
        endpoint_loss = self._weighted_mean(direct_loss, importance_weight)
        contact_loss = contact_loss_ps.mean()
        kinematic_loss = self._weighted_mean(kinematic_ps, importance_weight)
        smoothness_loss = self._weighted_mean(smoothness_ps, importance_weight)
        delta_consistency_loss = self._weighted_mean(
            delta_consistency_ps, importance_weight
        )
        torque_contact_loss = self._weighted_mean(
            torque_contact_ps, importance_weight
        )
        total = (
            self.flow_weight * flow_loss
            + self.endpoint_weight * endpoint_loss
            + self.contact_weight * contact_loss
            + self.kinematic_consistency_weight * kinematic_loss
            + self.ddq_smoothness_weight * self._ddq_smoothness_factor * smoothness_loss
            + self.delta_q_consistency_weight * delta_consistency_loss
            + self.torque_contact_weight * torque_contact_loss
        )
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": self._weighted_mean(flow_q_ps, importance_weight).detach(),
            "flow_dq_loss": self._weighted_mean(flow_dq_ps, importance_weight).detach(),
            "flow_delta_q_loss": self._weighted_mean(flow_delta_q_ps, importance_weight).detach(),
            "flow_tau_loss": self._weighted_mean(flow_tau_ps, importance_weight).detach(),
            **{
                f"{key}_loss": self._weighted_mean(
                    direct[key], importance_weight
                ).detach()
                for key in self.predicted_state_streams
            },
            "endpoint_loss": endpoint_loss.detach(),
            "endpoint_weight": flow_loss.new_tensor(self.endpoint_weight),
            "contact_loss": contact_loss.detach(),
            "kinematic_consistency_loss": kinematic_loss.detach(),
            "ddq_smoothness_loss": smoothness_loss.detach(),
            "ddq_smoothness_factor": flow_loss.new_tensor(self._ddq_smoothness_factor),
            "delta_q_consistency_loss": delta_consistency_loss.detach(),
            "torque_contact_loss": torque_contact_loss.detach(),
            "importance_weight_mean": (
                flow_loss.new_tensor(1.0)
                if importance_weight is None
                else torch.as_tensor(
                    importance_weight, device=flow_loss.device, dtype=flow_loss.dtype
                ).mean().detach()
            ),
        }
        if self.emit_physical_diagnostics:
            for key in self.predicted_state_streams:
                out[f"{key}_pred_physical"] = self._physical(
                    key, out[f"{key}_pred"]
                )
            if "dq" in self.predicted_state_streams:
                out["ddq_pred_physical"] = (
                    torch.diff(out["dq_pred_physical"], dim=1) / self.dt
                    if out["dq_pred_physical"].shape[1] > 1
                    else out["dq_pred_physical"].new_zeros(
                        out["dq_pred_physical"].shape
                    )
                )
        return total, loss_dict
