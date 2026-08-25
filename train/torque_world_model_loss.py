"""Data losses for the q/dq/delta_q/tau/contact world model.

There is deliberately no dynamics/RNEA branch here.  ``delta_q`` is a real
dataset channel and is supervised directly; it is never reconstructed from a
predicted q trajectory.  The optional kinematic and velocity-smoothness terms
are soft regularizers only.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F


STATE_KEYS = ("q", "dq", "delta_q", "tau")


class TorqueWorldModelLoss:
    """Combine flow matching, direct state MSE and contact CE."""

    def __init__(self, config: Mapping):
        self.config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        loss_config = config.get("loss") or {}
        self.joint_dim = int(model_config.get("joint_dim", 7))
        self.contact_state_count = int(model_config.get("contact_state_count", 3))
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
        self.kinematic_consistency_weight = float(
            loss_config.get("kinematic_consistency_weight", 0.0)
        )
        self.ddq_smoothness_weight = float(
            loss_config.get("ddq_smoothness_weight", loss_config.get("ddq_weight", 0.0))
        )
        self.ddq_smoothness_warmup_steps = int(
            loss_config.get("ddq_smoothness_warmup_steps", 1000)
        )
        self.ddq_smoothness_huber_delta = float(
            loss_config.get("ddq_smoothness_huber_delta", 1.0)
        )
        self.ddq_smoothness_normalize = bool(
            loss_config.get("ddq_smoothness_normalize", True)
        )
        self._global_step = 0
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
        estimator_dt = (model_config.get("state_estimator") or {}).get("sampling_dt")
        if "dt" not in loss_config and estimator_dt is not None:
            self.dt = float(estimator_dt)
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
        # Legacy binary option is retained as a no-op compatibility property.
        self.contact_positive_class_weight_is_auto = False
        self.contact_positive_class_weight = None
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

    def set_contact_positive_class_weight(self, value: float):
        self.contact_positive_class_weight = float(value)

    def set_global_step(self, global_step: int):
        self._global_step = max(int(global_step), 0)
        if self.ddq_smoothness_warmup_steps <= 0:
            self._ddq_smoothness_factor = 1.0
        else:
            self._ddq_smoothness_factor = min(
                1.0, self._global_step / float(self.ddq_smoothness_warmup_steps)
            )

    @staticmethod
    def _required(batch, key, ndim=3):
        if key not in batch:
            raise KeyError(f"torque world-model loss requires {key!r}")
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
        for key in STATE_KEYS:
            slices[key] = slice(offset, offset + self.joint_dim)
            offset += self.joint_dim
        slices["contact"] = slice(offset, offset + self.contact_state_count)
        return slices

    def flow_loss_components(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Flow prediction and target shapes differ")
        slices = self._flow_slices()
        losses = {
            key: F.mse_loss(prediction[..., sl], target[..., sl])
            for key, sl in slices.items()
            if key in STATE_KEYS
        }
        zero = prediction.new_zeros(())
        total = (
            self.flow_q_weight * losses["q"]
            + self.flow_dq_weight * losses["dq"]
            + self.flow_delta_q_weight * losses["delta_q"]
            + self.flow_tau_weight * losses["tau"]
        )
        # Contact logits are categorical and are trained only by the data CE
        # below.  They intentionally do not enter continuous flow MSE.
        return total, losses["q"], losses["dq"], losses["delta_q"], losses["tau"], zero

    def _direct_losses(self, out, batch):
        result = {}
        for key in STATE_KEYS:
            prediction = self._required(out, f"{key}_pred")
            target = self._required(batch, f"{key}_future").to(device=prediction.device, dtype=prediction.dtype)
            if prediction.shape != target.shape:
                raise ValueError(f"{key}_pred and {key}_future shapes differ")
            result[key] = F.mse_loss(prediction, target)
        return result

    def _contact_loss(self, out, batch, reference):
        if self.contact_state_count <= 0 or "contact_logits" not in out:
            return reference.new_zeros(())
        target = self._required(batch, "contact_future").to(device=reference.device)
        labels = target.squeeze(-1).round().long().clamp(0, self.contact_state_count - 1)
        logits = out["contact_logits"]
        if logits.shape[:2] != labels.shape:
            raise ValueError("contact logits and labels have incompatible shapes")
        weight = None
        if self.contact_class_weights is not None:
            weight = torch.as_tensor(self.contact_class_weights, device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits.reshape(-1, self.contact_state_count), labels.reshape(-1), weight=weight)

    def _kinematic_consistency(self, out, batch):
        if self.kinematic_consistency_weight <= 0.0:
            return out["q_pred"].new_zeros(())
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
        return F.mse_loss(increments, velocity_integral)

    def _ddq_smoothness(self, out):
        if self.ddq_smoothness_weight <= 0.0:
            return out["dq_pred"].new_zeros(())
        dq_future = self._physical("dq", self._required(out, "dq_pred"))
        if dq_future.shape[1] < 2:
            return dq_future.new_zeros(())
        ddq = torch.diff(dq_future, dim=1) / self.dt
        # Smoothness is a change-of-acceleration (jerk) penalty.  Penalizing
        # ddq^2 itself would bias the model toward zero acceleration and can
        # erase legitimate high-speed motion; the optional kinematic term is
        # the place where q/dq consistency is enforced.
        if ddq.shape[1] < 2:
            return ddq.new_zeros(())
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
        return F.huber_loss(
            jerk,
            torch.zeros_like(jerk),
            delta=self.ddq_smoothness_huber_delta,
        )

    # Diagnostic compatibility helper. It intentionally does not derive
    # delta_q; callers get the explicitly predicted stream instead.
    def _derived_state(self, out, batch):
        q_history = self._physical("q", self._required(batch, "q"))
        q_future = self._physical("q", self._required(out, "q_pred"))
        dq_future = self._physical("dq", self._required(out, "dq_pred"))
        ddq = torch.diff(dq_future, dim=1) / self.dt if dq_future.shape[1] > 1 else dq_future.new_zeros(dq_future.shape)
        return q_history, q_future, dq_future, ddq

    def __call__(self, out, batch):
        flow_prediction = out.get("flow_velocity_pred")
        flow_target = out.get("flow_velocity_target")
        if flow_prediction is None or flow_target is None:
            raise KeyError("model output must contain flow velocity prediction and target")
        flow_loss, flow_q, flow_dq, flow_delta_q, flow_tau, flow_contact = self.flow_loss_components(flow_prediction, flow_target)
        direct = self._direct_losses(out, batch)
        contact_loss = self._contact_loss(out, batch, out["q_pred"])
        kinematic_loss = self._kinematic_consistency(out, batch)
        smoothness_loss = self._ddq_smoothness(out)
        total = (
            self.flow_weight * flow_loss
            + self.q_weight * direct["q"]
            + self.dq_weight * direct["dq"]
            + self.delta_q_weight * direct["delta_q"]
            + self.tau_weight * direct["tau"]
            + self.contact_weight * contact_loss
            + self.kinematic_consistency_weight * kinematic_loss
            + self.ddq_smoothness_weight * self._ddq_smoothness_factor * smoothness_loss
        )
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": flow_q.detach(),
            "flow_dq_loss": flow_dq.detach(),
            "flow_delta_q_loss": flow_delta_q.detach(),
            "flow_tau_loss": flow_tau.detach(),
            "q_loss": direct["q"].detach(),
            "dq_loss": direct["dq"].detach(),
            "delta_q_loss": direct["delta_q"].detach(),
            "tau_loss": direct["tau"].detach(),
            "contact_loss": contact_loss.detach(),
            "kinematic_consistency_loss": kinematic_loss.detach(),
            "ddq_smoothness_loss": smoothness_loss.detach(),
            "ddq_smoothness_factor": flow_loss.new_tensor(self._ddq_smoothness_factor),
        }
        for key in STATE_KEYS:
            out[f"{key}_pred_physical"] = self._physical(key, out[f"{key}_pred"])
        out["ddq_pred_physical"] = torch.diff(out["dq_pred_physical"], dim=1) / self.dt if out["dq_pred_physical"].shape[1] > 1 else out["dq_pred_physical"].new_zeros(out["dq_pred_physical"].shape)
        return total, loss_dict


WorldModelLoss = TorqueWorldModelLoss
