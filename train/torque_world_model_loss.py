"""Losses for the q/dq/tau/wrench-conditioned torque world model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from model.pinn_model.causal_state import (
    CausalStateEstimatorConfig,
    causal_joint_acceleration_from_velocity,
    causal_joint_state_from_position,
    future_joint_acceleration_from_velocity,
    future_joint_state_from_position,
)
from physics.nero_dynamics import (
    NeroDynamicsCache,
    RNEALinearization,
    load_tau_f_predictor,
    predict_nero_wrench,
)


class TorqueWorldModelLoss:
    """Combine Flow, trajectory, and local dynamics supervision.

    The model predicts q, dq, tau, and optionally wrench. Acceleration is reconstructed causally
    from predicted dq with a configurable q-derived blend. The differentiable
    RNEA path is a first-order expansion around recorded or reconstructed
    future states; Pinocchio cache construction is intentionally owned by the
    trainer.
    """

    def __init__(self, config: Mapping):
        self.config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        loss_config = config.get("loss") or {}
        physics_config = config.get("physics") or {}

        self.joint_dim = int(model_config.get("joint_dim", 7))
        self.state_contract = str(
            model_config.get("state_contract", "q_dq_tau_wrench")
        ).lower()
        self.q_tau_contact_contract = self.state_contract == "q_tau_contact"
        self.wrench_dim = int(model_config.get("wrench_dim", 0))
        self.normalize_mode = data_config.get("normalize_mode")
        self.state_estimator_config = CausalStateEstimatorConfig.from_model_config(
            config
        )
        self.flow_weight = float(loss_config.get("flow_weight", 1.0))
        self.flow_q_weight = float(loss_config.get("flow_q_weight", 1.0))
        self.flow_tau_weight = float(loss_config.get("flow_tau_weight", 1.0))
        self.flow_contact_weight = float(
            loss_config.get("flow_contact_weight", 1.0)
        )
        self.flow_dq_weight = float(loss_config.get("flow_dq_weight", 1.0))
        self.flow_wrench_weight = float(
            loss_config.get(
                "flow_wrench_weight",
                1.0 if self.wrench_dim else 0.0,
            )
        )
        self.q_weight = float(loss_config.get("q_weight", 1.0))
        self.dq_weight = float(loss_config.get("dq_weight", 0.2))
        self.tau_weight = float(loss_config.get("tau_weight", 1.0))
        self.contact_weight = float(loss_config.get("contact_weight", 1.0))
        self.ddq_weight = float(loss_config.get("ddq_weight", 0.1))
        self.derived_warmup_steps = int(
            loss_config.get("derived_warmup_steps", 0)
        )
        self.derived_multiplier = (
            0.0 if self.derived_warmup_steps > 0 else 1.0
        )
        self.wrench_weight = float(loss_config.get("wrench_weight", 0.0))
        self.wrench_observation_weight = float(
            loss_config.get(
                "wrench_observation_weight",
                1.0 if self.wrench_dim else 0.0,
            )
        )
        self.wrench_warmup_steps = int(
            loss_config.get("wrench_warmup_steps", 0)
        )
        self.wrench_multiplier = (
            0.0 if self.wrench_warmup_steps > 0 else 1.0
        )
        self.standardize_derived_residuals = bool(
            loss_config.get("standardize_derived_residuals", True)
        )
        self.standardize_wrench_residual = bool(
            loss_config.get("standardize_wrench_residual", True)
        )
        self.ddq_q_blend = float(loss_config.get("ddq_q_blend", 0.2))
        self.wrench_damping = float(
            physics_config.get(
                "wrench_damping",
                loss_config.get("wrench_damping", 0.02),
            )
        )
        self.soft_contact_gate = bool(
            physics_config.get("soft_contact_gate", True)
        )
        checkpoint = physics_config.get(
            "tau_f_checkpoint_path",
            loss_config.get("tau_f_checkpoint_path"),
        )
        self.tau_f_checkpoint_path = Path(checkpoint) if checkpoint else None
        self.tau_f_window_batch_size = int(
            physics_config.get(
                "tau_f_window_batch_size",
                loss_config.get("tau_f_window_batch_size", 1024),
            )
        )
        rnea_config = (
            physics_config.get("rnea")
            or physics_config.get("rnea_rollout")
            or {}
        )
        configured_rnea_enabled = physics_config.get("rnea_rollout_enabled")
        self.rnea_enabled = bool(
            rnea_config.get(
                "enabled",
                self.wrench_weight > 0.0
                if configured_rnea_enabled is None
                else configured_rnea_enabled,
            )
        )

        self.contact_positive_class_weight_is_auto = False
        self.contact_positive_class_weight = None
        contact_config = config.get("contact_gate") or {}
        configured_class_weights = contact_config.get(
            "class_weights",
            loss_config.get("contact_class_weights", "auto"),
        )
        if isinstance(configured_class_weights, str):
            self.contact_class_weights_is_auto = (
                configured_class_weights.lower() == "auto"
            )
            self.contact_class_weights = None
        else:
            values = tuple(float(value) for value in configured_class_weights)
            if len(values) != 3:
                raise ValueError("contact_gate.class_weights must contain 3 values")
            self.contact_class_weights_is_auto = False
            self.contact_class_weights = values

        self.normalizer = None
        self.tau_f_predictor = None
        self._validate()

    @property
    def physics_enabled(self) -> bool:
        return self.rnea_enabled and self.wrench_weight > 0.0

    def _validate(self):
        if self.wrench_dim < 0:
            raise ValueError("model.wrench_dim must be non-negative")
        weights = {
            "flow_weight": self.flow_weight,
            "flow_q_weight": self.flow_q_weight,
            "flow_dq_weight": self.flow_dq_weight,
            "flow_contact_weight": self.flow_contact_weight,
            "flow_wrench_weight": self.flow_wrench_weight,
            "flow_tau_weight": self.flow_tau_weight,
            "q_weight": self.q_weight,
            "dq_weight": self.dq_weight,
            "tau_weight": self.tau_weight,
            "contact_weight": self.contact_weight,
            "ddq_weight": self.ddq_weight,
            "wrench_weight": self.wrench_weight,
            "wrench_observation_weight": self.wrench_observation_weight,
        }
        invalid = [name for name, value in weights.items() if value < 0.0]
        if invalid:
            raise ValueError(f"loss weights must be non-negative: {invalid}")
        if self.q_tau_contact_contract and self.wrench_dim != 0:
            raise ValueError(
                "model.wrench_dim must be zero for state_contract=q_tau_contact"
            )
        if self.contact_class_weights is not None and any(
            not math.isfinite(value) or value <= 0.0
            for value in self.contact_class_weights
        ):
            raise ValueError("contact class weights must be finite and positive")
        if self.wrench_warmup_steps < 0:
            raise ValueError("loss.wrench_warmup_steps must be non-negative")
        if self.derived_warmup_steps < 0:
            raise ValueError("loss.derived_warmup_steps must be non-negative")
        if self.tau_f_window_batch_size <= 0:
            raise ValueError("physics.tau_f_window_batch_size must be positive")
        if not 0.0 <= self.ddq_q_blend <= 1.0:
            raise ValueError("loss.ddq_q_blend must be in [0, 1]")
        if not math.isfinite(self.wrench_damping) or self.wrench_damping <= 0.0:
            raise ValueError("physics.wrench_damping must be positive")
        if self.physics_enabled and self.tau_f_checkpoint_path is None:
            raise ValueError(
                "physics.tau_f_checkpoint_path is required when loss.wrench_weight > 0"
            )
        if self.wrench_observation_weight > 0.0 and self.wrench_dim == 0:
            raise ValueError(
                "loss.wrench_observation_weight requires model.wrench_dim > 0"
            )

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_contact_positive_class_weight(self, value: float):
        self.contact_positive_class_weight = float(value)

    def set_contact_class_weights(self, values):
        values = tuple(float(value) for value in values)
        if len(values) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("contact class weights must contain 3 positive values")
        self.contact_class_weights = values

    def set_global_step(self, global_step: int):
        if self.wrench_warmup_steps <= 0:
            self.wrench_multiplier = 1.0
        else:
            self.wrench_multiplier = min(
                max(float(global_step), 0.0) / self.wrench_warmup_steps,
                1.0,
            )
        if self.derived_warmup_steps <= 0:
            self.derived_multiplier = 1.0
        else:
            self.derived_multiplier = min(
                max(float(global_step), 0.0) / self.derived_warmup_steps,
                1.0,
            )

    def load_tau_f_checkpoint(self, device):
        if not self.physics_enabled:
            return
        self.tau_f_predictor = load_tau_f_predictor(
            self.tau_f_checkpoint_path,
            device=device,
            max_model_batch_size=self.tau_f_window_batch_size,
        )

    @staticmethod
    def _required(batch, key: str, ndim: int):
        if key not in batch:
            raise KeyError(f"torque world-model loss requires {key!r}")
        value = batch[key]
        if not torch.is_tensor(value) or value.ndim != ndim:
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value)
            raise ValueError(f"{key!r} must have {ndim} dimensions, got {shape}")
        return value

    def _physical(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode is None:
            return value
        if self.normalizer is None:
            raise RuntimeError("loss normalizer has not been set by the trainer")
        functions = {
            "gaussian": self.normalizer.gaussian_denormalize,
            "limit": self.normalizer.limit_denormalize,
            "quantile": self.normalizer.quantile_denormalize,
        }
        if self.normalize_mode not in functions:
            raise ValueError(f"unsupported normalize_mode {self.normalize_mode!r}")
        return functions[self.normalize_mode](key, value)

    def _residual_scale(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None or key not in self.normalizer.stats:
            return torch.ones(value.shape[-1], device=value.device, dtype=value.dtype)
        statistics = self.normalizer.stats[key]
        if self.normalize_mode == "gaussian":
            scale = statistics["std"]
        elif self.normalize_mode == "limit":
            scale = 0.5 * (statistics["max"] - statistics["min"])
        elif self.normalize_mode == "quantile":
            scale = 0.5 * (statistics["q99"] - statistics["q01"])
        else:
            return torch.ones(value.shape[-1], device=value.device, dtype=value.dtype)
        return scale.to(device=value.device, dtype=value.dtype).clamp_min(
            float(self.normalizer.eps)
        )

    def flow_loss_components(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("Flow prediction and target shapes differ")
        q_loss = F.mse_loss(
            prediction[..., : self.joint_dim],
            target[..., : self.joint_dim],
        )
        dq_loss = F.mse_loss(
            prediction[..., self.joint_dim : 2 * self.joint_dim],
            target[..., self.joint_dim : 2 * self.joint_dim],
        )
        tau_loss = F.mse_loss(
            prediction[..., 2 * self.joint_dim : 3 * self.joint_dim],
            target[..., 2 * self.joint_dim : 3 * self.joint_dim],
        )
        if self.wrench_dim:
            wrench_loss = F.mse_loss(
                prediction[..., 3 * self.joint_dim : 3 * self.joint_dim + self.wrench_dim],
                target[..., 3 * self.joint_dim : 3 * self.joint_dim + self.wrench_dim],
            )
        else:
            wrench_loss = prediction.new_zeros(())
        total = (
            self.flow_q_weight * q_loss
            + self.flow_dq_weight * dq_loss
            + self.flow_tau_weight * tau_loss
            + self.flow_wrench_weight * wrench_loss
        )
        return total, q_loss, dq_loss, tau_loss, wrench_loss

    def _predicted_state(self, out, batch):
        q_history = self._physical("q", self._required(batch, "q", 3))
        dq_history = self._physical("dq", self._required(batch, "dq", 3))
        q_future = self._physical("q", self._required(out, "q_pred", 3))
        dq_future = self._physical("dq", self._required(out, "dq_pred", 3))
        ddq_from_dq = future_joint_acceleration_from_velocity(
            dq_history,
            dq_future,
            self.state_estimator_config,
        )
        ddq_from_q = future_joint_state_from_position(
            q_history,
            q_future,
            self.state_estimator_config,
        )["a"]
        ddq_future = (
            (1.0 - self.ddq_q_blend) * ddq_from_dq
            + self.ddq_q_blend * ddq_from_q
        )
        return q_history, dq_history, q_future, dq_future, ddq_future

    def _derived_state(self, out, batch):
        """Backward-compatible derived-state view used by diagnostics."""

        q_history, _, q_future, dq_future, ddq_future = self._predicted_state(
            out, batch
        )
        return q_history, q_future, dq_future, ddq_future

    def _derived_mse(self, key, prediction, target):
        error = prediction - target.to(prediction)
        if self.standardize_derived_residuals:
            error = error / self._residual_scale(key, error)
        return error.square().mean()

    def _physics_cache(self, batch, reference):
        q_reference = self._required(batch, "q_future_raw", 3).to(reference)
        dq_reference = self._required(batch, "dq_future_raw", 3).to(reference)
        ddq_reference = self._required(batch, "ddq_future_raw", 3).to(reference)
        linearization = RNEALinearization(
            q_reference=q_reference,
            dq_reference=dq_reference,
            ddq_reference=ddq_reference,
            tau_id_reference=self._required(
                batch, "rnea_tau_id_future", 3
            ).to(reference),
            d_tau_d_q=self._required(batch, "rnea_d_tau_d_q_future", 4).to(
                reference
            ),
            d_tau_d_dq=self._required(
                batch, "rnea_d_tau_d_dq_future", 4
            ).to(reference),
            d_tau_d_ddq=self._required(
                batch, "rnea_d_tau_d_ddq_future", 4
            ).to(reference),
        )
        return NeroDynamicsCache(
            rnea=linearization,
            frame_jacobian=self._required(batch, "frame_jacobian_future", 4).to(
                reference
            ),
        )

    def _wrench_loss(
        self,
        out,
        batch,
        q_history,
        dq_history,
        q_future,
        dq_future,
        ddq_future,
        tau_future,
    ):
        if self.tau_f_predictor is None:
            raise RuntimeError("frozen tau_f predictor has not been loaded")
        _, ddq_history_from_dq = causal_joint_acceleration_from_velocity(
            self._physical("dq", self._required(batch, "dq", 3)),
            self.state_estimator_config,
        )
        _, _, ddq_history_from_q = causal_joint_state_from_position(
            q_history,
            self.state_estimator_config,
        )
        ddq_history = (
            (1.0 - self.ddq_q_blend) * ddq_history_from_dq
            + self.ddq_q_blend * ddq_history_from_q
        )
        tau_history = self._physical("tau", self._required(batch, "tau", 3))
        q_complete = torch.cat((q_history, q_future), dim=1)
        delta_q_complete = torch.zeros_like(q_complete)
        delta_q_complete[:, 1:] = q_complete[:, 1:] - q_complete[:, :-1]
        history_steps = q_history.shape[1]
        delta_q_history = delta_q_complete[:, :history_steps]
        delta_q_future = delta_q_complete[:, history_steps:]
        tau_f = self.tau_f_predictor(
            history={
                "q": q_history,
                "dq": dq_history,
                "ddq": ddq_history,
                "delta_q": delta_q_history,
                "tau": tau_history,
            },
            future={
                "q": q_future,
                "dq": dq_future,
                "ddq": ddq_future,
                "delta_q": delta_q_future,
                "tau": tau_future,
            },
        )
        prediction = predict_nero_wrench(
            q=q_future,
            dq=dq_future,
            ddq=ddq_future,
            tau_measured=tau_future,
            tau_f=tau_f,
            cache=self._physics_cache(batch, q_future),
            damping=self.wrench_damping,
        )
        wrench = prediction.wrench
        if self.soft_contact_gate and "contact_probability" in out:
            wrench = wrench * out["contact_probability"].to(wrench)
        target = self._required(batch, "wrench_future_raw", 3).to(wrench)
        error = wrench - target
        if self.standardize_wrench_residual:
            error = error / self._residual_scale("wrench", error)
        loss = error.square().mean()
        diagnostics = {
            "tau_id_pred": prediction.tau_id,
            "tau_f_pred": prediction.tau_f,
            "tau_external_pred": prediction.tau_external,
            "wrench_raw_pred": prediction.wrench,
            "wrench_pred": wrench,
        }
        return loss, (wrench - target).square().mean().sqrt(), diagnostics

    def _call_qtau_contact(self, out, batch):
        flow_prediction = self._required(out, "flow_velocity_pred", 3)
        flow_target = self._required(out, "flow_velocity_target", 3)
        if flow_prediction.shape != flow_target.shape:
            raise ValueError("Flow prediction and target shapes differ")
        q_slice = slice(0, self.joint_dim)
        tau_slice = slice(self.joint_dim, 2 * self.joint_dim)
        contact_slice = slice(2 * self.joint_dim, None)
        flow_q_loss = F.mse_loss(
            flow_prediction[..., q_slice], flow_target[..., q_slice]
        )
        flow_tau_loss = F.mse_loss(
            flow_prediction[..., tau_slice], flow_target[..., tau_slice]
        )
        flow_contact_loss = F.mse_loss(
            flow_prediction[..., contact_slice], flow_target[..., contact_slice]
        )

        q_pred = self._required(out, "q_pred", 3)
        tau_pred = self._required(out, "tau_pred", 3)
        q_target = self._required(batch, "q_future", 3).to(q_pred)
        q_loss = F.mse_loss(q_pred, q_target)
        tau_loss = F.mse_loss(
            tau_pred, self._required(batch, "tau_future", 3).to(tau_pred)
        )
        q_history_physical = self._physical(
            "q", self._required(batch, "q", 3)
        )
        q_pred_physical = self._physical("q", q_pred)
        q_target_physical = self._physical("q", q_target)
        predicted_derived = future_joint_state_from_position(
            q_history_physical,
            q_pred_physical,
            self.state_estimator_config,
        )
        target_derived = future_joint_state_from_position(
            q_history_physical,
            q_target_physical,
            self.state_estimator_config,
        )
        dq_loss = self._derived_mse(
            "dq", predicted_derived["v"], target_derived["v"]
        )
        ddq_loss = self._derived_mse(
            "ddq", predicted_derived["a"], target_derived["a"]
        )
        contact_logits = self._required(out, "contact_logits", 3)
        labels = self._required(batch, "contact_future", 3).to(contact_logits)
        labels = labels.squeeze(-1).round().to(dtype=torch.long)
        if torch.any((labels < 0) | (labels >= 3)):
            raise ValueError("contact_future values must be integer states 0, 1, or 2")
        if self.contact_class_weights is None:
            raise RuntimeError(
                "three-phase contact class weights have not been fitted"
            )
        class_weights = contact_logits.new_tensor(self.contact_class_weights)
        contact_loss = F.cross_entropy(
            contact_logits.reshape(-1, 3),
            labels.reshape(-1),
            weight=class_weights,
        )
        flow_loss = (
            self.flow_q_weight * flow_q_loss
            + self.flow_tau_weight * flow_tau_loss
            + self.flow_contact_weight * flow_contact_loss
        )
        total = (
            self.flow_weight * flow_loss
            + self.q_weight * q_loss
            + self.tau_weight * tau_loss
            + self.derived_multiplier
            * (self.dq_weight * dq_loss + self.ddq_weight * ddq_loss)
            + self.contact_weight * contact_loss
        )
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": flow_q_loss.detach(),
            "flow_tau_loss": flow_tau_loss.detach(),
            "flow_contact_loss": flow_contact_loss.detach(),
            "q_loss": q_loss.detach(),
            "tau_loss": tau_loss.detach(),
            "dq_loss": dq_loss.detach(),
            "ddq_loss": ddq_loss.detach(),
            "derived_multiplier": self.derived_multiplier,
            "contact_loss": contact_loss.detach(),
        }
        out["contact_phase_probability"] = torch.softmax(contact_logits, dim=-1)
        out["contact_state_pred"] = out["contact_phase_probability"].argmax(
            dim=-1, keepdim=True
        ).to(dtype=contact_logits.dtype)
        out["q_pred_physical"] = q_pred_physical
        out["dq_pred_physical"] = predicted_derived["v"]
        out["ddq_pred_physical"] = predicted_derived["a"]
        out["dq_target_physical"] = target_derived["v"]
        out["ddq_target_physical"] = target_derived["a"]
        out["tau_pred_physical"] = self._physical("tau", tau_pred)
        return total, loss_dict

    def __call__(self, out, batch):
        if self.q_tau_contact_contract:
            return self._call_qtau_contact(out, batch)
        flow_loss, flow_q, flow_dq, flow_tau, flow_wrench = self.flow_loss_components(
            self._required(out, "flow_velocity_pred", 3),
            self._required(out, "flow_velocity_target", 3),
        )
        q_pred_normalized = self._required(out, "q_pred", 3)
        dq_pred_normalized = self._required(out, "dq_pred", 3)
        tau_pred_normalized = self._required(out, "tau_pred", 3)
        wrench_pred_normalized = out.get("wrench_pred")
        q_loss = F.mse_loss(
            q_pred_normalized,
            self._required(batch, "q_future", 3).to(q_pred_normalized),
        )
        dq_loss = F.mse_loss(
            dq_pred_normalized,
            self._required(batch, "dq_future", 3).to(dq_pred_normalized),
        )
        tau_loss = F.mse_loss(
            tau_pred_normalized,
            self._required(batch, "tau_future", 3).to(tau_pred_normalized),
        )
        if self.wrench_dim:
            if wrench_pred_normalized is None:
                raise KeyError("model output is missing wrench_pred")
            wrench_target = self._required(batch, "wrench_future", 3).to(
                wrench_pred_normalized
            )
            wrench_direct_loss = F.mse_loss(
                wrench_pred_normalized,
                wrench_target,
            )
        else:
            wrench_direct_loss = q_pred_normalized.new_zeros(())

        q_history, dq_history, q_future, dq_future, ddq_future = self._predicted_state(
            out,
            batch,
        )
        if self.ddq_weight > 0.0 and "ddq_future_raw" in batch:
            ddq_loss = self._derived_mse(
                "ddq",
                ddq_future,
                self._required(batch, "ddq_future_raw", 3),
            )
        else:
            ddq_loss = q_future.new_zeros(())
        tau_future = self._physical("tau", tau_pred_normalized)

        if self.physics_enabled:
            wrench_loss, wrench_rmse, physics_out = self._wrench_loss(
                out,
                batch,
                q_history,
                dq_history,
                q_future,
                dq_future,
                ddq_future,
                tau_future,
            )
            for key, value in physics_out.items():
                if key == "wrench_pred" and wrench_pred_normalized is not None:
                    out["wrench_physics_pred"] = value
                else:
                    out[key] = value
        else:
            wrench_loss = q_future.new_zeros(())
            wrench_rmse = q_future.new_zeros(())

        total = (
            self.flow_weight * flow_loss
            + self.q_weight * q_loss
            + self.dq_weight * dq_loss
            + self.tau_weight * tau_loss
            + self.wrench_observation_weight * wrench_direct_loss
            + self.ddq_weight * ddq_loss
            + self.wrench_weight * self.wrench_multiplier * wrench_loss
        )
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": flow_q.detach(),
            "flow_dq_loss": flow_dq.detach(),
            "flow_tau_loss": flow_tau.detach(),
            "flow_wrench_loss": flow_wrench.detach(),
            "q_loss": q_loss.detach(),
            "dq_loss": dq_loss.detach(),
            "tau_loss": tau_loss.detach(),
            "ddq_loss": ddq_loss.detach(),
            "wrench_loss": wrench_loss.detach(),
            "wrench_direct_loss": wrench_direct_loss.detach(),
            "wrench_physical_rmse": wrench_rmse.detach(),
            "wrench_multiplier": self.wrench_multiplier,
            "ddq_q_blend": self.ddq_q_blend,
        }
        out["q_pred_physical"] = q_future
        out["dq_pred_physical"] = dq_future
        out["ddq_pred_physical"] = ddq_future
        out["tau_pred_physical"] = tau_future
        return total, loss_dict


WorldModelLoss = TorqueWorldModelLoss
