"""Losses for the q/tau-conditioned torque world model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F

from model.pinn_model.causal_state import (
    CausalStateEstimatorConfig,
    causal_joint_state_from_position,
    future_joint_state_from_position,
)
from physics.nero_dynamics import (
    NeroDynamicsCache,
    RNEALinearization,
    load_tau_f_predictor,
    predict_nero_wrench,
)


class TorqueWorldModelLoss:
    """Combine Flow, trajectory, contact, and local dynamics supervision.

    Only the model's predicted q/tau/contact trajectory is consumed here.
    Dataset dq, ddq, and wrench are labels.  The differentiable RNEA path is a
    first-order expansion around each recorded future state; Pinocchio cache
    construction is intentionally owned by the trainer.
    """

    def __init__(self, config: Mapping):
        self.config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        loss_config = config.get("loss") or {}
        physics_config = config.get("physics") or {}

        self.joint_dim = int(model_config.get("joint_dim", 7))
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
        self.q_weight = float(loss_config.get("q_weight", 1.0))
        self.tau_weight = float(loss_config.get("tau_weight", 1.0))
        self.dq_weight = float(loss_config.get("dq_weight", 0.2))
        self.ddq_weight = float(loss_config.get("ddq_weight", 0.1))
        self.contact_weight = float(loss_config.get("contact_weight", 1.0))
        self.wrench_weight = float(loss_config.get("wrench_weight", 0.0))
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

        contact_config = config.get("contact_gate") or {}
        positive_weight = contact_config.get("positive_class_weight", "auto")
        self.contact_positive_class_weight_is_auto = (
            isinstance(positive_weight, str)
            and positive_weight.lower() == "auto"
        )
        self.contact_positive_class_weight = (
            None
            if self.contact_positive_class_weight_is_auto
            else float(positive_weight)
        )

        self.normalizer = None
        self.tau_f_predictor = None
        self._validate()

    @property
    def physics_enabled(self) -> bool:
        return self.wrench_weight > 0.0

    def _validate(self):
        weights = {
            "flow_weight": self.flow_weight,
            "flow_q_weight": self.flow_q_weight,
            "flow_tau_weight": self.flow_tau_weight,
            "flow_contact_weight": self.flow_contact_weight,
            "q_weight": self.q_weight,
            "tau_weight": self.tau_weight,
            "dq_weight": self.dq_weight,
            "ddq_weight": self.ddq_weight,
            "contact_weight": self.contact_weight,
            "wrench_weight": self.wrench_weight,
        }
        invalid = [name for name, value in weights.items() if value < 0.0]
        if invalid:
            raise ValueError(f"loss weights must be non-negative: {invalid}")
        if self.wrench_warmup_steps < 0:
            raise ValueError("loss.wrench_warmup_steps must be non-negative")
        if not math.isfinite(self.wrench_damping) or self.wrench_damping <= 0.0:
            raise ValueError("physics.wrench_damping must be positive")
        if self.contact_positive_class_weight is not None and (
            not math.isfinite(self.contact_positive_class_weight)
            or self.contact_positive_class_weight <= 0.0
        ):
            raise ValueError(
                "contact_gate.positive_class_weight must be 'auto' or positive"
            )
        if self.physics_enabled and self.tau_f_checkpoint_path is None:
            raise ValueError(
                "physics.tau_f_checkpoint_path is required when loss.wrench_weight > 0"
            )

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def set_contact_positive_class_weight(self, value: float):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("contact positive-class weight must be positive")
        self.contact_positive_class_weight = value

    def set_global_step(self, global_step: int):
        if self.wrench_warmup_steps <= 0:
            self.wrench_multiplier = 1.0
            return
        self.wrench_multiplier = min(
            max(float(global_step), 0.0) / self.wrench_warmup_steps,
            1.0,
        )

    def load_tau_f_checkpoint(self, device):
        if not self.physics_enabled:
            return
        self.tau_f_predictor = load_tau_f_predictor(
            self.tau_f_checkpoint_path,
            device=device,
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
        tau_loss = F.mse_loss(
            prediction[..., self.joint_dim : 2 * self.joint_dim],
            target[..., self.joint_dim : 2 * self.joint_dim],
        )
        contact_loss = F.mse_loss(
            prediction[..., 2 * self.joint_dim :],
            target[..., 2 * self.joint_dim :],
        )
        total = (
            self.flow_q_weight * q_loss
            + self.flow_tau_weight * tau_loss
            + self.flow_contact_weight * contact_loss
        )
        return total, q_loss, tau_loss, contact_loss

    def _derived_state(self, out, batch):
        q_history = self._physical("q", self._required(batch, "q", 3))
        q_future = self._physical("q", self._required(out, "q_pred", 3))
        state = future_joint_state_from_position(
            q_history,
            q_future,
            self.state_estimator_config,
        )
        return q_history, q_future, state["v"], state["a"]

    def _derived_mse(self, key, prediction, target):
        error = prediction - target.to(prediction)
        if self.standardize_derived_residuals:
            error = error / self._residual_scale(key, error)
        return error.square().mean()

    def _contact_loss(self, logits, labels):
        positive_weight = self.contact_positive_class_weight
        if positive_weight is None:
            raise RuntimeError(
                "automatic contact class weight has not been fitted on training data"
            )
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.to(logits),
            pos_weight=logits.new_tensor(positive_weight),
        )

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
        q_future,
        dq_future,
        ddq_future,
        tau_future,
    ):
        if self.tau_f_predictor is None:
            raise RuntimeError("frozen tau_f predictor has not been loaded")
        _, dq_history, ddq_history = causal_joint_state_from_position(
            q_history,
            self.state_estimator_config,
        )
        tau_history = self._physical("tau", self._required(batch, "tau", 3))
        tau_f = self.tau_f_predictor(
            history={
                "q": q_history,
                "dq": dq_history,
                "ddq": ddq_history,
                "tau": tau_history,
            },
            future={
                "q": q_future,
                "dq": dq_future,
                "ddq": ddq_future,
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
        if self.soft_contact_gate:
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

    def __call__(self, out, batch):
        flow_loss, flow_q, flow_tau, flow_contact = self.flow_loss_components(
            self._required(out, "flow_velocity_pred", 3),
            self._required(out, "flow_velocity_target", 3),
        )
        q_pred_normalized = self._required(out, "q_pred", 3)
        tau_pred_normalized = self._required(out, "tau_pred", 3)
        q_loss = F.mse_loss(
            q_pred_normalized,
            self._required(batch, "q_future", 3).to(q_pred_normalized),
        )
        tau_loss = F.mse_loss(
            tau_pred_normalized,
            self._required(batch, "tau_future", 3).to(tau_pred_normalized),
        )
        contact_loss = self._contact_loss(
            self._required(out, "contact_logits", 3),
            self._required(batch, "contact_future", 3),
        )

        q_history, q_future, dq_future, ddq_future = self._derived_state(out, batch)
        dq_loss = self._derived_mse(
            "dq",
            dq_future,
            self._required(batch, "dq_future_raw", 3),
        )
        ddq_loss = self._derived_mse(
            "ddq",
            ddq_future,
            self._required(batch, "ddq_future_raw", 3),
        )
        tau_future = self._physical("tau", tau_pred_normalized)

        if self.physics_enabled:
            wrench_loss, wrench_rmse, physics_out = self._wrench_loss(
                out,
                batch,
                q_history,
                q_future,
                dq_future,
                ddq_future,
                tau_future,
            )
            out.update(physics_out)
        else:
            wrench_loss = q_future.new_zeros(())
            wrench_rmse = q_future.new_zeros(())

        total = (
            self.flow_weight * flow_loss
            + self.q_weight * q_loss
            + self.tau_weight * tau_loss
            + self.dq_weight * dq_loss
            + self.ddq_weight * ddq_loss
            + self.contact_weight * contact_loss
            + self.wrench_weight * self.wrench_multiplier * wrench_loss
        )
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": flow_q.detach(),
            "flow_tau_loss": flow_tau.detach(),
            "flow_contact_loss": flow_contact.detach(),
            "q_loss": q_loss.detach(),
            "tau_loss": tau_loss.detach(),
            "dq_loss": dq_loss.detach(),
            "ddq_loss": ddq_loss.detach(),
            "contact_loss": contact_loss.detach(),
            "wrench_loss": wrench_loss.detach(),
            "wrench_physical_rmse": wrench_rmse.detach(),
            "wrench_multiplier": self.wrench_multiplier,
        }
        out["q_pred_physical"] = q_future
        out["dq_pred_physical"] = dq_future
        out["ddq_pred_physical"] = ddq_future
        out["tau_pred_physical"] = tau_future
        return total, loss_dict


WorldModelLoss = TorqueWorldModelLoss
