"""Data losses for the configurable state/contact world model.

``delta_q`` is a real dataset channel. Continuous targets use Flow Matching;
current measured tau also supervises the optional shared free-dynamics head.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from model.pinn_model.contact_world_model import PREDICTED_STATE_STREAMS


class ContactWorldModelLoss:
    """Combine flow matching, contact CE and optional free-dynamics supervision."""

    FLOW_STREAM_WEIGHTS = {"q": 1.0, "dq": 1.0, "delta_q": 1.0, "tau": 1.0}
    CONTACT_LOSS_WEIGHT = 0.1

    def __init__(self, config: Mapping):
        self.config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        loss_config = config.get("loss") or {}
        train_config = config.get("train") or {}
        self.joint_dim = int(model_config.get("joint_dim", 7))
        configured_inputs = model_config.get("inputs", PREDICTED_STATE_STREAMS)
        if isinstance(configured_inputs, str):
            configured_inputs = [configured_inputs]
        configured_outputs = model_config.get("outputs")
        if configured_outputs is None:
            configured_outputs = configured_inputs
        if isinstance(configured_outputs, str):
            configured_outputs = [configured_outputs]
        self.predicted_state_streams = tuple(str(value).lower() for value in configured_outputs)
        if not self.predicted_state_streams:
            raise ValueError("model.outputs must contain at least one state stream")
        unknown = sorted(set(self.predicted_state_streams) - set(PREDICTED_STATE_STREAMS))
        if unknown:
            raise ValueError(f"model.outputs contains unsupported values {unknown}")
        if len(set(self.predicted_state_streams)) != len(self.predicted_state_streams):
            raise ValueError("model.outputs must not contain duplicates")
        self.contact_state_count = int(model_config.get("contact_state_count", 3))
        if self.contact_state_count < 2:
            raise ValueError("model.contact_state_count must be at least 2")
        self.normalize_mode = data_config.get("normalize_mode")
        self.flow_weight = 1.0
        # Stream weights are part of the fixed objective, not experiment YAML.
        self.flow_q_weight = self.FLOW_STREAM_WEIGHTS["q"]
        self.flow_dq_weight = self.FLOW_STREAM_WEIGHTS["dq"]
        self.flow_delta_q_weight = self.FLOW_STREAM_WEIGHTS["delta_q"]
        self.flow_tau_weight = self.FLOW_STREAM_WEIGHTS["tau"]
        removed = {"tau_history_mask_warmup_steps", "tau_free_warmup_steps"} & train_config.keys()
        if removed:
            raise ValueError(f"Removed tau history masking options: {sorted(removed)}")
        self.free_dynamics_weight = float(loss_config.get("free_dynamics_weight", 0.0))
        if not math.isfinite(self.free_dynamics_weight) or self.free_dynamics_weight < 0:
            raise ValueError("loss.free_dynamics_weight must be finite and non-negative")
        self.q_weight = float(loss_config.get("q_weight", 1.0))
        self.dq_weight = float(loss_config.get("dq_weight", 1.0))
        self.delta_q_weight = float(loss_config.get("delta_q_weight", 1.0))
        self.tau_weight = float(loss_config.get("tau_weight", 1.0))
        self.contact_weight = self.CONTACT_LOSS_WEIGHT
        removed_loss = {
            "endpoint_loss", "kinematic_consistency_weight", "kinematic_joint_scales",
            "delta_q_consistency_weight", "torque_contact_weight", "ddq_smoothness_weight",
            "ddq_smoothness_warmup_steps", "ddq_smoothness_huber_delta", "ddq_smoothness_normalize",
        } & loss_config.keys()
        if removed_loss:
            raise ValueError(f"Removed unused WM loss options: {sorted(removed_loss)}")
        self.emit_physical_diagnostics = bool(loss_config.get("emit_physical_diagnostics", False))
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
        labels = target.squeeze(-1).round().long()
        if torch.any(labels < 0) or torch.any(labels >= self.contact_state_count):
            raise ValueError(
                "contact_future contains a phase outside model.contact_state_count"
            )
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

    def free_dynamics_loss(self, out, batch):
        """MSE in the existing normalized tau space, conditional on free history."""
        batch = out.get("_prepared_batch", batch)
        prediction = out["free_tau_pred"]
        tau = batch["tau"]
        mask = batch["free_dynamics_mask"]
        if prediction.shape != tau[:, -1].shape or mask.shape != prediction.shape[:1]:
            raise ValueError("free dynamics target/mask must align with current tau")
        target = tau[:, -1].detach()
        mask = mask & torch.isfinite(target).all(dim=-1)
        weights = batch.get("importance_weight", prediction.new_ones(prediction.shape[0]))
        weights = weights.to(device=prediction.device, dtype=torch.float32).reshape(-1)
        if weights.shape != mask.shape or not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("importance_weight must contain one finite non-negative weight per sample")
        # Index before MSE so invalid/non-free targets (including NaNs) cannot
        # contaminate the zero-sample case. The empty sum retains a grad path.
        selected_weights = weights[mask]
        mse = (prediction[mask].float() - target[mask].float()).square().mean(dim=-1)
        denominator = selected_weights.sum()
        loss = (selected_weights * mse).sum() / denominator.clamp_min(torch.finfo(torch.float32).tiny)
        return loss, mask.sum()

    def __call__(self, out, batch):
        batch = out.get("_prepared_batch", batch)
        flow_prediction = out.get("flow_velocity_pred")
        flow_target = out.get("flow_velocity_target")
        if flow_prediction is None or flow_target is None:
            raise KeyError("model output must contain flow velocity prediction and target")
        flow_loss_ps, flow_q_ps, flow_dq_ps, flow_delta_q_ps, flow_tau_ps = self.flow_loss_components(flow_prediction, flow_target)
        flow_loss_ps = (
            self.flow_q_weight * flow_q_ps
            + self.flow_dq_weight * flow_dq_ps
            + self.flow_delta_q_weight * flow_delta_q_ps
            + self.flow_tau_weight * flow_tau_ps
        )
        # Future direct-state MSE remains diagnostic only.
        direct = self._direct_losses(out, batch)
        contact_loss_ps = self._contact_loss(out, batch, out[f"{self.predicted_state_streams[0]}_pred"])
        importance_weight = batch.get("importance_weight")
        flow_loss = self._weighted_mean(flow_loss_ps, importance_weight)
        # Continuous streams use flow matching; the categorical contact head
        # is trained jointly with its fixed CE term. Other state losses remain
        # diagnostics only.
        contact_loss = self._weighted_mean(contact_loss_ps, importance_weight)
        free_loss, free_count = (
            self.free_dynamics_loss(out, batch) if self.free_dynamics_weight > 0
            else (flow_loss.new_zeros(()), flow_loss.new_zeros(()))
        )
        total = (self.flow_weight * flow_loss + self.contact_weight * contact_loss
                 + self.free_dynamics_weight * free_loss)
        loss_dict = {
            "total_loss": total.detach(),
            "flow_loss": flow_loss.detach(),
            "flow_q_loss": self._weighted_mean(flow_q_ps, importance_weight).detach(),
            "flow_dq_loss": self._weighted_mean(flow_dq_ps, importance_weight).detach(),
            "flow_delta_q_loss": self._weighted_mean(flow_delta_q_ps, importance_weight).detach(),
            "flow_tau_loss": self._weighted_mean(flow_tau_ps, importance_weight).detach(),
            # Unweighted per-stream means make scale imbalance explicit:
            # compare these values before changing the fixed stream weights.
            "flow_q_raw": self._weighted_mean(flow_q_ps, importance_weight).detach(),
            "flow_dq_raw": self._weighted_mean(flow_dq_ps, importance_weight).detach(),
            "flow_delta_q_raw": self._weighted_mean(flow_delta_q_ps, importance_weight).detach(),
            "flow_tau_raw": self._weighted_mean(flow_tau_ps, importance_weight).detach(),
            "flow_q_contribution": (self.flow_q_weight * self._weighted_mean(flow_q_ps, importance_weight)).detach(),
            "flow_dq_contribution": (self.flow_dq_weight * self._weighted_mean(flow_dq_ps, importance_weight)).detach(),
            "flow_delta_q_contribution": (self.flow_delta_q_weight * self._weighted_mean(flow_delta_q_ps, importance_weight)).detach(),
            "flow_tau_contribution": (self.flow_tau_weight * self._weighted_mean(flow_tau_ps, importance_weight)).detach(),
            "free_dynamics_loss": free_loss.detach(),
            "free_dynamics_contribution": (self.free_dynamics_weight * free_loss).detach(),
            "free_dynamics_count": free_count.detach(),
            **{f"{key}_loss": self._weighted_mean(value, importance_weight).detach() for key, value in direct.items()},
            "contact_loss": contact_loss.detach(),
            "contact_contribution": (self.contact_weight * contact_loss).detach(),
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
