"""Conditional Flow Matching world model for joint position and torque.

The neural condition is deliberately restricted to a high-rate ``q/tau``
history and a future end-effector target chunk.  Derivatives, wrench, and
contact labels are supervision-only quantities handled outside the condition
encoder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn


def _sinusoidal_position_encoding(
    length: int,
    width: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Return a device/dtype-matched ``[1, length, width]`` encoding."""

    position = torch.arange(
        length,
        device=reference.device,
        dtype=reference.dtype,
    )[:, None]
    frequency = torch.exp(
        torch.arange(
            0,
            width,
            2,
            device=reference.device,
            dtype=reference.dtype,
        )
        * (-math.log(10_000.0) / width)
    )
    angles = position * frequency[None, :]
    encoding = reference.new_zeros(length, width)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding[None, :, :]


class FlowTimeEmbedding(nn.Module):
    """Embed the scalar Flow time without coupling it to trajectory time."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        if flow_time.ndim == 1:
            flow_time = flow_time[:, None]
        if flow_time.ndim != 2 or flow_time.shape[-1] != 1:
            raise ValueError(
                "flow_time must have shape [B] or [B, 1], got "
                f"{tuple(flow_time.shape)}"
            )
        features = torch.cat(
            (
                flow_time,
                torch.sin(math.pi * flow_time),
                torch.cos(math.pi * flow_time),
                torch.sin(2.0 * math.pi * flow_time),
                torch.cos(2.0 * math.pi * flow_time),
            ),
            dim=-1,
        )
        return self.projection(features)


class FlowDecoderBlock(nn.Module):
    """Non-causal trajectory attention followed by condition attention."""

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.condition_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.self_norm(trajectory)
        attended, _ = self.self_attention(
            query=normalized,
            key=normalized,
            value=normalized,
            need_weights=False,
        )
        trajectory = trajectory + self.dropout(attended)

        attended, _ = self.condition_attention(
            query=self.condition_norm(trajectory),
            key=memory,
            value=memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        trajectory = trajectory + self.dropout(attended)
        return trajectory + self.dropout(self.ffn(self.ffn_norm(trajectory)))


class TorqueWorldModel(nn.Module):
    """Generate future ``q``, ``tau``, and contact logits with CFM.

    Condition keys:

    - ``q``: normalized joint-position history ``[B, H, D]``
    - ``tau``: normalized measured-torque history ``[B, H, D]``
    - ``target_relative_pose``: normalized held action targets ``[B, A, U]``
    - ``target_relative_pose_mask``: optional valid-token mask ``[B, A]``

    ``H``, ``A``, and the generated future horizon ``T`` are independent.
    Training additionally requires ``q_future``, ``tau_future``, and
    ``contact_future`` targets, but none of those are condition-encoder inputs.
    """

    CONDITION_KEYS = (
        "q",
        "tau",
        "target_relative_pose",
        "target_relative_pose_mask",
    )
    TARGET_KEYS = ("q_future", "tau_future", "contact_future")
    MODEL_VERSION = "torque_world_model_v1"

    def __init__(self, config: Mapping):
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}

        self.history_horizon = int(
            data_config.get(
                "state_history_horizon",
                data_config.get("history_horizon", 50),
            )
        )
        self.future_horizon = int(
            data_config.get(
                "prediction_horizon",
                data_config.get("future_horizon", 40),
            )
        )
        configured_action_horizon = data_config.get("action_condition_horizon")
        self.action_condition_horizon = (
            None
            if configured_action_horizon is None
            else int(configured_action_horizon)
        )
        self.joint_dim = int(model_config.get("joint_dim", 7))
        self.action_dim = int(model_config.get("action_dim", 7))
        self.hidden_dim = int(model_config.get("hidden_dim", 128))
        self.state_layers = int(
            model_config.get("state_layers", model_config.get("num_layers", 2))
        )
        self.action_layers = int(
            model_config.get("action_layers", model_config.get("num_layers", 2))
        )
        self.attention_heads = int(model_config.get("attention_heads", 4))
        self.flow_layers = int(model_config.get("flow_layers", 4))
        self.flow_attention_heads = int(
            model_config.get("flow_attention_heads", self.attention_heads)
        )
        self.flow_ffn_multiplier = int(
            model_config.get("flow_ffn_multiplier", 4)
        )
        self.flow_inference_steps = int(
            model_config.get("flow_inference_steps", 8)
        )
        self.flow_solver = str(model_config.get("flow_solver", "heun")).lower()
        self.contact_logit_scale = float(
            model_config.get("contact_logit_scale", 4.0)
        )
        self.dropout = float(model_config.get("dropout", 0.1))
        self.flow_dim = 2 * self.joint_dim + 1
        self._validate_config()

        state_dropout = self.dropout if self.state_layers > 1 else 0.0
        action_dropout = self.dropout if self.action_layers > 1 else 0.0
        self.state_encoder = nn.GRU(
            input_size=2 * self.joint_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.state_layers,
            dropout=state_dropout,
            batch_first=True,
        )
        self.action_encoder = nn.GRU(
            input_size=self.action_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.action_layers,
            dropout=action_dropout,
            batch_first=True,
        )
        self.state_token_norm = nn.LayerNorm(self.hidden_dim)
        self.action_token_norm = nn.LayerNorm(self.hidden_dim)

        # This direction is intentional: current state tokens query the known
        # future action target tokens.  The action sequence remains available
        # separately in condition_memory after this operation.
        self.state_queries_action = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.state_action_dropout = nn.Dropout(self.dropout)
        self.state_action_norm = nn.LayerNorm(self.hidden_dim)
        self.state_action_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.state_action_ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.global_condition_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.flow_input_projection = nn.Sequential(
            nn.LayerNorm(self.flow_dim),
            nn.Linear(self.flow_dim, self.hidden_dim),
        )
        self.flow_time_embedding = FlowTimeEmbedding(self.hidden_dim)
        self.flow_blocks = nn.ModuleList(
            FlowDecoderBlock(
                self.hidden_dim,
                self.flow_attention_heads,
                self.flow_ffn_multiplier,
                self.dropout,
            )
            for _ in range(self.flow_layers)
        )
        self.flow_output = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.flow_dim),
        )

    def _validate_config(self) -> None:
        positive = {
            "state_history_horizon": self.history_horizon,
            "prediction_horizon": self.future_horizon,
            "joint_dim": self.joint_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_layers": self.state_layers,
            "action_layers": self.action_layers,
            "attention_heads": self.attention_heads,
            "flow_layers": self.flow_layers,
            "flow_attention_heads": self.flow_attention_heads,
            "flow_ffn_multiplier": self.flow_ffn_multiplier,
            "flow_inference_steps": self.flow_inference_steps,
        }
        if self.action_condition_horizon is not None:
            positive["action_condition_horizon"] = self.action_condition_horizon
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {invalid}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by attention_heads"
            )
        if self.hidden_dim % self.flow_attention_heads != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by flow_attention_heads"
            )
        if self.flow_solver not in {"euler", "heun"}:
            raise ValueError("model.flow_solver must be 'euler' or 'heun'")
        if not math.isfinite(self.contact_logit_scale) or self.contact_logit_scale <= 0:
            raise ValueError("model.contact_logit_scale must be positive")

    @staticmethod
    def _require_sequence(
        batch: Mapping[str, torch.Tensor],
        key: str,
        *,
        horizon: int | None,
        feature_dim: int,
    ) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"missing batch key {key!r}")
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key!r} must be a tensor")
        if value.ndim != 3 or value.shape[-1] != feature_dim:
            expected_horizon = "A" if horizon is None else str(horizon)
            raise ValueError(
                f"{key!r} must have shape [B, {expected_horizon}, "
                f"{feature_dim}], got {tuple(value.shape)}"
            )
        if horizon is not None and value.shape[1] != horizon:
            raise ValueError(
                f"{key!r} must have horizon {horizon}, got {value.shape[1]}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{key!r} must be a floating-point tensor")
        return value

    def _condition_inputs(self, batch: Mapping[str, torch.Tensor]):
        q = self._require_sequence(
            batch,
            "q",
            horizon=self.history_horizon,
            feature_dim=self.joint_dim,
        )
        tau = self._require_sequence(
            batch,
            "tau",
            horizon=self.history_horizon,
            feature_dim=self.joint_dim,
        )
        action = self._require_sequence(
            batch,
            "target_relative_pose",
            horizon=self.action_condition_horizon,
            feature_dim=self.action_dim,
        )
        if q.shape[0] != tau.shape[0] or q.shape[0] != action.shape[0]:
            raise ValueError("q, tau, and action batch dimensions must match")
        if q.device != tau.device or q.device != action.device:
            raise ValueError("q, tau, and action must be on the same device")
        if q.dtype != tau.dtype or q.dtype != action.dtype:
            raise ValueError("q, tau, and action must use the same dtype")

        mask = batch.get("target_relative_pose_mask")
        if mask is None:
            valid_action = torch.ones(
                action.shape[:2], device=action.device, dtype=torch.bool
            )
        else:
            if not torch.is_tensor(mask) or tuple(mask.shape) != tuple(action.shape[:2]):
                actual = None if not torch.is_tensor(mask) else tuple(mask.shape)
                raise ValueError(
                    "'target_relative_pose_mask' must have shape [B, A], got "
                    f"{actual}"
                )
            mask = mask.to(device=action.device)
            if mask.dtype == torch.bool:
                valid_action = mask
            else:
                if not mask.is_floating_point() and mask.dtype not in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                ):
                    raise TypeError("target_relative_pose_mask must be bool or numeric")
                if not torch.isfinite(mask.to(dtype=torch.float32)).all():
                    raise ValueError("target_relative_pose_mask must be finite")
                valid_action = mask > 0
        if torch.any(valid_action.sum(dim=1) == 0):
            raise ValueError("each sample must contain at least one valid action token")
        return q, tau, action, valid_action

    def encode_conditions(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        q, tau, action, valid_action = self._condition_inputs(batch)
        state_input = torch.cat((q, tau), dim=-1)
        state_features, state_hidden = self.state_encoder(state_input)

        masked_action = action.masked_fill(~valid_action[..., None], 0.0)
        action_features, _ = self.action_encoder(masked_action)
        state_tokens = self.state_token_norm(
            state_features
            + _sinusoidal_position_encoding(
                state_features.shape[1], self.hidden_dim, state_features
            )
        )
        action_tokens = self.action_token_norm(
            action_features
            + _sinusoidal_position_encoding(
                action_features.shape[1], self.hidden_dim, action_features
            )
        )

        attended_action, attention_weights = self.state_queries_action(
            query=state_tokens,
            key=action_tokens,
            value=action_tokens,
            key_padding_mask=~valid_action,
            need_weights=True,
            average_attn_weights=False,
        )
        state_action_features = self.state_action_norm(
            state_tokens + self.state_action_dropout(attended_action)
        )
        state_action_features = self.state_action_ffn_norm(
            state_action_features
            + self.state_action_ffn(state_action_features)
        )

        # Preserve the action tokens instead of compressing the full plan into
        # one vector.  Future Flow tokens can attend to either stream.
        condition_memory = torch.cat(
            (state_action_features, action_tokens), dim=1
        )
        memory_padding_mask = torch.cat(
            (
                torch.zeros(
                    q.shape[0],
                    self.history_horizon,
                    device=q.device,
                    dtype=torch.bool,
                ),
                ~valid_action,
            ),
            dim=1,
        )
        action_denominator = valid_action.sum(dim=1, keepdim=True).to(q.dtype)
        pooled_action = (
            action_tokens * valid_action[..., None]
        ).sum(dim=1) / action_denominator
        global_condition = self.global_condition_projection(
            torch.cat((state_hidden[-1], pooled_action), dim=-1)
        )
        return {
            "state_features": state_tokens,
            "action_features": action_tokens,
            "state_action_features": state_action_features,
            "state_action_attention_weights": attention_weights,
            "target_relative_pose_mask": valid_action,
            "condition_memory": condition_memory,
            "condition_memory_padding_mask": memory_padding_mask,
            "global_condition": global_condition,
        }

    def _target_flow_state(
        self,
        batch: Mapping[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        q_future = self._require_sequence(
            batch,
            "q_future",
            horizon=self.future_horizon,
            feature_dim=self.joint_dim,
        )
        tau_future = self._require_sequence(
            batch,
            "tau_future",
            horizon=self.future_horizon,
            feature_dim=self.joint_dim,
        )
        contact = self._require_sequence(
            batch,
            "contact_future",
            horizon=self.future_horizon,
            feature_dim=1,
        )
        for key, value in (
            ("q_future", q_future),
            ("tau_future", tau_future),
            ("contact_future", contact),
        ):
            if value.shape[0] != reference.shape[0]:
                raise ValueError(f"{key!r} batch dimension does not match q")
            if value.device != reference.device or value.dtype != reference.dtype:
                value_type = f"{value.device}/{value.dtype}"
                reference_type = f"{reference.device}/{reference.dtype}"
                raise ValueError(
                    f"{key!r} must match condition device/dtype "
                    f"({value_type} != {reference_type})"
                )
        if torch.any((contact < 0.0) | (contact > 1.0)):
            raise ValueError("contact_future values must be in [0, 1]")
        contact_latent = (2.0 * contact - 1.0) * self.contact_logit_scale
        return torch.cat((q_future, tau_future, contact_latent), dim=-1)

    def _history_flow_source(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Causally align the latest observed history to the T-step source.

        When H >= T, the source is the latest T observations at their original
        cadence. When H < T, the first observation is left-padded. This keeps
        H and T independent without time-warping or introducing future data.
        """
        q, tau, _, _ = self._condition_inputs(batch)

        def align_history(value):
            if value.shape[1] >= self.future_horizon:
                return value[:, -self.future_horizon :]
            padding = value[:, :1].expand(
                -1, self.future_horizon - value.shape[1], -1
            )
            return torch.cat((padding, value), dim=1)

        q_source = align_history(q)
        tau_source = align_history(tau)
        contact_source = q_source.new_zeros(
            q_source.shape[0], self.future_horizon, 1
        )
        return torch.cat((q_source, tau_source, contact_source), dim=-1)

    def _prepare_flow_time(
        self,
        reference: torch.Tensor,
        flow_time: torch.Tensor | float | None,
    ) -> torch.Tensor:
        batch_size = reference.shape[0]
        if flow_time is None:
            if self.training:
                result = torch.rand(
                    batch_size,
                    1,
                    device=reference.device,
                    dtype=reference.dtype,
                )
            else:
                result = reference.new_full((batch_size, 1), 0.5)
        else:
            result = torch.as_tensor(
                flow_time,
                device=reference.device,
                dtype=reference.dtype,
            )
            if result.ndim == 0:
                result = result.expand(batch_size).reshape(batch_size, 1)
            elif result.ndim == 1:
                if result.numel() == 1:
                    result = result.expand(batch_size)
                if result.shape[0] != batch_size:
                    raise ValueError("flow_time batch dimension does not match q")
                result = result[:, None]
            elif tuple(result.shape) != (batch_size, 1):
                raise ValueError("flow_time must be scalar, [B], or [B, 1]")
        if not torch.isfinite(result).all() or torch.any(
            (result < 0.0) | (result > 1.0)
        ):
            raise ValueError("flow_time must be finite and in [0, 1]")
        return result

    def flow_velocity(
        self,
        trajectory_state: torch.Tensor,
        flow_time: torch.Tensor,
        encoded: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            trajectory_state.shape[0],
            self.future_horizon,
            self.flow_dim,
        )
        if trajectory_state.ndim != 3 or tuple(trajectory_state.shape) != expected:
            raise ValueError(
                "trajectory_state must have shape "
                f"[B, {self.future_horizon}, {self.flow_dim}], got "
                f"{tuple(trajectory_state.shape)}"
            )
        if tuple(flow_time.shape) != (trajectory_state.shape[0], 1):
            raise ValueError("flow_time must have shape [B, 1]")
        memory = encoded["condition_memory"]
        if memory.shape[0] != trajectory_state.shape[0]:
            raise ValueError("condition memory batch does not match trajectory")
        global_condition = encoded["global_condition"][:, None, :]
        flow_features = (
            self.flow_input_projection(trajectory_state)
            + self.flow_time_embedding(flow_time)[:, None, :]
            + _sinusoidal_position_encoding(
                self.future_horizon, self.hidden_dim, trajectory_state
            )
            + global_condition
        )
        for block in self.flow_blocks:
            flow_features = block(
                flow_features,
                memory,
                encoded.get("condition_memory_padding_mask"),
            )
        return self.flow_output(flow_features), flow_features

    def _decoded_output(self, flow_state: torch.Tensor) -> dict[str, torch.Tensor]:
        q_pred = flow_state[..., : self.joint_dim]
        tau_pred = flow_state[..., self.joint_dim : 2 * self.joint_dim]
        contact_logits = flow_state[..., 2 * self.joint_dim :]
        return {
            "flow_state_pred": flow_state,
            "joint_pred": flow_state[..., : 2 * self.joint_dim],
            "q_pred": q_pred,
            "tau_pred": tau_pred,
            "state_pred": {"q": q_pred, "tau": tau_pred},
            "contact_logit": contact_logits,
            "contact_logits": contact_logits,
            "contact_probability": torch.sigmoid(contact_logits),
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        flow_time: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate history-to-future conditional Flow Matching."""

        encoded = self.encode_conditions(batch)
        target_state = self._target_flow_state(batch, batch["q"])
        source_state = self._history_flow_source(batch)
        time = self._prepare_flow_time(target_state, flow_time)
        interpolation_time = time[:, None, :]
        interpolated = (
            (1.0 - interpolation_time) * source_state
            + interpolation_time * target_state
        )
        velocity_target = target_state - source_state
        velocity_pred, flow_features = self.flow_velocity(
            interpolated,
            time,
            encoded,
        )
        endpoint_estimate = interpolated + (
            1.0 - interpolation_time
        ) * velocity_pred
        result = {
            **encoded,
            "flow_source_state": source_state,
            "flow_target_state": target_state,
            "flow_interpolated": interpolated,
            "flow_time": time,
            "flow_velocity_pred": velocity_pred,
            "flow_velocity_target": velocity_target,
            "velocity_pred": velocity_pred,
            "velocity_target": velocity_target,
            "flow_features": flow_features,
        }
        result.update(self._decoded_output(endpoint_estimate))
        return result

    def integrate_flow(
        self,
        source_state: torch.Tensor,
        encoded: Mapping[str, torch.Tensor],
        *,
        steps: int | None = None,
        solver: str | None = None,
    ) -> torch.Tensor:
        """Integrate the learned ODE from observed history to future state."""

        steps = self.flow_inference_steps if steps is None else int(steps)
        solver = self.flow_solver if solver is None else str(solver).lower()
        if steps <= 0:
            raise ValueError("Flow integration steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        trajectory = source_state
        step_size = 1.0 / steps
        for step in range(steps):
            flow_time = trajectory.new_full(
                (trajectory.shape[0], 1), step / steps
            )
            first_velocity, _ = self.flow_velocity(
                trajectory, flow_time, encoded
            )
            if solver == "euler":
                trajectory = trajectory + step_size * first_velocity
                continue
            proposal = trajectory + step_size * first_velocity
            next_time = trajectory.new_full(
                (trajectory.shape[0], 1), (step + 1) / steps
            )
            second_velocity, _ = self.flow_velocity(
                proposal, next_time, encoded
            )
            trajectory = trajectory + 0.5 * step_size * (
                first_velocity + second_velocity
            )
        return trajectory

    @torch.no_grad()
    def predict(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        steps: int | None = None,
        solver: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Roll the complete aligned q/tau history into one future trajectory."""

        encoded = self.encode_conditions(batch)
        source = self._history_flow_source(batch)
        generated = self.integrate_flow(
            source,
            encoded,
            steps=steps,
            solver=solver,
        )
        result = {
            **encoded,
            "flow_source_state": source,
        }
        result.update(self._decoded_output(generated))
        return result


# Descriptive alias for callers that prefer the algorithm in the type name.
ConditionalFlowTorqueWorldModel = TorqueWorldModel
