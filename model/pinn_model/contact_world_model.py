"""GRU-token-conditioned Flow world model for configurable state streams."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn


SUPPORTED_STATE_STREAMS = ("q", "dq", "delta_q", "tau")
PREDICTED_STATE_STREAMS = SUPPORTED_STATE_STREAMS


class PhysicalTimeEmbedding(nn.Module):
    """Embed elapsed physical seconds instead of a token index."""

    def __init__(self, hidden_dim: int, frequencies=(1.0, 10.0, 25.0)):
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.as_tensor(frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(2 + 2 * len(frequencies), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, elapsed_seconds: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(elapsed_seconds)
        if value.ndim == 2:
            value = value[..., None]
        if value.ndim != 3 or value.shape[-1] != 1:
            raise ValueError(
                "physical time must have shape [B,T] or [B,T,1], got "
                f"{tuple(value.shape)}"
            )
        output_dtype = value.dtype
        value = value.to(dtype=self.projection[0].weight.dtype)
        frequencies = self.frequencies.to(device=value.device, dtype=value.dtype)
        phase = 2.0 * math.pi * value * frequencies
        features = [value, value.square()]
        for index in range(frequencies.numel()):
            features.extend(
                [
                    torch.sin(phase[..., index:index + 1]),
                    torch.cos(phase[..., index:index + 1]),
                ]
            )
        return self.projection(torch.cat(features, dim=-1)).to(dtype=output_dtype)


class FlowTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        if flow_time.ndim == 1:
            flow_time = flow_time[:, None]
        if flow_time.ndim != 2 or flow_time.shape[-1] != 1:
            raise ValueError(f"flow_time must have shape [B] or [B, 1], got {tuple(flow_time.shape)}")
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
    """One conditional flow block: self-attention, cross-attention, then FFN."""

    def __init__(self, hidden_dim, attention_heads, ffn_multiplier, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.condition_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trajectory, memory, memory_padding_mask=None):
        normalized = self.self_norm(trajectory)
        attended, _ = self.self_attention(
            query=normalized, key=normalized, value=normalized, need_weights=False
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


class StateToActionBlock(nn.Module):
    """Make state modality tokens action-aware before memory concatenation."""

    def __init__(self, hidden_dim, attention_heads, ffn_multiplier, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state_tokens, action_tokens, action_valid=None):
        key_padding_mask = None
        if action_valid is not None:
            key_padding_mask = ~action_valid
        attended, _ = self.cross_attention(
            query=self.query_norm(state_tokens),
            key=action_tokens,
            value=action_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = state_tokens + self.dropout(attended)
        return fused + self.dropout(self.ffn(self.ffn_norm(fused)))


class ContactWorldModel(nn.Module):
    """Generate selected future state streams and contact phase with CFM."""

    SUPPORTED_STATE_STREAMS = SUPPORTED_STATE_STREAMS
    PREDICTED_STATE_STREAMS = PREDICTED_STATE_STREAMS
    CONDITION_KEYS = (
        "q", "dq", "delta_q", "tau", "action", "action_mask",
        "action_time", "future_time",
    )
    TARGET_KEYS = tuple(f"{key}_future" for key in PREDICTED_STATE_STREAMS) + (
        "contact_future",
    )
    # The simplified token/memory architecture is not state-dict compatible
    # with the earlier CARS-WM implementation.
    MODEL_VERSION = "carswm_v3"

    def __init__(self, config: Mapping):
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        self._config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        self.history_horizon = int(data_config.get("state_history_horizon", 50))
        self.future_horizon = int(data_config.get("prediction_horizon", 40))
        self.action_condition_horizon = int(data_config.get("action_condition_horizon", 8))
        self.joint_dim = int(model_config.get("joint_dim", 7))
        self.action_dim = int(model_config.get("action_dim", 7))
        self.state_rate_hz = float(data_config.get("high_fps", 100.0))
        self.action_rate_hz = float(data_config.get("expert_fps", 25.0))
        self.action_start_offset = int(data_config.get("action_start_offset", 1))
        self.action_time_alignment = str(
            model_config.get("action_time_alignment", "zoh")
        ).lower()

        configured_inputs = model_config.get("inputs")
        if configured_inputs is None:
            raise ValueError("model.inputs is required and must select state streams")
        if isinstance(configured_inputs, str):
            configured_inputs = [configured_inputs]
        self.inputs = tuple(str(value).lower() for value in configured_inputs)
        if not self.inputs:
            raise ValueError("model.inputs must contain at least one state stream")
        unknown = sorted(set(self.inputs) - set(SUPPORTED_STATE_STREAMS))
        if unknown:
            raise ValueError(f"model.inputs contains unsupported values {unknown}; choose from {list(SUPPORTED_STATE_STREAMS)}")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("model.inputs must not contain duplicates")
        configured_outputs = model_config.get("outputs")
        if configured_outputs is None:
            configured_outputs = self.inputs
        if isinstance(configured_outputs, str):
            configured_outputs = [configured_outputs]
        self.outputs = tuple(str(value).lower() for value in configured_outputs)
        if not self.outputs:
            raise ValueError("model.outputs must contain at least one state stream")
        unknown = sorted(set(self.outputs) - set(SUPPORTED_STATE_STREAMS))
        if unknown:
            raise ValueError(
                "model.outputs contains unsupported values "
                f"{unknown}; choose from {list(SUPPORTED_STATE_STREAMS)}"
            )
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("model.outputs must not contain duplicates")
        # Inputs condition the model; outputs are the continuous streams
        # transported by the flow.  They intentionally need not be equal.
        self.predicted_state_streams = self.outputs
        # Expose the selected contract on instances while retaining the
        # module-level vocabulary constants used by the dataset.
        self.PREDICTED_STATE_STREAMS = self.outputs
        self.CONDITION_KEYS = self.inputs + (
            "action", "action_mask", "action_time", "future_time"
        )
        self.TARGET_KEYS = tuple(f"{key}_future" for key in self.outputs) + (
            "contact_future",
        )
        self.contact_state_count = int(model_config.get("contact_state_count", 3))
        if self.contact_state_count < 2:
            raise ValueError("model.contact_state_count must be at least 2")
        self.hidden_dim = int(model_config.get("hidden_dim", 128))
        self.state_layers = int(model_config.get("state_layers", 2))
        self.action_layers = int(model_config.get("action_layers", 2))
        self.flow_layers = int(model_config.get("flow_layers", 4))
        self.flow_attention_heads = int(model_config.get("flow_attention_heads", 4))
        self.flow_ffn_multiplier = int(model_config.get("flow_ffn_multiplier", 4))
        self.flow_inference_steps = int(model_config.get("flow_inference_steps", 8))
        self.flow_solver = str(model_config.get("flow_solver", "heun")).lower()
        self.flow_source_mode = str(model_config.get("flow_source_mode", "gaussian")).lower()
        # Keep early v3 checkpoints on their final-hidden path. New training
        # configs can opt into learned-query attention pooling.
        self.state_pooling = str(model_config.get("state_pooling", "last")).lower()
        self.dropout = float(model_config.get("dropout", 0.1))
        self.runtime_checks = bool(model_config.get("runtime_checks", True))
        self.use_action_padding_mask = bool(
            model_config.get("use_action_padding_mask", True)
        )
        self.emit_contact_probabilities = bool(
            model_config.get("emit_contact_probabilities", True)
        )
        # Contact is categorical and is deliberately not transported by the
        # continuous flow.  For the default four streams this is exactly 28.
        self.flow_dim = len(self.predicted_state_streams) * self.joint_dim
        self._validate_config()

        state_dropout = self.dropout if self.state_layers > 1 else 0.0
        action_dropout = self.dropout if self.action_layers > 1 else 0.0
        self.state_encoders = nn.ModuleDict({
            key: nn.GRU(
                self.joint_dim, self.hidden_dim, self.state_layers,
                dropout=state_dropout, batch_first=True
            )
            for key in self.inputs
        })
        # Keep each configured state stream independent until it becomes one
        # modal token.  Distinct embeddings preserve the stream identity even
        # when two modalities have similar numerical ranges.
        self.modality_embeddings = nn.ParameterDict()
        if self.state_pooling == "attention":
            self.state_pool_queries = nn.ParameterDict()
        for key in self.inputs:
            embedding = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.normal_(embedding, mean=0.0, std=0.02)
            self.modality_embeddings[key] = embedding
            if self.state_pooling == "attention":
                query = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
                nn.init.normal_(query, mean=0.0, std=0.02)
                self.state_pool_queries[key] = query
        if self.state_pooling == "attention":
            self.state_pool_attention = nn.MultiheadAttention(
                self.hidden_dim,
                self.flow_attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )
        self.action_encoder = nn.GRU(
            self.action_dim, self.hidden_dim, self.action_layers,
            dropout=action_dropout, batch_first=True
        )
        self.action_time_embedding = PhysicalTimeEmbedding(self.hidden_dim)
        self.future_time_embedding = PhysicalTimeEmbedding(self.hidden_dim)
        self.state_token_norm = nn.LayerNorm(self.hidden_dim)
        self.action_token_norm = nn.LayerNorm(self.hidden_dim)
        self.state_to_action_attention = StateToActionBlock(
            self.hidden_dim,
            int(model_config.get("state_to_action_attention_heads", self.flow_attention_heads)),
            self.flow_ffn_multiplier,
            self.dropout,
        )
        self.flow_input_projection = nn.Sequential(
            nn.LayerNorm(self.flow_dim), nn.Linear(self.flow_dim, self.hidden_dim)
        )
        self.flow_time_embedding = FlowTimeEmbedding(self.hidden_dim)
        self.flow_blocks = nn.ModuleList(
            FlowDecoderBlock(
                self.hidden_dim, self.flow_attention_heads,
                self.flow_ffn_multiplier, self.dropout
            )
            for _ in range(self.flow_layers)
        )
        self.flow_output = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.flow_dim)
        )
        contact_hidden_dim = int(
            model_config.get("contact_head_hidden_dim", max(self.hidden_dim // 2, 16))
        )
        self.contact_state_projection = nn.Sequential(
            nn.LayerNorm(self.flow_dim),
            nn.Linear(self.flow_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.contact_condition_norm = nn.LayerNorm(self.hidden_dim)
        self.contact_head = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, contact_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(contact_hidden_dim, self.contact_state_count),
        )

    def _validate_config(self):
        positive = {
            "state_history_horizon": self.history_horizon,
            "prediction_horizon": self.future_horizon,
            "joint_dim": self.joint_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_layers": self.state_layers,
            "action_layers": self.action_layers,
            "flow_layers": self.flow_layers,
            "flow_attention_heads": self.flow_attention_heads,
            "flow_ffn_multiplier": self.flow_ffn_multiplier,
            "flow_inference_steps": self.flow_inference_steps,
            "action_condition_horizon": self.action_condition_horizon,
            "state_to_action_attention_heads": int(
                self._config.get("model", {}).get(
                    "state_to_action_attention_heads", self.flow_attention_heads
                )
            ),
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {invalid}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.hidden_dim % self.flow_attention_heads != 0:
            raise ValueError("model.hidden_dim must be divisible by flow_attention_heads")
        state_to_action_heads = int(
            self._config.get("model", {}).get(
                "state_to_action_attention_heads", self.flow_attention_heads
            )
        )
        if state_to_action_heads <= 0 or self.hidden_dim % state_to_action_heads != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by state_to_action_attention_heads"
            )
        if self.flow_solver not in {"euler", "heun"}:
            raise ValueError("model.flow_solver must be 'euler' or 'heun'")
        if self.flow_source_mode != "gaussian":
            raise ValueError(
                "model.flow_source_mode must be 'gaussian'; state-to-state "
                "sources are no longer supported"
            )
        if self.state_pooling not in {"last", "attention"}:
            raise ValueError("model.state_pooling must be 'last' or 'attention'")
        if not math.isfinite(self.state_rate_hz) or self.state_rate_hz <= 0.0:
            raise ValueError("dataloader.high_fps must be positive")
        if not math.isfinite(self.action_rate_hz) or self.action_rate_hz <= 0.0:
            raise ValueError("dataloader.expert_fps must be positive")
        if self.action_time_alignment not in {"zoh", "linear"}:
            raise ValueError("model.action_time_alignment must be 'zoh' or 'linear'")

    def checkpoint_contract(self):
        """Return the complete semantic contract stored beside model weights."""

        data_config = self._config.get("dataloader") or {}
        action_config = self._config.get("action_contract") or {}
        contact_config = self._config.get("contact_gate") or {}
        thresholds = (contact_config.get("thresholds") or {}).get(
            str(contact_config.get("metric", "tau_ext_l1")).lower(), {}
        )
        return {
            "schema_version": 3,
            "model_version": self.MODEL_VERSION,
            "state_contract": "robot_state_streams_v1",
            "architecture": {
                "condition_encoder": (
                    "modality_gru_action_gru_state_to_action_cross_attention"
                ),
                "state_token": (
                    "attention_pooling"
                    if self.state_pooling == "attention"
                    else "final_gru_hidden"
                ),
                "flow_decoder": "self_attention_cross_attention_ffn",
                "condition_memory": "state_action_aware_state_plus_raw_action",
                "action_time_encoding": "physical_seconds_fourier_mlp",
                "future_time_encoding": "physical_seconds_fourier_mlp",
            },
            "input_state_streams": list(self.inputs),
            "predicted_continuous_streams": list(self.predicted_state_streams),
            "joint_dim": self.joint_dim,
            "history_horizon": self.history_horizon,
            "future_horizon": self.future_horizon,
            "action_horizon": self.action_condition_horizon,
            "state_rate_hz": self.state_rate_hz,
            "action_rate_hz": self.action_rate_hz,
            "time": {
                "state_rate_hz": self.state_rate_hz,
                "action_rate_hz": self.action_rate_hz,
                "action_start_offset": self.action_start_offset,
                "action_time_key": "action_time",
                "future_time_key": "future_time",
                "fallback": "regular_elapsed_seconds",
            },
            "action": {
                "semantic": str(
                    action_config.get("semantic", "expert_policy_action_target")
                ),
                "type": str(action_config.get("type", "absolute_ee_pose")),
                "dimension": self.action_dim,
                "coordinate_frame": str(
                    action_config.get("coordinate_frame", "link7")
                ),
                "representation": str(
                    action_config.get("representation", "xyz_quaternion")
                ),
                "quaternion_order": str(
                    action_config.get("quaternion_order", "xyzw")
                ),
                "quaternion_sign": str(
                    action_config.get("quaternion_sign", "canonical_w_nonnegative")
                ),
                "absolute_or_relative": str(
                    action_config.get("absolute_or_relative", "absolute")
                ),
                "dataset_key": str(data_config.get("action_key", "action.ee_pose")),
                "start_offset": int(data_config.get("action_start_offset", 1)),
                "dataset_alignment": str(
                    (self._config.get("train_data") or {}).get(
                        "action_alignment", data_config.get("action_resample", "previous")
                    )
                ),
                "future_token_alignment": self.action_time_alignment,
                "inference_delay_s": float(data_config.get("inference_delay_s", 0.0)),
            },
            "flow": {
                "dimension": self.flow_dim,
                "source": self.flow_source_mode,
                "solver": self.flow_solver,
                "steps": self.flow_inference_steps,
            },
            "contact": {
                "classes": (
                    ["free", "precontact_or_transition", "contact"]
                    if self.contact_state_count == 3
                    else [f"phase_{index}" for index in range(self.contact_state_count)]
                ),
                "label_mode": str(contact_config.get("label_mode", "three_phase")),
                "phase_label_mode": str(
                    contact_config.get("phase_label_mode", "transition_band")
                ),
                "tau_ext_source": str(
                    contact_config.get("tau_ext_source", "tau_measured_minus_tau_free")
                ),
                "norm": str(contact_config.get("metric", "tau_ext_l1")),
                "off_threshold": thresholds.get(
                    "off", thresholds.get(False, contact_config.get("off_threshold"))
                ),
                "on_threshold": thresholds.get(
                    "on", thresholds.get(True, contact_config.get("on_threshold"))
                ),
                "hysteresis_frames": int(contact_config.get("consecutive_frames", 3)),
                "precontact_frames": contact_config.get("precontact_frames"),
                "precontact_duration_s": contact_config.get("precontact_duration_s"),
            },
        }

    def validate_checkpoint_contract(self, actual):
        expected = self.checkpoint_contract()
        if not isinstance(actual, Mapping):
            raise ValueError(
                "checkpoint has no carswm_contract; legacy checkpoints are structurally "
                "incompatible and must be retrained"
            )
        if dict(actual) != expected:
            raise ValueError(
                "CARS-WM checkpoint contract mismatch: "
                f"checkpoint={dict(actual)!r}, expected={expected!r}"
            )
        return expected

    @staticmethod
    def _require_sequence(batch, key, *, horizon=None, feature_dim=None):
        if key not in batch:
            raise KeyError(f"missing batch key {key!r}")
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"{key!r} must be a tensor")
        if value.ndim != 3 or (feature_dim is not None and value.shape[-1] != feature_dim):
            raise ValueError(f"{key!r} must have shape [B, H, {feature_dim}], got {tuple(value.shape)}")
        if horizon is not None and value.shape[1] != horizon:
            raise ValueError(f"{key!r} must have horizon {horizon}, got {value.shape[1]}")
        if not value.is_floating_point():
            raise TypeError(f"{key!r} must be floating point")
        return value

    def _state_history(self, batch, key):
        return self._require_sequence(
            batch, key, horizon=self.history_horizon, feature_dim=self.joint_dim
        )

    def _action_inputs(self, batch):
        action = batch.get("action")
        if action is None:
            raise KeyError("missing batch key 'action'")
        action = self._require_sequence(
            {"action": action}, "action", horizon=self.action_condition_horizon, feature_dim=self.action_dim
        )
        mask = batch.get("action_mask")
        if not self.use_action_padding_mask:
            # The direct-action dataset used by the fast training path always
            # supplies a complete action chunk.  Returning an all-valid mask
            # preserves pooling semantics while allowing fused attention.
            return action, torch.ones(
                action.shape[:2], device=action.device, dtype=torch.bool
            )
        if mask is None:
            valid = torch.ones(action.shape[:2], device=action.device, dtype=torch.bool)
        else:
            if not torch.is_tensor(mask) or tuple(mask.shape) != tuple(action.shape[:2]):
                actual = None if not torch.is_tensor(mask) else tuple(mask.shape)
                raise ValueError(f"action_mask must have shape [B, A], got {actual}")
            valid = mask.to(device=action.device)
            if valid.dtype != torch.bool:
                if self.runtime_checks and not torch.isfinite(valid.to(dtype=torch.float32)).all():
                    raise ValueError("action_mask must be finite")
                valid = valid > 0
        if self.runtime_checks and torch.any(valid.sum(dim=1) == 0):
            raise ValueError("each sample must contain at least one valid action")
        return action, valid

    def _condition_inputs(self, batch):
        states = {key: self._state_history(batch, key) for key in self.inputs}
        action, valid = self._action_inputs(batch)
        reference = states[self.inputs[0]]
        for key, value in states.items():
            if value.shape[0] != reference.shape[0] or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{key} does not match the active state batch")
        if action.shape[0] != reference.shape[0] or action.device != reference.device or action.dtype != reference.dtype:
            raise ValueError("action does not match the state batch")
        return states, action, valid

    def _relative_time_values(
        self,
        batch,
        *,
        value_key,
        timestamp_key,
        length,
        reference,
        rate_hz,
        default_offset,
    ):
        """Return [B,T] elapsed seconds, preferring recorded timestamps."""

        value = batch.get(value_key)
        if value is not None:
            value = torch.as_tensor(value, device=reference.device)
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2 or value.shape != (reference.shape[0], length):
                raise ValueError(
                    f"{value_key} must have shape [B, {length}], got "
                    f"{tuple(value.shape)}"
                )
            return value.to(device=reference.device, dtype=reference.dtype)

        timestamps = batch.get(timestamp_key)
        history_timestamps = batch.get("history_timestamp_ns")
        if timestamps is not None and history_timestamps is not None:
            timestamps = torch.as_tensor(timestamps, device=reference.device)
            history_timestamps = torch.as_tensor(
                history_timestamps, device=reference.device
            )
            if timestamps.ndim == 3 and timestamps.shape[-1] == 1:
                timestamps = timestamps[..., 0]
            if history_timestamps.ndim == 3 and history_timestamps.shape[-1] == 1:
                history_timestamps = history_timestamps[..., 0]
            expected = (reference.shape[0], length)
            if tuple(timestamps.shape) != expected:
                raise ValueError(
                    f"{timestamp_key} must have shape [B, {length}], got "
                    f"{tuple(timestamps.shape)}"
                )
            if history_timestamps.ndim != 2 or history_timestamps.shape[0] != reference.shape[0]:
                raise ValueError(
                    "history_timestamp_ns must have one timestamp sequence per batch"
                )
            anchor = history_timestamps[:, -1:]
            return (timestamps.to(dtype=reference.dtype) - anchor.to(dtype=reference.dtype)) / 1.0e9

        offsets = torch.arange(
            length, device=reference.device, dtype=reference.dtype
        ) + float(default_offset)
        return offsets[None].expand(reference.shape[0], -1) / float(rate_hz)

    def encode_conditions(self, batch: Mapping[str, torch.Tensor]):
        states, action, valid_action = self._condition_inputs(batch)
        reference = states[self.inputs[0]]
        action_time = self._relative_time_values(
            batch,
            value_key="action_time",
            timestamp_key="action_chunk_timestamp_ns",
            length=action.shape[1],
            reference=reference,
            rate_hz=self.action_rate_hz,
            default_offset=self.action_start_offset,
        )
        future_time = self._relative_time_values(
            batch,
            value_key="future_time",
            timestamp_key="future_timestamp_ns",
            length=self.future_horizon,
            reference=reference,
            rate_hz=self.state_rate_hz,
            default_offset=1,
        )
        state_token_features = []
        for key in self.inputs:
            encoded, hidden = self.state_encoders[key](states[key])
            if self.state_pooling == "last":
                token = hidden[-1]
            else:
                # A learned query can assign high weight to a short impact,
                # reversal, or pre-contact transient anywhere in history.
                query = self.state_pool_queries[key].expand(
                    encoded.shape[0], -1, -1
                )
                pooled, _ = self.state_pool_attention(
                    query=query,
                    key=encoded,
                    value=encoded,
                    need_weights=False,
                )
                token = pooled[:, 0]
            token = token + self.modality_embeddings[key]
            state_token_features.append(token)
        # [B, M, d], where M=len(model.inputs).  Keeping M dynamic enables
        # input-stream ablations without changing the prediction head.
        state_features = torch.stack(state_token_features, dim=1)
        masked_action = action.masked_fill(~valid_action[..., None], 0.0)
        action_features, _ = self.action_encoder(masked_action)
        action_features = action_features + self.action_time_embedding(action_time)
        state_tokens = self.state_token_norm(state_features)
        action_tokens = self.action_token_norm(action_features)
        state_action_tokens = self.state_to_action_attention(
            state_tokens, action_tokens, valid_action
        )
        condition_memory = torch.cat((state_action_tokens, action_tokens), dim=1)
        memory_padding_mask = None
        if self.use_action_padding_mask:
            memory_padding_mask = torch.cat(
                (
                    torch.zeros(
                        state_tokens.shape[0],
                        state_tokens.shape[1],
                        device=state_tokens.device,
                        dtype=torch.bool,
                    ),
                    ~valid_action,
                ),
                dim=1,
            )
        result = {
            "predicted_state_streams": self.predicted_state_streams,
            "state_tokens": state_tokens,
            "state_action_tokens": state_action_tokens,
            "action_tokens": action_tokens,
            "action_time": action_time,
            "future_time": future_time,
            "condition_memory": condition_memory,
            "condition_memory_padding_mask": memory_padding_mask,
        }
        if memory_padding_mask is None:
            result["condition_summary"] = condition_memory.mean(dim=1)
        else:
            valid_memory = (~memory_padding_mask).to(condition_memory.dtype)
            result["condition_summary"] = (
                (condition_memory * valid_memory[..., None]).sum(dim=1)
                / valid_memory.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
        return result

    def _target_flow_state(self, batch, reference):
        values = []
        for key in self.predicted_state_streams:
            value = self._require_sequence(batch, f"{key}_future", horizon=self.future_horizon, feature_dim=self.joint_dim)
            if value.shape[0] != reference.shape[0] or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{key}_future does not match the condition batch")
            values.append(value)
        return torch.cat(values, dim=-1)

    def _gaussian_flow_source(self, reference, source_noise=None):
        """Return the independent Gaussian source for flow matching.

        History observations remain conditions only.  They are deliberately
        not copied into the source trajectory, so the model learns a genuine
        noise-to-future transport and does not inherit a state-to-state
        shortcut.  ``source_noise`` is injectable for OPD, where Teacher and
        Student must see the same Monte-Carlo source.
        """

        if not torch.is_tensor(reference) or reference.ndim != 3:
            raise ValueError("reference state must have shape [B, H, D]")
        shape = (reference.shape[0], self.future_horizon, self.flow_dim)
        if source_noise is None:
            return torch.randn(shape, device=reference.device, dtype=reference.dtype)
        if not torch.is_tensor(source_noise) or tuple(source_noise.shape) != shape:
            actual = None if not torch.is_tensor(source_noise) else tuple(source_noise.shape)
            raise ValueError(f"source_noise must have shape {shape}, got {actual}")
        if source_noise.device != reference.device or source_noise.dtype != reference.dtype:
            source_noise = source_noise.to(device=reference.device, dtype=reference.dtype)
        if self.runtime_checks and not torch.isfinite(source_noise).all():
            raise ValueError("source_noise must be finite")
        return source_noise

    def _prepare_flow_time(self, reference, flow_time):
        batch_size = reference.shape[0]
        if flow_time is None:
            result = torch.rand(batch_size, 1, device=reference.device, dtype=reference.dtype) if self.training else reference.new_full((batch_size, 1), 0.5)
        else:
            result = torch.as_tensor(flow_time, device=reference.device, dtype=reference.dtype)
            if result.ndim == 0:
                result = result.expand(batch_size).reshape(batch_size, 1)
            elif result.ndim == 1:
                if result.numel() == 1:
                    result = result.expand(batch_size)
                if result.shape[0] != batch_size:
                    raise ValueError("flow_time batch dimension does not match state")
                result = result[:, None]
            elif tuple(result.shape) != (batch_size, 1):
                raise ValueError("flow_time must be scalar, [B], or [B, 1]")
        if self.runtime_checks and (
            not torch.isfinite(result).all() or torch.any((result < 0) | (result > 1))
        ):
            raise ValueError("flow_time must be finite and in [0, 1]")
        return result

    def flow_velocity(self, trajectory_state, flow_time, encoded):
        expected = (trajectory_state.shape[0], self.future_horizon, self.flow_dim)
        if trajectory_state.ndim != 3 or tuple(trajectory_state.shape) != expected:
            raise ValueError(f"trajectory_state must have shape [B, {self.future_horizon}, {self.flow_dim}], got {tuple(trajectory_state.shape)}")
        if tuple(flow_time.shape) != (trajectory_state.shape[0], 1):
            raise ValueError("flow_time must have shape [B, 1]")
        features = (
            self.flow_input_projection(trajectory_state)
            + self.flow_time_embedding(flow_time)[:, None, :]
            + self.future_time_embedding(encoded["future_time"])
        )
        for block in self.flow_blocks:
            features = block(features, encoded["condition_memory"], encoded.get("condition_memory_padding_mask"))
        return self.flow_output(features), features

    def _time_aligned_action_features(self, encoded):
        """Map each high-rate future token to its action by physical time."""

        action = encoded["action_tokens"]
        action_time = encoded["action_time"].float()
        future_time = encoded["future_time"].float()
        if action_time.ndim != 2 or future_time.ndim != 2:
            raise ValueError("encoded physical times must have shape [B,T]")
        if self.action_time_alignment == "zoh":
            index = torch.searchsorted(action_time, future_time, right=True) - 1
            index = index.clamp(0, action.shape[1] - 1)
            return torch.gather(
                action,
                1,
                index[..., None].expand(-1, -1, action.shape[-1]),
            )
        right = torch.searchsorted(action_time, future_time, right=False)
        right = right.clamp(0, action.shape[1] - 1)
        left = (right - 1).clamp(0, action.shape[1] - 1)
        left_time = torch.gather(action_time, 1, left)
        right_time = torch.gather(action_time, 1, right)
        alpha = ((future_time - left_time) / (right_time - left_time).clamp_min(1.0e-6)).clamp(0.0, 1.0)
        left_features = torch.gather(
            action, 1, left[..., None].expand(-1, -1, action.shape[-1])
        )
        right_features = torch.gather(
            action, 1, right[..., None].expand(-1, -1, action.shape[-1])
        )
        return (1.0 - alpha[..., None]) * left_features + alpha[..., None] * right_features

    def contact_logits(self, continuous_trajectory, encoded):
        expected = (
            continuous_trajectory.shape[0],
            self.future_horizon,
            self.flow_dim,
        )
        if continuous_trajectory.ndim != 3 or tuple(continuous_trajectory.shape) != expected:
            raise ValueError(
                "continuous_trajectory must have shape "
                f"[B, {self.future_horizon}, {self.flow_dim}]"
            )
        state = self.contact_state_projection(continuous_trajectory)
        action = self._time_aligned_action_features(encoded)
        condition = encoded.get("condition_summary")
        if condition is None:
            condition = encoded["condition_memory"].mean(dim=1)
        condition = self.contact_condition_norm(condition)[:, None, :].expand(
            -1, self.future_horizon, -1
        )
        return self.contact_head(torch.cat((state, action, condition), dim=-1))

    def _decoded_output(self, flow_state, encoded):
        result = {"flow_state_pred": flow_state}
        offset = 0
        for key in self.predicted_state_streams:
            result[f"{key}_pred"] = flow_state[..., offset:offset + self.joint_dim]
            offset += self.joint_dim
        logits = self.contact_logits(flow_state, encoded)
        result["contact_logits"] = logits
        if self.emit_contact_probabilities or not self.training:
            probability = torch.softmax(logits, dim=-1)
            state = probability.argmax(dim=-1, keepdim=True).to(flow_state.dtype)
            result.update(
                {
                    "contact_probability": probability,
                    "contact_state_pred": state,
                }
            )
        return result

    def forward(self, batch, *, flow_time=None, source_noise=None):
        encoded = self.encode_conditions(batch)
        reference = batch[self.inputs[0]]
        target_state = self._target_flow_state(batch, reference)
        source_state = self._gaussian_flow_source(reference, source_noise)
        time = self._prepare_flow_time(target_state, flow_time)
        interpolation_time = time[:, None, :]
        interpolated = (1.0 - interpolation_time) * source_state + interpolation_time * target_state
        velocity_target = target_state - source_state
        velocity_pred, flow_features = self.flow_velocity(interpolated, time, encoded)
        endpoint = interpolated + (1.0 - interpolation_time) * velocity_pred
        result = {
            **encoded,
            "flow_source_state": source_state,
            "flow_source_noise": source_state,
            "flow_target_state": target_state,
            "flow_interpolated": interpolated,
            "flow_time": time,
            "flow_velocity_pred": velocity_pred,
            "flow_velocity_target": velocity_target,
            "velocity_pred": velocity_pred,
            "velocity_target": velocity_target,
            "flow_features": flow_features,
        }
        result.update(self._decoded_output(endpoint, encoded))
        return result

    def integrate_flow(self, source_state, encoded, *, steps=None, solver=None):
        steps = self.flow_inference_steps if steps is None else int(steps)
        solver = self.flow_solver if solver is None else str(solver).lower()
        if steps <= 0:
            raise ValueError("Flow integration steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        trajectory = source_state
        step_size = 1.0 / steps
        for step in range(steps):
            flow_time = trajectory.new_full((trajectory.shape[0], 1), step / steps)
            first, _ = self.flow_velocity(trajectory, flow_time, encoded)
            if solver == "euler":
                trajectory = trajectory + step_size * first
                continue
            proposal = trajectory + step_size * first
            next_time = trajectory.new_full((trajectory.shape[0], 1), (step + 1) / steps)
            second, _ = self.flow_velocity(proposal, next_time, encoded)
            trajectory = trajectory + 0.5 * step_size * (first + second)
        return trajectory

    @torch.no_grad()
    def predict(self, batch, *, steps=None, solver=None, source_noise=None):
        encoded = self.encode_conditions(batch)
        reference = batch[self.inputs[0]]
        source = self._gaussian_flow_source(reference, source_noise)
        generated = self.integrate_flow(source, encoded, steps=steps, solver=solver)
        result = {
            **encoded,
            "flow_source_state": source,
            "flow_source_noise": source,
        }
        result.update(self._decoded_output(generated, encoded))
        return result

    @torch.no_grad()
    def sample(self, batch, *, num_samples=1, steps=None, solver=None, source_noise=None):
        """Draw K conditional futures and retain the sample dimension."""

        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if source_noise is not None:
            expected = (
                batch[self.inputs[0]].shape[0],
                num_samples,
                self.future_horizon,
                self.flow_dim,
            )
            if not torch.is_tensor(source_noise) or tuple(source_noise.shape) != expected:
                actual = None if not torch.is_tensor(source_noise) else tuple(source_noise.shape)
                raise ValueError(f"source_noise must have shape {expected}, got {actual}")
        draws = [
            self.predict(
                batch,
                steps=steps,
                solver=solver,
                source_noise=(None if source_noise is None else source_noise[:, index]),
            )
            for index in range(num_samples)
        ]
        keys = [f"{key}_pred" for key in self.predicted_state_streams]
        keys.extend(("flow_state_pred", "contact_logits", "contact_probability", "contact_state_pred"))
        return {
            key: torch.stack([draw[key] for draw in draws], dim=1)
            for key in keys
            if all(key in draw for draw in draws)
        }

    def predict_differentiable(self, batch, *, steps=None, solver=None, source_noise=None):
        encoded = self.encode_conditions(batch)
        reference = batch[self.inputs[0]]
        source = self._gaussian_flow_source(reference, source_noise)
        generated = self.integrate_flow(source, encoded, steps=steps, solver=solver)
        result = {
            **encoded,
            "flow_source_state": source,
            "flow_source_noise": source,
        }
        result.update(self._decoded_output(generated, encoded))
        return result
