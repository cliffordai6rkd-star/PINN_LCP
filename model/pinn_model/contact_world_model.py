"""GRU-token-conditioned Flow world model for configurable state streams."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn


SUPPORTED_STATE_STREAMS = ("q", "dq", "delta_q", "tau")
PREDICTED_STATE_STREAMS = SUPPORTED_STATE_STREAMS


def _npu_legacy_attention(device: torch.device) -> bool:
    """Use the non-SDPA attention path on Ascend NPU.

    Current torch_npu releases can fail when fused SDPA receives tensors with
    different internal formats (``can not cast format when output is input``).
    Requesting attention weights forces PyTorch's compatible bmm/softmax path.
    """
    return torch.device(device).type == "npu"


def _run_gru_compat(gru: nn.GRU, value: torch.Tensor):
    """Run GRU through an NPU-safe FP32 island.

    Several torch_npu releases fail while compiling the fused FP16 GRU
    kernel (``tiling offset out of range``).  Keeping only the recurrent
    operator in FP32 avoids that kernel; the surrounding model remains under
    the configured autocast context.  CUDA/CPU use the original fast path.
    """
    if value.device.type != "npu":
        return gru(value)
    original_dtype = value.dtype
    with torch.autocast(device_type="npu", enabled=False):
        encoded, hidden = gru(value.float().contiguous())
    if original_dtype != encoded.dtype:
        encoded = encoded.to(dtype=original_dtype)
        hidden = hidden.to(dtype=original_dtype)
    return encoded, hidden


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
    """Future self-attention, parallel history/action cross-attention, then FFN."""

    def __init__(self, hidden_dim, attention_heads, ffn_multiplier, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.history_cross_attn = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.action_cross_attn = nn.MultiheadAttention(
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

    def forward(self, trajectory, history, action, action_padding_mask=None):
        normalized = self.self_norm(trajectory)
        attended, _ = self.self_attention(
            query=normalized.contiguous(),
            key=normalized.contiguous(),
            value=normalized.contiguous(),
            need_weights=_npu_legacy_attention(normalized.device),
        )
        trajectory = trajectory + self.dropout(attended)
        history_attended, _ = self.history_cross_attn(
            query=self.history_norm(trajectory).contiguous(),
            key=history.contiguous(),
            value=history.contiguous(),
            need_weights=_npu_legacy_attention(trajectory.device),
        )
        action_attended, _ = self.action_cross_attn(
            query=self.action_norm(trajectory).contiguous(),
            key=action.contiguous(),
            value=action.contiguous(),
            key_padding_mask=action_padding_mask,
            need_weights=_npu_legacy_attention(trajectory.device),
        )
        trajectory = trajectory + self.dropout(history_attended) + self.dropout(action_attended)
        return trajectory + self.dropout(self.ffn(self.ffn_norm(trajectory)))


class ContactWorldModel(nn.Module):
    """Generate selected future state streams and contact phase with CFM."""

    SUPPORTED_STATE_STREAMS = SUPPORTED_STATE_STREAMS
    PREDICTED_STATE_STREAMS = PREDICTED_STATE_STREAMS
    CONDITION_KEYS = (
        "q", "dq", "delta_q", "tau", "action", "action_mask",
    )
    TARGET_KEYS = tuple(f"{key}_future" for key in PREDICTED_STATE_STREAMS) + (
        "contact_future",
    )
    # Shared learned history recency positions change the checkpoint contract.
    MODEL_VERSION = "carswm_v7"

    def __init__(self, config: Mapping):
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        self._config = config
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        if "state_to_action_attention_heads" in model_config:
            raise ValueError("model.state_to_action_attention_heads was removed")
        train_config = config.get("train") or {}
        downsample = train_config.get("downsample", False)
        self.temporal_stride = (2 if downsample is True else 1) if isinstance(downsample, bool) else int(downsample)
        if self.temporal_stride < 1:
            raise ValueError("train.downsample must be false/true or a positive integer")
        self.temporal_downsample = self.temporal_stride > 1
        self.external_history_horizon = int(data_config.get("state_history_horizon", 50))
        self.external_future_horizon = int(data_config.get("prediction_horizon", 40))
        self.external_action_condition_horizon = int(data_config.get("action_condition_horizon", 8))
        for name, value in (("history", self.external_history_horizon), ("future", self.external_future_horizon)):
            if value % self.temporal_stride:
                raise ValueError(f"{name} horizon {value} must be divisible by temporal stride {self.temporal_stride}")
        self.history_horizon = self.external_history_horizon // self.temporal_stride
        self.future_horizon = self.external_future_horizon // self.temporal_stride
        self.action_condition_horizon = self.external_action_condition_horizon
        self.joint_dim = int(model_config.get("joint_dim", 7))
        self.action_dim = int(model_config.get("action_dim", 7))
        self.external_state_rate_hz = float(data_config.get("high_fps", 100.0))
        self.external_action_rate_hz = float(data_config.get("expert_fps", 25.0))
        self.state_rate_hz = self.external_state_rate_hz / self.temporal_stride
        self.action_rate_hz = self.external_action_rate_hz
        self.action_start_offset = int(data_config.get("action_start_offset", 1))
        removed = {
            "action_time_alignment", "use_physical_time",
            "action_time_encoding", "future_time_encoding",
        } & model_config.keys()
        if removed:
            raise ValueError(f"Removed model time-encoding options: {sorted(removed)}")
        # During training, complete free-motion windows can optionally hide
        # their historical measured torque.  The future torque target remains
        # untouched; this augmentation only reduces shortcut dependence on
        # tau history.  Keep the legacy key as a compatibility alias.
        self.tau_history_mask_warmup_steps = int(
            train_config.get(
                "tau_history_mask_warmup_steps",
                train_config.get("tau_free_warmup_steps", 0),
            )
        )
        if self.tau_history_mask_warmup_steps < 0:
            raise ValueError("train.tau_history_mask_warmup_steps must be non-negative")
        self._global_step = 0

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
            "action", "action_mask"
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
        if "state_pooling" in model_config:
            raise ValueError("model.state_pooling was removed; all GRU temporal outputs are retained")
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
        # Broadcast each stream's identity to all of its temporal GRU outputs.
        self.modality_embeddings = nn.ParameterDict()
        for key in self.inputs:
            embedding = nn.Parameter(torch.empty(self.hidden_dim))
            nn.init.normal_(embedding, mean=0.0, std=0.02)
            self.modality_embeddings[key] = embedding
        self.action_encoder = nn.GRU(
            self.action_dim, self.hidden_dim, self.action_layers,
            dropout=action_dropout, batch_first=True
        )
        self.action_pos_embedding = nn.Embedding(
            self.action_condition_horizon, self.hidden_dim
        )
        self.future_pos_embedding = nn.Embedding(self.future_horizon, self.hidden_dim)
        self.history_pos_embedding = nn.Embedding(self.history_horizon, self.hidden_dim)
        self.state_token_norm = nn.LayerNorm(self.hidden_dim)
        self.action_token_norm = nn.LayerNorm(self.hidden_dim)
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
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {invalid}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.hidden_dim % self.flow_attention_heads != 0:
            raise ValueError("model.hidden_dim must be divisible by flow_attention_heads")
        if self.flow_solver not in {"euler", "heun"}:
            raise ValueError("model.flow_solver must be 'euler' or 'heun'")
        if self.flow_source_mode != "gaussian":
            raise ValueError(
                "model.flow_source_mode must be 'gaussian'; state-to-state "
                "sources are no longer supported"
            )
        if not math.isfinite(self.state_rate_hz) or self.state_rate_hz <= 0.0:
            raise ValueError("dataloader.high_fps must be positive")
        if not math.isfinite(self.action_rate_hz) or self.action_rate_hz <= 0.0:
            raise ValueError("dataloader.expert_fps must be positive")

    def checkpoint_contract(self):
        """Return the complete semantic contract stored beside model weights."""

        data_config = self._config.get("dataloader") or {}
        action_config = self._config.get("action_contract") or {}
        contact_config = self._config.get("contact_gate") or {}
        thresholds = (contact_config.get("thresholds") or {}).get(
            str(contact_config.get("metric", "tau_ext_l1")).lower(), {}
        )
        return {
            "schema_version": 8,
            "model_version": self.MODEL_VERSION,
            "state_contract": "robot_state_streams_v1",
            "architecture": {
                "condition_encoder": (
                    "modality_gru_action_gru"
                ),
                "state_token": "all_gru_temporal_outputs_modality_major",
                "history_position_encoding": "shared_learned_recency_index_newest_zero",
                "flow_decoder": "self_attention_parallel_dual_cross_attention_residual_sum_ffn",
                "condition_memories": "independent_history_and_action",
                "action_position_encoding": "learned_sequence_index",
                "future_position_encoding": "learned_sequence_index",
            },
            "input_state_streams": list(self.inputs),
            "predicted_continuous_streams": list(self.predicted_state_streams),
            "joint_dim": self.joint_dim,
            "history_horizon": self.history_horizon,
            "future_horizon": self.future_horizon,
            "action_horizon": self.action_condition_horizon,
            "external_history_horizon": self.external_history_horizon,
            "external_future_horizon": self.external_future_horizon,
            "external_action_horizon": self.external_action_condition_horizon,
            "temporal_downsample": self.temporal_downsample,
            "temporal_stride": self.temporal_stride,
            "external_state_rate_hz": self.external_state_rate_hz,
            "internal_state_rate_hz": self.state_rate_hz,
            "state_rate_hz": self.state_rate_hz,
            "action_rate_hz": self.action_rate_hz,
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
                "future_token_alignment": "nominal_rate_zoh",
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

    def prepare_batch(self, batch: Mapping[str, torch.Tensor]):
        """Downsample state windows only; preserve every recorded action token."""
        if self.temporal_stride == 1:
            return batch
        result = dict(batch)
        for key, value in batch.items():
            if not torch.is_tensor(value) or value.ndim < 2:
                continue
            if key in self.inputs or key == "contact":
                expected = self.external_history_horizon
            elif key.endswith("_future") or key.endswith("_future_raw") or key in {"future_time", "future_timestamp_ns", "future_indices"}:
                expected = self.external_future_horizon
            elif key in {"history_timestamp_ns", "history_indices"}:
                expected = self.external_history_horizon
            else:
                continue
            if value.shape[1] == expected:
                # History must still end at the current 100 Hz state anchor.
                is_history = key in self.inputs or key in {
                    "contact", "history_timestamp_ns", "history_indices"
                }
                offset = self.temporal_stride - 1 if is_history else 0
                result[key] = value[:, offset :: self.temporal_stride, ...]
            elif value.shape[1] != expected // self.temporal_stride:
                raise ValueError(f"{key} has temporal length {value.shape[1]}, expected {expected} or {expected // self.temporal_stride}")
        return result

    def set_global_step(self, global_step: int):
        """Set optimizer-update progress used by training-only curricula."""
        self._global_step = max(int(global_step), 0)

    def _mask_tau_history(self, batch: Mapping[str, torch.Tensor]):
        """Curricularly mask historical tau on complete free-motion windows."""
        reference = batch.get("tau")
        device = reference.device if torch.is_tensor(reference) else None
        if (
            not self.training
            or "tau" not in self.inputs
            or not torch.is_tensor(reference)
            or self.tau_history_mask_warmup_steps == 0
        ):
            zero = reference.new_zeros(()) if torch.is_tensor(reference) else torch.tensor(0.0)
            return dict(batch), zero, zero, zero

        probability = min(
            1.0,
            self._global_step / float(self.tau_history_mask_warmup_steps),
        )
        history_contact = batch.get("contact")
        future_contact = batch.get("contact_future")
        if not torch.is_tensor(history_contact) or not torch.is_tensor(future_contact):
            zero = reference.new_zeros(())
            return dict(batch), reference.new_tensor(probability), zero, zero
        history_contact = history_contact.to(device=device)
        future_contact = future_contact.to(device=device)
        if history_contact.shape[0] != reference.shape[0] or future_contact.shape[0] != reference.shape[0]:
            raise ValueError("contact and tau must have the same batch dimension")
        free_history = history_contact.reshape(reference.shape[0], -1).round().eq(0).all(dim=1)
        free_future = future_contact.reshape(reference.shape[0], -1).round().eq(0).all(dim=1)
        free_window = free_history & free_future
        sampled = torch.rand(reference.shape[0], device=device) < probability
        mask_samples = free_window & sampled
        masked = dict(batch)
        if torch.any(mask_samples):
            tau = reference.clone()
            tau[mask_samples] = 0.0
            masked["tau"] = tau
        return (
            masked,
            reference.new_tensor(probability),
            mask_samples.to(dtype=reference.dtype).mean(),
            free_window.to(dtype=reference.dtype).mean(),
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
            # preserves all action tokens while allowing fused attention.
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

    def encode_conditions(self, batch: Mapping[str, torch.Tensor]):
        batch = self.prepare_batch(batch)
        batch, tau_mask_probability, tau_mask_fraction, tau_free_fraction = self._mask_tau_history(batch)
        states, action, valid_action = self._condition_inputs(batch)
        state_token_features = []
        for key in self.inputs:
            sequence, _ = _run_gru_compat(self.state_encoders[key], states[key])
            # GRU outputs remain oldest-to-newest; recency zero is current.
            positions = torch.arange(sequence.shape[1] - 1, -1, -1, device=sequence.device)
            history_position = self.history_pos_embedding(positions)[None].to(sequence.dtype)
            state_token_features.append(
                sequence + self.modality_embeddings[key] + history_position
            )
        # [B, M*T, D]: modality-major concatenation preserves each stream's
        # chronological order; T is the actual internal history length.
        state_features = torch.cat(state_token_features, dim=1)
        masked_action = action.masked_fill(~valid_action[..., None], 0.0)
        action_features, _ = _run_gru_compat(self.action_encoder, masked_action)
        positions = torch.arange(action.shape[1], device=action.device)
        action_features = action_features + self.action_pos_embedding(positions)[None].to(
            action_features.dtype
        )
        state_tokens = self.state_token_norm(state_features)
        action_tokens = self.action_token_norm(action_features)
        action_padding_mask = ~valid_action if self.use_action_padding_mask else None
        result = {
            "predicted_state_streams": self.predicted_state_streams,
            "state_tokens": state_tokens,
            "action_tokens": action_tokens,
            "action_padding_mask": action_padding_mask,
            "tau_history_mask_probability": tau_mask_probability,
            "tau_history_mask_fraction": tau_mask_fraction,
            "tau_history_free_fraction": tau_free_fraction,
            "_prepared_batch": batch,
        }
        # Preserve the contact head's token-weighted summary without building
        # a concatenated memory for decoder attention.
        action_weights = valid_action.to(action_tokens.dtype)
        result["condition_summary"] = (
            state_tokens.sum(dim=1)
            + (action_tokens * action_weights[..., None]).sum(dim=1)
        ) / (state_tokens.shape[1] + action_weights.sum(dim=1, keepdim=True))
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
            + self.future_pos_embedding(
                torch.arange(trajectory_state.shape[1], device=trajectory_state.device)
            )[None].to(trajectory_state.dtype)
        )
        for block in self.flow_blocks:
            features = block(
                features, encoded["state_tokens"], encoded["action_tokens"],
                encoded["action_padding_mask"],
            )
        return self.flow_output(features), features

    def _aligned_action_features(self, encoded):
        """Preserve nominal-rate ZOH contact alignment without timestamp inputs.

        This selects contact features only; trajectory PE uses sequence indices.
        """
        action = encoded["action_tokens"]
        future_indices = torch.arange(self.future_horizon, device=action.device)
        index = torch.floor(
            (future_indices + 1).float() * self.action_rate_hz / self.state_rate_hz
        ).long() - self.action_start_offset
        return action.index_select(1, index.clamp(0, action.shape[1] - 1))

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
        action = self._aligned_action_features(encoded)
        condition = encoded["condition_summary"]
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

    def _expand_external_outputs(self, result):
        if self.temporal_stride == 1:
            return result
        expanded = dict(result)
        keys = [f"{key}_pred" for key in self.predicted_state_streams]
        keys.extend(("flow_state_pred", "contact_logits", "contact_probability", "contact_state_pred"))
        for key in keys:
            value = expanded.get(key)
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == self.future_horizon:
                expanded[key] = value.repeat_interleave(self.temporal_stride, dim=1)
        return expanded

    def forward(self, batch, *, flow_time=None, source_noise=None):
        batch = self.prepare_batch(batch)
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
            "_prepared_batch": encoded.get("_prepared_batch", batch),
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
        batch = self.prepare_batch(batch)
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
        return self._expand_external_outputs(result)

    @torch.no_grad()
    def sample(self, batch, *, num_samples=1, steps=None, solver=None, source_noise=None):
        """Draw K conditional futures and retain the sample dimension."""

        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        batch = self.prepare_batch(batch)
        if source_noise is not None:
            if (torch.is_tensor(source_noise) and self.temporal_stride > 1
                    and source_noise.ndim == 4
                    and source_noise.shape[2] == self.external_future_horizon):
                source_noise = source_noise[:, :, :: self.temporal_stride, ...].contiguous()
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
        batch = self.prepare_batch(batch)
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
        result["_prepared_batch"] = batch
        return result
