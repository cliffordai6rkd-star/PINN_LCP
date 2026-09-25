"""Independent GRU-conditioned, future-query Transformer state baseline."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import nn


SUPPORTED_STATE_STREAMS = ("q", "dq", "delta_q", "tau")
MODEL_VERSION = "deterministic_wm_v1"


def _run_gru(gru, value):
    # Keep the recurrent kernel in FP32 on Ascend, as in the WM encoder.
    if value.device.type != "npu":
        return gru(value)[0]
    with torch.autocast(device_type="npu", enabled=False):
        sequence, _ = gru(value.float().contiguous())
    return sequence.to(value.dtype)


def _attend(layer, query, memory, padding_mask=None):
    # Ascend's legacy bmm path avoids fused SDPA internal-format failures.
    return layer(
        query.contiguous(), memory.contiguous(), memory.contiguous(),
        key_padding_mask=padding_mask, need_weights=query.device.type == "npu",
    )[0]


def _streams(value, name):
    values = (value,) if isinstance(value, str) else tuple(value)
    if (not values or len(set(values)) != len(values)
            or set(values) - set(SUPPORTED_STATE_STREAMS)):
        raise ValueError(f"model.{name} must select unique streams from {SUPPORTED_STATE_STREAMS}")
    return values


class DeterministicDecoderBlock(nn.Module):
    """Pre-norm self attention, parallel history/action attention, and FFN."""

    def __init__(self, hidden_dim, heads, ffn_multiplier, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.history_norm = nn.LayerNorm(hidden_dim)
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.history_cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.action_cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_multiplier * hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(ffn_multiplier * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, future, history, action, padding_mask=None):
        normalized = self.self_norm(future)
        future = future + self.dropout(_attend(self.self_attention, normalized, normalized))
        history_update = _attend(self.history_cross_attn, self.history_norm(future), history)
        action_update = _attend(self.action_cross_attn, self.action_norm(future), action, padding_mask)
        future = future + self.dropout(history_update) + self.dropout(action_update)
        return future + self.dropout(self.ffn(self.ffn_norm(future)))


class _ContactTemporalBlock(nn.Module):
    """Equivalent pre-norm temporal encoder with an explicit NPU attention path."""

    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        normalized = self.norm1(features)
        features = features + self.dropout(_attend(self.attention, normalized, normalized))
        return features + self.dropout(self.ffn(self.norm2(features)))


class DeterministicRobotStateWorldModel(nn.Module):
    """Predict one complete normalized future; targets are never model inputs.

    forward() uses the internal temporal grid. predict() restores the external
    grid by repeating frames, matching the existing WM inference convention.
    Call eval() for inference (training retains ordinary dropout).
    """

    MODEL_VERSION = MODEL_VERSION
    SUPPORTED_STATE_STREAMS = SUPPORTED_STATE_STREAMS
    is_deterministic = True

    def __init__(self, config: Mapping):
        super().__init__()
        data = config.get("dataloader") or {}
        model = config.get("model") or {}
        train = config.get("train") or {}
        self.inputs = _streams(model.get("inputs", SUPPORTED_STATE_STREAMS), "inputs")
        self.outputs = _streams(model.get("outputs", ("q", "tau")), "outputs")
        self.predicted_state_streams = self.outputs
        self.PREDICTED_STATE_STREAMS = self.outputs
        self.CONDITION_KEYS = self.inputs + ("action", "action_mask")
        self.TARGET_KEYS = tuple(f"{key}_future" for key in self.outputs) + ("contact_future",)
        stride = train.get("downsample", False)
        if isinstance(stride, bool):
            stride = 2 if stride else 1
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("train.downsample must be false/true or a positive integer")
        self.temporal_stride = stride
        self.temporal_downsample = stride > 1
        if data.get("prediction_horizon", 16) is None:
            raise ValueError("dataloader.prediction_horizon must be set explicitly")
        self.external_history_horizon = int(data.get("state_history_horizon", 50))
        self.external_future_horizon = int(data.get("prediction_horizon", 16))
        self.external_action_condition_horizon = int(data.get("action_condition_horizon", 8))
        for horizon in (self.external_history_horizon, self.external_future_horizon):
            if horizon <= 0 or horizon % stride:
                raise ValueError("history/future horizons must be positive and divisible by temporal stride")
        self.history_horizon = self.external_history_horizon // stride
        self.future_horizon = self.external_future_horizon // stride
        self.action_condition_horizon = self.external_action_condition_horizon
        for key, default in (("joint_dim", 7), ("action_dim", 7), ("hidden_dim", 128),
                             ("state_layers", 2), ("action_layers", 2), ("decoder_layers", 4),
                             ("attention_heads", 4), ("ffn_multiplier", 4), ("contact_state_count", 3)):
            value = int(model.get(key, default))
            if value < 1:
                raise ValueError(f"model.{key} must be positive")
            setattr(self, key, value)
        if self.action_condition_horizon < 1 or self.contact_state_count < 2:
            raise ValueError("action horizon must be positive and contact_state_count at least two")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.dropout = float(model.get("dropout", 0.01))
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.use_action_padding_mask = bool(model.get("use_action_padding_mask", True))
        self.emit_contact_probabilities = bool(model.get("emit_contact_probabilities", False))
        self.external_state_rate_hz = float(data.get("high_fps", 100.0))
        self.action_rate_hz = float(data.get("expert_fps", 25.0))
        if any(not math.isfinite(v) or v <= 0 for v in (self.external_state_rate_hz, self.action_rate_hz)):
            raise ValueError("state and action rates must be finite and positive")
        self.state_rate_hz = self.external_state_rate_hz / stride
        self.action_start_offset = int(data.get("action_start_offset", 1))
        self.output_dim = len(self.outputs) * self.joint_dim
        self.contact_head_hidden_dim = int(model.get("contact_head_hidden_dim", max(self.hidden_dim // 2, 16)))
        if self.contact_head_hidden_dim < 1:
            raise ValueError("contact_head_hidden_dim must be positive")
        dim = self.hidden_dim
        self.state_encoders = nn.ModuleDict({
            key: nn.GRU(self.joint_dim, dim, self.state_layers, batch_first=True,
                        dropout=self.dropout if self.state_layers > 1 else 0.0)
            for key in self.inputs
        })
        self.modality_embeddings = nn.ParameterDict({
            key: nn.Parameter(torch.empty(dim)) for key in self.inputs
        })
        for value in self.modality_embeddings.values():
            nn.init.normal_(value, std=0.02)
        self.action_encoder = nn.GRU(
            self.action_dim, dim, self.action_layers, batch_first=True,
            dropout=self.dropout if self.action_layers > 1 else 0.0,
        )
        self.history_pos_embedding = nn.Embedding(self.history_horizon, dim)
        self.action_pos_embedding = nn.Embedding(self.action_condition_horizon, dim)
        self.future_pos_embedding = nn.Embedding(self.future_horizon, dim)
        self.state_token_norm = nn.LayerNorm(dim)
        self.action_token_norm = nn.LayerNorm(dim)
        self.future_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.future_query, std=0.02)
        self.decoder_blocks = nn.ModuleList([
            DeterministicDecoderBlock(dim, self.attention_heads, self.ffn_multiplier, self.dropout)
            for _ in range(self.decoder_layers)
        ])
        self.state_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.output_dim))
        self.contact_condition_norm = nn.LayerNorm(dim)
        self.contact_fusion = nn.Linear(3 * dim, dim)
        self.contact_temporal = _ContactTemporalBlock(dim, self.attention_heads, self.dropout)
        self.contact_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, self.contact_head_hidden_dim), nn.SiLU(),
            nn.Dropout(self.dropout), nn.Linear(self.contact_head_hidden_dim, self.contact_state_count),
        )
        self._data_contract = copy.deepcopy({
            "action_contract": config.get("action_contract"),
            **{key: data.get(key) for key in ("action_key", "normalize_mode", "normalize_lowdim_keys")},
        })

    def baseline_contract(self):
        """Architecture/data contract, retaining the original checkpoint schema."""
        keys = ("inputs", "outputs", "joint_dim", "action_dim", "hidden_dim", "state_layers",
                "action_layers", "decoder_layers", "attention_heads", "ffn_multiplier", "dropout",
                "contact_head_hidden_dim", "contact_state_count", "external_history_horizon",
                "external_future_horizon", "action_condition_horizon", "temporal_stride",
                "external_state_rate_hz", "action_rate_hz", "action_start_offset", "use_action_padding_mask")
        return {"model_version": self.MODEL_VERSION, "schema": 1,
                **{key: getattr(self, key) for key in keys}, "data": copy.deepcopy(self._data_contract)}

    def checkpoint_contract(self):
        return self.baseline_contract()

    def validate_checkpoint_contract(self, actual):
        if actual != self.checkpoint_contract():
            raise ValueError("deterministic WM checkpoint contract differs from configured architecture/data")

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get("model_version") != self.MODEL_VERSION:
            raise ValueError("checkpoint model_version does not match deterministic_wm_v1")
        # New files use the shared WM envelope; keep already-trained v1 files
        # loadable, and validate both contracts if both are present.
        contracts = [checkpoint[key] for key in ("carswm_contract", "deterministic_wm_contract")
                     if key in checkpoint]
        for contract in contracts or [None]:
            self.validate_checkpoint_contract(contract)

    def prepare_batch(self, batch):
        """Idempotent state-only striding, retaining the current history anchor."""
        result = dict(batch)
        if self.temporal_stride == 1:
            return result
        history_keys = set(self.inputs) | {"contact", "history_timestamp_ns", "history_indices", "history_valid_mask"}
        for key, value in batch.items():
            if not torch.is_tensor(value) or value.ndim < 2:
                continue
            is_history = key in history_keys
            if is_history:
                expected = self.external_history_horizon
            elif key.endswith(("_future", "_future_raw")) or key in {"future_time", "future_timestamp_ns", "future_indices"}:
                expected = self.external_future_horizon
            else:
                continue
            if value.shape[1] == expected:
                offset = self.temporal_stride - 1 if is_history else 0
                result[key] = value[:, offset::self.temporal_stride]
            elif value.shape[1] != expected // self.temporal_stride:
                raise ValueError(f"{key} has invalid temporal length {value.shape[1]}")
        return result

    @staticmethod
    def _sequence(batch, key, horizon, width):
        value = batch[key]
        if (not torch.is_tensor(value) or value.ndim != 3
                or value.shape[1:] != (horizon, width) or not value.is_floating_point()):
            raise ValueError(f"{key} must be a floating tensor [B, {horizon}, {width}]")
        return value

    def encode_conditions(self, batch):
        batch = self.prepare_batch(batch)
        reference = self._sequence(batch, self.inputs[0], self.history_horizon, self.joint_dim)
        history = []
        positions = torch.arange(self.history_horizon - 1, -1, -1, device=reference.device)
        for key in self.inputs:
            value = self._sequence(batch, key, self.history_horizon, self.joint_dim)
            if value.shape[0] != reference.shape[0] or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{key} must share the state batch size, device and dtype")
            sequence = _run_gru(self.state_encoders[key], value)
            history.append(sequence + self.modality_embeddings[key]
                           + self.history_pos_embedding(positions)[None].to(sequence.dtype))
        state_tokens = self.state_token_norm(torch.cat(history, dim=1))
        action = self._sequence(batch, "action", self.action_condition_horizon, self.action_dim)
        if action.shape[0] != reference.shape[0] or action.device != reference.device or action.dtype != reference.dtype:
            raise ValueError("action must share the state batch size, device and dtype")
        valid = torch.ones(action.shape[:2], dtype=torch.bool, device=action.device)
        if self.use_action_padding_mask and batch.get("action_mask") is not None:
            mask = batch["action_mask"]
            if not torch.is_tensor(mask) or mask.shape != valid.shape:
                raise ValueError("action_mask must have shape [B, H_action]")
            if not torch.isfinite(mask).all():
                raise ValueError("action_mask must be finite")
            valid = mask.to(action.device) > 0
            if not valid.any(dim=1).all():
                raise ValueError("each sample must have at least one valid action")
        action_features = _run_gru(self.action_encoder, action.masked_fill(~valid[..., None], 0))
        action_positions = torch.arange(self.action_condition_horizon, device=action.device)
        action_tokens = self.action_token_norm(
            action_features + self.action_pos_embedding(action_positions)[None].to(action_features.dtype)
        )
        summary = (state_tokens.sum(1) + (action_tokens * valid[..., None]).sum(1)) / (
            state_tokens.shape[1] + valid.sum(1, keepdim=True)
        )
        return {"state_tokens": state_tokens, "action_tokens": action_tokens,
                "action_padding_mask": ~valid if self.use_action_padding_mask else None,
                "condition_summary": summary, "_prepared_batch": batch}

    def forward(self, batch):
        encoded = self.encode_conditions(batch)
        return self.predict_from_conditions(encoded)

    def predict_from_conditions(self, encoded):
        """Decode an encoded batch once on the internal temporal grid."""
        history = encoded["state_tokens"]
        positions = torch.arange(self.future_horizon, device=history.device)
        future_pe = self.future_pos_embedding(positions)[None].to(history.dtype)
        features = (self.future_query.to(history.dtype)[None, None] + future_pe).expand(history.shape[0], -1, -1)
        for block in self.decoder_blocks:
            features = block(features, history, encoded["action_tokens"], encoded["action_padding_mask"])
        states = self.state_head(features)
        # Same nominal-rate ZOH action alignment as the existing contact head.
        indices = torch.floor((positions + 1).float() * self.action_rate_hz / self.state_rate_hz).long()
        indices = (indices - self.action_start_offset).clamp(0, self.action_condition_horizon - 1)
        action_tokens = encoded["action_tokens"]
        if encoded["action_padding_mask"] is not None:
            action_tokens = action_tokens.masked_fill(encoded["action_padding_mask"][..., None], 0)
        aligned_action = action_tokens.index_select(1, indices)
        summary = self.contact_condition_norm(encoded["condition_summary"])[:, None].expand(-1, self.future_horizon, -1)
        contact_features = self.contact_fusion(torch.cat((features, aligned_action, summary), dim=-1)) + future_pe
        logits = self.contact_head(self.contact_temporal(contact_features))
        result = {**encoded, "decoder_features": features, "contact_logits": logits}
        result.update({f"{key}_pred": chunk for key, chunk in zip(self.outputs, states.split(self.joint_dim, dim=-1))})
        if self.emit_contact_probabilities or not self.training:
            result["contact_probability"] = logits.softmax(dim=-1)
            result["contact_state_pred"] = logits.argmax(dim=-1, keepdim=True)
        return result

    @torch.no_grad()
    def predict(self, batch, *, steps=None, solver=None, source_noise=None):
        """WM-compatible prediction; flow options have no effect on this model."""
        result = self(batch)
        if self.temporal_stride > 1:
            for key in (*[f"{key}_pred" for key in self.outputs], "contact_logits", "contact_probability", "contact_state_pred"):
                if key in result:
                    result[key] = result[key].repeat_interleave(self.temporal_stride, dim=1)
        return result

    @torch.no_grad()
    def sample(self, batch, *, num_samples=1, steps=None, solver=None, source_noise=None):
        """Return identical futures in WM's [B,K,T,D] format, decoding once."""
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        result = self.predict(batch, steps=steps, solver=solver, source_noise=source_noise)
        keys = [f"{key}_pred" for key in self.outputs]
        keys.extend(("contact_logits", "contact_probability", "contact_state_pred"))
        return {key: result[key][:, None].repeat(1, num_samples, 1, 1)
                for key in keys if key in result}
