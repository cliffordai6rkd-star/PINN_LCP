"""Latent Flow with a conditional residual codec and stable transfer targets."""
from __future__ import annotations

import copy

import torch
from torch import nn
import torch.nn.functional as F

from data_process.relative_time_grid import RelativeTimeGrid, grid_sinusoidal
from model.pinn_model.contact_world_model import FlowDecoderBlock, FlowTimeEmbedding
from model.pinn_model.latent_pretrained import MOTION_KEYS, convert_scale, load_motion_checkpoint

MODEL_VERSION = "latent_carswm_lstm_v1"
CONDITIONAL_MODEL_VERSION = "latent_carswm_lstm_v2"
GRID_KEYS = ("history_grid_positions", "action_grid_positions", "future_grid_positions")


def two_linear(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class FutureEncoder(nn.Module):
    def __init__(self, joint_dim, hidden_dim, latent_dim, classes, local_dim=0):
        super().__init__()
        self.network = two_linear(2*joint_dim+classes+local_dim, hidden_dim, latent_dim)

    def forward(self, value):
        return self.network(value)


class FlowVelocityAdapter(nn.Module):
    """Low-rank residual velocity update; the initial field is unchanged."""
    def __init__(self, hidden_dim, latent_dim, rank):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, latent_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return self.up(self.down(features))


class LatentContactWorldModel(nn.Module):
    MODEL_VERSION = MODEL_VERSION
    CONDITION_KEYS = (*MOTION_KEYS, "tau", "action", "action_mask", *GRID_KEYS)
    TARGET_KEYS = ("q_future", "tau_future", "contact_future")
    predicted_state_streams = ("q", "tau")
    PREDICTED_STATE_STREAMS = predicted_state_streams
    CODEC_MODULES = ("future_encoder", "q_head", "tau_head", "contact_head")
    CONDITION_MODULES = ("motion_encoder", "tau_encoder", "action_encoder", "modality_embeddings",
                         "history_norm", "action_norm")
    FLOW_MODULES = ("latent_projection", "flow_time_embedding", "flow_blocks", "velocity_head")

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        data, m, train = (config.get(k) or {} for k in ("dataloader", "model", "train"))
        self.MODEL_VERSION = m.get("family", MODEL_VERSION)
        if self.MODEL_VERSION not in {MODEL_VERSION, CONDITIONAL_MODEL_VERSION}:
            raise ValueError("model.family must be latent_carswm_lstm_v1 or latent_carswm_lstm_v2")
        self.conditional_codec = self.MODEL_VERSION == CONDITIONAL_MODEL_VERSION
        if self.conditional_codec:
            self.CODEC_MODULES = (*self.CODEC_MODULES, "local_state_encoder", "target_state_encoder")
        if m.get("inputs", ["q", "dq", "delta_q", "tau"]) != ["q", "dq", "delta_q", "tau"]:
            raise ValueError("latent LSTM model inputs must be [q,dq,delta_q,tau]")
        if m.get("outputs", ["q", "tau"]) != ["q", "tau"]:
            raise ValueError("latent v1 outputs must be [q,tau] plus contact")
        if any(m.get(key, 2) != 2 for key in ("lstm_layers", "future_encoder_layers", "decoder_layers")):
            raise ValueError("condition LSTMs and codec MLPs must each have exactly two layers")
        self.temporal_position_encoding = m.get("temporal_position_encoding", "relative_grid_sinusoidal")
        if self.temporal_position_encoding not in {"relative_grid_sinusoidal", "contact_wm_learned_index"}:
            raise ValueError("unsupported temporal_position_encoding")
        self.learned_positions = self.temporal_position_encoding == "contact_wm_learned_index"
        if self.learned_positions:
            if not self.conditional_codec:
                raise ValueError("contact_wm_learned_index requires latent v2")
            self.CONDITION_KEYS = tuple(key for key in self.CONDITION_KEYS if key not in GRID_KEYS)
            self.CONDITION_MODULES = (*self.CONDITION_MODULES, "history_pos_embedding", "action_pos_embedding")
            self.FLOW_MODULES = (*self.FLOW_MODULES, "future_pos_embedding")
        if m.get("flow_source_mode", "gaussian") != "gaussian":
            raise ValueError("latent flow source must be Gaussian")
        for key, default in (("joint_dim", 7), ("action_dim", 7), ("hidden_dim", 128),
                             ("latent_dim", 128), ("decoder_hidden_dim", 128), ("contact_state_count", 3)):
            setattr(self, key, int(m.get(key, default)))
            if getattr(self, key) <= 0:
                raise ValueError(f"model.{key} must be positive")
        if self.contact_state_count != 3:
            raise ValueError("latent v1 requires three contact phases")
        self.external_history_horizon = int(data.get("state_history_horizon", 50))
        self.external_future_horizon = int(data.get("prediction_horizon", 40))
        self.action_condition_horizon = int(data.get("action_condition_horizon", 10))
        stride = train.get("downsample", False)
        self.temporal_stride = (2 if stride else 1) if isinstance(stride, bool) else int(stride)
        if self.temporal_stride < 1 or any(h <= 0 or h % self.temporal_stride for h in
                                         (self.external_history_horizon, self.external_future_horizon)):
            raise ValueError("state horizons must be positive multiples of state stride")
        if self.action_condition_horizon < 1:
            raise ValueError("action horizon must be positive")
        self.history_horizon = self.external_history_horizon // self.temporal_stride
        self.future_horizon = self.external_future_horizon // self.temporal_stride
        self.grid = RelativeTimeGrid.from_config(config)
        self.action_start_offset = int(data.get("action_start_offset", 1))
        self.flow_inference_steps = int(m.get("flow_inference_steps", 16))
        self.flow_solver = m.get("flow_solver", "heun")
        self.dropout = float(m.get("dropout", 0.01))
        heads = int(m.get("flow_attention_heads", 4))
        if self.hidden_dim % heads or not 0 <= self.dropout < 1:
            raise ValueError("invalid attention head count or dropout")
        if self.flow_inference_steps < 1 or self.flow_solver not in {"euler", "heun"}:
            raise ValueError("invalid ODE steps or solver")
        h = self.hidden_dim
        if self.learned_positions:
            self.history_pos_embedding = nn.Embedding(self.history_horizon, h)
            self.action_pos_embedding = nn.Embedding(self.action_condition_horizon, h)
            self.future_pos_embedding = nn.Embedding(self.future_horizon, h)
        self.motion_encoder = nn.LSTM(3*self.joint_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.tau_encoder = nn.LSTM(self.joint_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.action_encoder = nn.LSTM(self.action_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.modality_embeddings = nn.ParameterDict({key:nn.Parameter(torch.randn(h)*0.02)
                                                      for key in ("motion", "tau", "action")})
        self.history_norm, self.action_norm = nn.LayerNorm(h), nn.LayerNorm(h)
        self.local_state_dim = int(m.get("local_state_dim", 64)) if self.conditional_codec else 0
        self.flow_adapter_rank = int(m.get("flow_adapter_rank", 8)) if self.conditional_codec else 0
        if self.conditional_codec and (self.local_state_dim < 1 or self.flow_adapter_rank < 1):
            raise ValueError("local_state_dim and flow_adapter_rank must be positive")
        if self.conditional_codec:
            self.local_state_encoder = two_linear(4*self.joint_dim, h, self.local_state_dim)
            # Used only for frozen target encoding after codec finalization.
            self.target_state_encoder = copy.deepcopy(self.local_state_encoder)
        self.future_encoder = FutureEncoder(self.joint_dim, h, self.latent_dim, self.contact_state_count,
                                            self.local_state_dim)
        for key, width in (("q", self.joint_dim), ("tau", self.joint_dim), ("contact", self.contact_state_count)):
            setattr(self, key + "_head", two_linear(self.latent_dim+self.local_state_dim, self.decoder_hidden_dim, width))
        self.latent_projection = nn.Linear(self.latent_dim, h)
        self.flow_time_embedding = FlowTimeEmbedding(h)
        self.flow_blocks = nn.ModuleList([FlowDecoderBlock(h, heads, int(m.get("flow_ffn_multiplier", 4)), self.dropout)
                                         for _ in range(int(m.get("flow_layers", 4)))])
        if not self.flow_blocks:
            raise ValueError("flow_layers must be positive")
        self.velocity_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, self.latent_dim))
        if self.conditional_codec:
            self.flow_adapter = FlowVelocityAdapter(h, self.latent_dim, self.flow_adapter_rank)
        self.frozen_motion = m.get("pretrained_taufree_path") is not None
        self.free_tau_head = None if self.frozen_motion else two_linear(h, h, self.joint_dim)
        self.pretrained_normalizer, self.pretrained_contract, self.wm_normalizer = None, None, None
        self.codec_snapshot = None
        self.codec_normalizer = None
        self.transfer_provenance = None
        self.sft_flow_mode = (config.get("sft") or {}).get("flow_mode", "frozen")
        if self.sft_flow_mode not in {"frozen", "adapter"}:
            raise ValueError("sft.flow_mode must be frozen or adapter")
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))
        self.register_buffer("codec_ready", torch.tensor(False))
        self.stage = "codec"
        self._adaptation_frozen = set()
        self.set_stage("codec")

    def get_extra_state(self):
        return copy.deepcopy({key:getattr(self, key) for key in
            ("pretrained_normalizer", "pretrained_contract", "wm_normalizer", "codec_snapshot", "stage", "_adaptation_frozen",
             "codec_normalizer", "transfer_provenance")})

    def set_extra_state(self, state):
        for key, value in copy.deepcopy(state).items():
            setattr(self, key, value)
        self.set_stage(self.stage)

    def checkpoint_contract(self):
        m, data = self.config.get("model") or {}, self.config.get("dataloader") or {}
        contract = {"model_version": self.MODEL_VERSION, "schema": 1, "codec": self.codec_contract(),
                "motion_input_order": list(MOTION_KEYS), "lstm_layers": 2, "history_mode": "stateless_sliding_window",
                "hidden_dim": self.hidden_dim, "action_dim": self.action_dim, "frozen_motion": self.frozen_motion,
                "history_horizon": self.external_history_horizon, "action_horizon": self.action_condition_horizon,
                "stride": self.temporal_stride, "action_start_offset": self.action_start_offset,
                "grid": self.grid.contract(), "flow_layers": len(self.flow_blocks),
                "attention_heads": m.get("flow_attention_heads", 4), "ffn_multiplier": m.get("flow_ffn_multiplier", 4),
                "dropout": self.dropout, "solver": self.flow_solver, "inference_steps": self.flow_inference_steps,
                "action_contract": self.config.get("action_contract"),
                "preprocessing": {k:data.get(k) for k in ("filters", "normalize_mode", "normalize_lowdim_keys",
                                                        "action_key", "high_keys", "action_condition_mode", "inference_delay_s")},
                "transfer": "pretrained_normalizer_and_contract_embedded_in_model_extra_state"}
        if self.conditional_codec:
            contract.update(schema=2, flow_adapter_rank=self.flow_adapter_rank,
                            sft_flow_mode=self.sft_flow_mode,
                            sft_train_motion_encoder=(self.config.get("sft") or {}).get("train_motion_encoder", True))
        if self.learned_positions:
            contract["position_encoding"] = {
                "scheme": "contact_wm_learned_index", "history": "shared_learned_recency_index_newest_zero",
                "action": "learned_sequence_index", "future": "learned_sequence_index",
                "history_length": self.history_horizon, "action_length": self.action_condition_horizon,
                "future_length": self.future_horizon}
        return contract

    def codec_contract(self):
        data = self.config.get("dataloader") or {}
        contract = {"schema": "latent_codec_v1", "joint_dim": self.joint_dim, "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim, "decoder_hidden_dim": self.decoder_hidden_dim, "linear_layers": 2,
                "classes": self.contact_state_count, "input": ["normalized_q", "normalized_tau", "contact_one_hot"],
                "activation": "silu", "future_horizon": self.external_future_horizon,
                "normalization": {k:data.get(k) for k in ("normalize_mode", "normalize_lowdim_keys", "filters")},
                "snapshot_policy": "final_raw", "stats_population": "training_split_future_windows_only",
                "std_floor": float((self.config.get("codec") or {}).get("latent_std_floor", 1e-4))}
        if self.conditional_codec:
            contract.update(schema="latent_codec_v2", local_state_dim=self.local_state_dim,
                            local_input=["q", "dq", "delta_q", "tau"],
                            input=["normalized_q_future_minus_q_current", "normalized_tau", "contact_one_hot", "local_state"],
                            q_output="normalized_q_current_plus_predicted_change",
                            target_state_policy="frozen_codec_snapshot", target_scale_policy="codec_normalizer")
        return contract

    def validate_checkpoint_contract(self, contract):
        if contract != self.checkpoint_contract():
            raise ValueError("latent CARS-WM architecture/data/grid checkpoint contract mismatch")

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get("model_version") != self.MODEL_VERSION:
            raise ValueError(f"checkpoint model_version is not {self.MODEL_VERSION}")
        self.validate_checkpoint_contract(checkpoint.get("carswm_contract"))

    def initialize_pretrained_motion(self, **kwargs):
        path = (self.config.get("model") or {}).get("pretrained_taufree_path")
        if path is not None:
            load_motion_checkpoint(self, path, self.config, **kwargs)
            self.set_stage(self.stage)

    def initialize_sft(self, path, *, use_ema=True):
        """Import a finalized v2 model, preserving its target codec coordinates.

        Episode splits, normalizers, action frames and window lengths can change
        for transfer. Architecture, action representation and time units cannot.
        This is initialization, separate from an exact training resume.
        """
        if not self.conditional_codec:
            raise ValueError("SFT requires a v2 conditional codec; retrain v1 pretraining with v2")
        source, payload = load_latent_checkpoint(path, use_ema=use_ema)
        if not source.conditional_codec or not source.codec_ready or source.stage not in {"flow", "sft"}:
            raise ValueError("SFT requires a finalized v2 Flow checkpoint")
        if source.temporal_position_encoding != self.temporal_position_encoding:
            raise ValueError("SFT temporal_position_encoding mismatch")
        if self.learned_positions:
            for key in ("history_horizon", "future_horizon", "action_condition_horizon"):
                if getattr(source, key) != getattr(self, key):
                    raise ValueError(f"SFT learned-position table mismatch: {key}")
        for key in ("joint_dim", "action_dim", "hidden_dim", "latent_dim", "decoder_hidden_dim",
                    "contact_state_count", "local_state_dim", "flow_adapter_rank", "frozen_motion", "temporal_stride"):
            if getattr(source, key) != getattr(self, key):
                raise ValueError(f"SFT architecture mismatch: {key}")
        for key in ("state_rate_hz", "action_rate_hz"):
            if getattr(source.grid, key) != getattr(self.grid, key):
                raise ValueError(f"SFT time-grid mismatch: {key}")
        source_contract, target_contract = source.checkpoint_contract(), self.checkpoint_contract()
        for key in ("flow_layers", "attention_heads", "ffn_multiplier"):
            if source_contract[key] != target_contract[key]:
                raise ValueError(f"SFT architecture mismatch: {key}")
        for key in ("type", "representation", "quaternion_order", "quaternion_sign", "absolute_or_relative"):
            if (source.config.get("action_contract") or {}).get(key) != (self.config.get("action_contract") or {}).get(key):
                raise ValueError(f"SFT action representation mismatch: {key}")
        target_normalizer = copy.deepcopy(self.wm_normalizer)
        for envelope in (target_normalizer, source.codec_normalizer):
            if envelope is None or envelope.get("normalize_mode") not in {None, "gaussian", "limit"}:
                raise ValueError("SFT requires embedded invertible codec/target normalizers (gaussian, limit, or none)")
        self.load_state_dict(source.state_dict(), strict=True)
        self.wm_normalizer = target_normalizer
        self._adaptation_frozen = set()
        self.transfer_provenance = {"checkpoint": str(path), "snapshot": "ema" if use_ema and payload.get("model_raw") is not None else "raw",
            "source_contract": source_contract, "source_transfer": source.transfer_provenance,
            "target_action_contract": copy.deepcopy(self.config.get("action_contract")),
            "flow_mode": self.sft_flow_mode}
        self.set_stage("sft")
        return payload

    def freeze_modules(self, names):
        for name in names:
            if name not in self.module_names():
                raise ValueError(f"unknown module {name}")
            self._adaptation_frozen.add(name)
        self.set_stage(self.stage)

    def module_names(self):
        return (*self.CODEC_MODULES, *self.CONDITION_MODULES, *self.FLOW_MODULES, "free_tau_head",
                *(("flow_adapter",) if self.conditional_codec else ()))

    def set_stage(self, stage):
        if stage not in {"codec", "flow", "sft"} or (stage == "sft" and not self.conditional_codec):
            raise ValueError("model stage must be codec, flow, or sft (v2 only)")
        self.stage = stage
        self.requires_grad_(False)
        if stage == "codec":
            names = tuple(name for name in self.CODEC_MODULES if name != "target_state_encoder")
        elif stage == "flow":
            names = (*self.CONDITION_MODULES, *self.FLOW_MODULES, "free_tau_head")
        else:
            names = (*self.CONDITION_MODULES, "local_state_encoder", "q_head", "tau_head", "contact_head", "free_tau_head",
                     *(("flow_adapter",) if self.sft_flow_mode == "adapter" else ()))
        for name in names:
            module = getattr(self, name)
            if module is not None and name not in self._adaptation_frozen:
                module.requires_grad_(True)
        if ((self.frozen_motion and stage != "sft")
                or (stage == "sft" and not (self.config.get("sft") or {}).get("train_motion_encoder", True))):
            self.motion_encoder.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name in self.module_names():
            module = getattr(self, name)
            if module is not None and not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

    def parameter_groups(self):
        return [{"name":name, "params":[p for p in getattr(self, name).parameters() if p.requires_grad]}
                for name in self.module_names()
                if getattr(self, name) is not None and any(p.requires_grad for p in getattr(self, name).parameters())]

    def prepare_batch(self, batch):
        result = dict(batch)
        if "free_dynamics_mask" not in result and "contact" in batch and "history_valid_mask" in batch:
            labels, validity = batch["contact"], batch["history_valid_mask"]
            if labels.shape != (*batch["q"].shape[:2], 1) or validity.shape != batch["q"].shape[:2]:
                raise ValueError("free supervision labels/mask must align with full history")
            # Before state stride, so omitted contact rows cannot become free.
            result["free_dynamics_mask"] = (torch.isfinite(validity).all(1) & validity.bool().all(1) & torch.isfinite(labels).all((1,2))
                                             & (labels == 0).all((1,2)))
        if self.temporal_stride == 1:
            return result
        history_keys = {*MOTION_KEYS, "tau", "contact", "history_valid_mask", "history_grid_positions"}
        future_keys = {*self.TARGET_KEYS, "future_grid_positions"}
        for key in history_keys | future_keys:
            value = result.get(key)
            if not torch.is_tensor(value):
                continue
            external = self.external_history_horizon if key in history_keys else self.external_future_horizon
            if value.shape[1] == external:
                start = self.temporal_stride-1 if key in history_keys else 0
                result[key] = value[:, start::self.temporal_stride]
            elif value.shape[1] != external // self.temporal_stride:
                raise ValueError(f"{key} has invalid temporal length")
        return result

    def local_state(self, batch, *, target=False):
        batch = self.prepare_batch(batch)
        values = []
        for key in (*MOTION_KEYS, "tau"):
            value = batch[key]
            if value.ndim != 3 or value.shape[1:] != (self.history_horizon, self.joint_dim) or not value.is_floating_point():
                raise ValueError(f"{key} must be floating [B,{self.history_horizon},{self.joint_dim}]")
            value = value[:, -1]
            if target and self.codec_normalizer is not None:
                value = self.to_codec_scale(key, value)
            values.append(value)
        return torch.cat(values, -1)

    def to_codec_scale(self, key, value):
        raw = convert_scale(key, value, self.wm_normalizer, inverse=True)
        return convert_scale(key, raw, self.codec_normalizer)

    @torch.no_grad()
    def snapshot_target_encoder(self):
        """Freeze the condition path that defines the pretrained latent coordinates."""
        if self.conditional_codec:
            self.target_state_encoder.load_state_dict(self.local_state_encoder.state_dict(), strict=True)
            self.target_state_encoder.requires_grad_(False).eval()
            self.codec_normalizer = copy.deepcopy(self.wm_normalizer)

    def future_input(self, batch, *, target=False):
        batch = self.prepare_batch(batch)
        q, tau, labels = (batch[key] for key in self.TARGET_KEYS)
        if q.shape[1:] != (self.future_horizon, self.joint_dim) or tau.shape != q.shape:
            raise ValueError("future q/tau shape mismatch")
        if labels.shape != (*q.shape[:2], 1) or not torch.isfinite(labels).all():
            raise ValueError("future contact shape or finite-label mismatch")
        if ((labels != labels.round()) | (labels < 0) | (labels >= self.contact_state_count)).any():
            raise ValueError("invalid contact phase")
        if self.conditional_codec:
            current = batch["q"][:, -1:]
            if target and self.codec_normalizer is not None:
                q, tau = self.to_codec_scale("q", q), self.to_codec_scale("tau", tau)
                current = self.to_codec_scale("q", current)
            q = q-current
        return torch.cat((q, tau, F.one_hot(labels[...,0].long(), self.contact_state_count).to(q.dtype)), -1)

    def encode_future(self, batch, *, target=False):
        value = self.future_input(batch, target=target)
        if self.conditional_codec:
            module = self.target_state_encoder if target else self.local_state_encoder
            local = module(self.local_state(batch, target=target))
            value = torch.cat((value, local[:, None].expand(-1, self.future_horizon, -1)), -1)
        return self.future_encoder(value)

    def decode(self, raw_latent, batch=None):
        value = raw_latent
        if self.conditional_codec:
            if batch is None:
                raise ValueError("conditional decoding requires observed history")
            batch = self.prepare_batch(batch)
            if raw_latent.ndim not in {3, 4} or raw_latent.shape[0] != batch["q"].shape[0] or raw_latent.shape[-1] != self.latent_dim:
                raise ValueError("conditional latent must be [B,H,D] or [B,S,H,D] aligned with history")
            local = self.local_state_encoder(self.local_state(batch))
            broadcast = (local.shape[0],) + (1,)*(raw_latent.ndim-2) + (self.local_state_dim,)
            local = local.reshape(broadcast).expand(*raw_latent.shape[:-1], self.local_state_dim)
            value = torch.cat((raw_latent, local), -1)
        logits = self.contact_head(value)
        q = self.q_head(value)
        if self.conditional_codec:
            # q_current and the head's change share the WM normalized scale.
            anchor = batch["q"][:, -1].reshape((q.shape[0],) + (1,)*(q.ndim-2) + (self.joint_dim,))
            q = anchor+q
        return {"q_pred": q, "tau_pred": self.tau_head(value),
                "contact_logits": logits, "contact_probability": logits.softmax(-1),
                "contact_state_pred": logits.argmax(-1, keepdim=True)}

    def codec_forward(self, batch):
        # SFT reconstructs frozen true latents, never a moving target encoder.
        z = self.target_latent(batch)*self.latent_std+self.latent_mean if self.stage == "sft" else self.encode_future(batch)
        return {**self.decode(z, batch), "raw_latent": z, "_prepared_batch": self.prepare_batch(batch)}

    @torch.no_grad()
    def target_latent(self, batch):
        if not self.codec_ready:
            raise RuntimeError("freeze a codec and fit train-only latent statistics before Flow")
        return ((self.encode_future(batch, target=True).float() - self.latent_mean) / self.latent_std).detach()

    @staticmethod
    def _lstm(module, value):
        if value.device.type != "npu":
            return module(value)[0]  # No carried hidden/cell state.
        with torch.autocast(device_type="npu", enabled=False):
            result = module(value.float().contiguous())[0]
        return result.to(value.dtype)

    def motion_features(self, batch):
        values = []
        for key in MOTION_KEYS:
            value = batch[key]
            if self.frozen_motion:
                if self.pretrained_normalizer is None:
                    raise RuntimeError("pretrained motion weights/input statistics have not been loaded")
                raw = convert_scale(key, value, self.wm_normalizer, inverse=True)
                value = convert_scale(key, raw, self.pretrained_normalizer)
            values.append(value)
        return self._lstm(self.motion_encoder, torch.cat(values, -1))

    def encode_conditions(self, batch, *, auxiliary=False):
        batch = self.prepare_batch(batch)
        for key, length, width in ((*[(k, self.history_horizon, self.joint_dim) for k in (*MOTION_KEYS, "tau")],
                                   ("action", self.action_condition_horizon, self.action_dim))):
            value = batch[key]
            if value.ndim != 3 or value.shape[1:] != (length, width) or not value.is_floating_point():
                raise ValueError(f"{key} must be floating [B,{length},{width}]")
        motion = self.motion_features(batch)
        tau = self._lstm(self.tau_encoder, batch["tau"])
        mask = batch.get("action_mask", torch.ones(batch["action"].shape[:2], device=motion.device))
        if not torch.isfinite(mask).all():
            raise ValueError("action mask must be finite")
        valid = mask > 0
        if valid.shape != batch["action"].shape[:2] or not valid.any(1).all():
            raise ValueError("action mask must provide at least one valid token per window")
        action = self._lstm(self.action_encoder, batch["action"].masked_fill(~valid[...,None], 0))
        def pe(key, value):
            positions = batch[key]
            if positions.shape != value.shape[:2] or positions.device != value.device:
                raise ValueError(f"{key} must match memory shape/device")
            return grid_sinusoidal(positions, self.hidden_dim, dtype=value.dtype)
        if self.learned_positions:
            # Identical indexing to ContactWorldModel: chronological tokens,
            # shared history recency (newest=0), independent action/future indices.
            history_positions = torch.arange(self.history_horizon-1, -1, -1, device=motion.device)
            history_pe = self.history_pos_embedding(history_positions)[None].to(motion.dtype)
            action_pe = self.action_pos_embedding(torch.arange(self.action_condition_horizon, device=motion.device))[None].to(action.dtype)
            future_pe = self.future_pos_embedding(torch.arange(self.future_horizon, device=motion.device))[None].to(motion.dtype)
            future_pe = future_pe.expand(motion.shape[0], -1, -1)
        else:
            history_pe = pe("history_grid_positions", motion)
            action_pe = pe("action_grid_positions", action)
            future_positions = batch["future_grid_positions"]
            if future_positions.shape != (motion.shape[0], self.future_horizon):
                raise ValueError("future_grid_positions must match predicted horizon")
            future_pe = grid_sinusoidal(future_positions, self.hidden_dim, dtype=motion.dtype)
        history = self.history_norm(torch.cat((motion + history_pe + self.modality_embeddings["motion"],
                                               tau + history_pe + self.modality_embeddings["tau"]), 1))
        action = self.action_norm(action + action_pe + self.modality_embeddings["action"])
        encoded = {"history":history, "action":action, "action_padding_mask":~valid,
                   "future_pe":future_pe,
                   "_prepared_batch":batch}
        if auxiliary and self.free_tau_head is not None:
            encoded["free_tau_pred"] = self.free_tau_head(motion[:, -1])
        return encoded

    def velocity(self, z, time, encoded):
        if z.shape[1:] != (self.future_horizon, self.latent_dim):
            raise ValueError("latent shape mismatch")
        if not torch.is_tensor(time):
            time = z.new_full((z.shape[0],), float(time))
        elif time.ndim == 0:
            time = time.expand(z.shape[0])
        features = self.latent_projection(z) + encoded["future_pe"] + self.flow_time_embedding(time)[:,None]
        for block in self.flow_blocks:
            features = block(features, encoded["history"], encoded["action"], encoded["action_padding_mask"])
        velocity = self.velocity_head(features)
        if self.conditional_codec:
            velocity = velocity+self.flow_adapter(features)
        return velocity

    def forward(self, batch, *, flow_time=None, source_noise=None):
        if self.stage == "codec":
            return self.codec_forward(batch)
        encoded = self.encode_conditions(batch, auxiliary=True)
        target = self.target_latent(encoded["_prepared_batch"])
        noise = torch.randn_like(target) if source_noise is None else source_noise
        if noise.shape != target.shape:
            raise ValueError("source_noise must match latent future")
        time = torch.rand(target.shape[0], device=target.device) if flow_time is None else torch.as_tensor(flow_time, device=target.device, dtype=target.dtype)
        if time.ndim == 0:
            time = time.expand(target.shape[0])
        time = time.reshape(target.shape[0])
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("flow_time must lie in [0,1]")
        state = (1-time[:,None,None])*noise + time[:,None,None]*target
        out = {**encoded, "flow_velocity_pred": self.velocity(state, time, encoded),
                "flow_velocity_target":target-noise, "flow_target_latent":target,
                "flow_source":noise, "flow_state":state, "flow_time":time}
        if self.stage == "sft":
            out["reconstruction"] = self.decode(target*self.latent_std+self.latent_mean, encoded["_prepared_batch"])
        return out

    @torch.no_grad()
    def sample(self, batch, *, num_samples=1, steps=None, solver=None, source_noise=None):
        if not self.codec_ready:
            raise RuntimeError("sampling requires a finalized codec with latent statistics")
        steps = self.flow_inference_steps if steps is None else int(steps)
        solver = self.flow_solver if solver is None else solver
        if steps < 1 or num_samples < 1 or solver not in {"euler", "heun"}:
            raise ValueError("invalid sample count, ODE steps or solver")
        # Encode one immutable snapshot per call; reuse it throughout [0,1].
        encoded = self.encode_conditions(batch)
        b = encoded["history"].shape[0]
        for key in ("history", "action", "action_padding_mask", "future_pe"):
            encoded[key] = encoded[key].repeat_interleave(num_samples, 0)
        shape = (b, num_samples, self.future_horizon, self.latent_dim)
        z = torch.randn(shape, device=encoded["history"].device) if source_noise is None else source_noise
        if z.shape != shape or z.device != encoded["history"].device:
            raise ValueError(f"source_noise must have shape {shape} on the condition device")
        z = z.clone().reshape(b*num_samples, self.future_horizon, self.latent_dim)
        dt = 1.0 / steps
        for i in range(steps):
            first = self.velocity(z, i*dt, encoded)
            if solver == "euler":
                z = z + dt*first
            else:
                second = self.velocity(z + dt*first, (i+1)*dt, encoded)
                z = z + dt*0.5*(first+second)
        raw = z.float()*self.latent_std + self.latent_mean
        out = self.decode(raw.reshape(shape), encoded["_prepared_batch"])
        out.update(latent=z.reshape(shape), raw_latent=raw.reshape(shape), nfe=steps*(2 if solver == "heun" else 1),
                   solver=solver, ode_steps=steps)
        return out

    @torch.no_grad()
    def predict(self, batch, **kwargs):
        out = self.sample(batch, num_samples=1, **kwargs)
        return {key:value[:,0] if torch.is_tensor(value) else value for key,value in out.items()}


def load_latent_checkpoint(path, *, device="cpu", use_ema=True):
    """Self-contained loading: neither NEXT nor the original codec path is opened."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = LatentContactWorldModel(payload["config"])
    model.validate_checkpoint(payload)
    state = payload["model"] if use_ema or payload.get("model_raw") is None else payload["model_raw"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload
