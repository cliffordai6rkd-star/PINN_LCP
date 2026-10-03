"""Three stateless LSTMs condition a CFM over a frozen deterministic codec."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from data_process.relative_time_grid import RelativeTimeGrid, grid_sinusoidal
from model.pinn_model.contact_world_model import FlowDecoderBlock, FlowTimeEmbedding
from model.pinn_model.inverse_gaussian_source import ConditionalGaussianLatentSource
from model.pinn_model.latent_pretrained import MOTION_KEYS, convert_scale, load_motion_checkpoint

MODEL_VERSION = "latent_carswm_lstm_v1"
GRID_KEYS = ("history_grid_positions", "action_grid_positions", "future_grid_positions")


def two_linear(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class FutureEncoder(nn.Module):
    def __init__(self, joint_dim, hidden_dim, latent_dim, classes):
        super().__init__()
        self.network = two_linear(2*joint_dim+classes, hidden_dim, latent_dim)

    def forward(self, value):
        return self.network(value)


class LatentContactWorldModel(nn.Module):
    MODEL_VERSION = MODEL_VERSION
    CONDITION_KEYS = (*MOTION_KEYS, "tau", "action", "action_mask", *GRID_KEYS)
    TARGET_KEYS = ("q_future", "tau_future", "contact_future")
    predicted_state_streams = ("q", "tau")
    PREDICTED_STATE_STREAMS = predicted_state_streams
    CODEC_MODULES = ("future_encoder", "q_head", "tau_head", "contact_head")
    CONDITION_MODULES = ("motion_encoder", "tau_encoder", "action_encoder", "modality_embeddings",
                         "history_norm", "action_norm")
    FLOW_MODULES = ("latent_projection", "flow_time_embedding", "flow_blocks", "velocity_head", "source_model")

    def __init__(self, config):
        super().__init__()
        self.config = copy.deepcopy(config)
        data, m, train = (config.get(k) or {} for k in ("dataloader", "model", "train"))
        if m.get("family", MODEL_VERSION) != MODEL_VERSION:
            raise ValueError("model.family must be latent_carswm_lstm_v1")
        if m.get("inputs", ["q", "dq", "delta_q", "tau"]) != ["q", "dq", "delta_q", "tau"]:
            raise ValueError("latent LSTM model inputs must be [q,dq,delta_q,tau]")
        if m.get("outputs", ["q", "tau"]) != ["q", "tau"]:
            raise ValueError("latent v1 outputs must be [q,tau] plus contact")
        if any(m.get(key, 2) != 2 for key in ("lstm_layers", "future_encoder_layers", "decoder_layers")):
            raise ValueError("condition LSTMs and codec MLPs must each have exactly two layers")
        if m.get("temporal_position_encoding", "relative_grid_sinusoidal") != "relative_grid_sinusoidal":
            raise ValueError("only shared relative_grid_sinusoidal robot-time PE is supported")
        self.source_mode = m.get("flow_source_mode", "gaussian")
        if self.source_mode not in {"gaussian", "conditional_gaussian"}:
            raise ValueError("latent flow source must be gaussian or conditional_gaussian")
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
        self.motion_encoder = nn.LSTM(3*self.joint_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.tau_encoder = nn.LSTM(self.joint_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.action_encoder = nn.LSTM(self.action_dim, h, 2, batch_first=True, dropout=self.dropout)
        self.modality_embeddings = nn.ParameterDict({key:nn.Parameter(torch.randn(h)*0.02)
                                                      for key in ("motion", "tau", "action")})
        self.history_norm, self.action_norm = nn.LayerNorm(h), nn.LayerNorm(h)
        self.future_encoder = FutureEncoder(self.joint_dim, h, self.latent_dim, self.contact_state_count)
        for key, width in (("q", self.joint_dim), ("tau", self.joint_dim), ("contact", self.contact_state_count)):
            setattr(self, key + "_head", two_linear(self.latent_dim, self.decoder_hidden_dim, width))
        self.latent_projection = nn.Linear(self.latent_dim, h)
        self.flow_time_embedding = FlowTimeEmbedding(h)
        self.flow_blocks = nn.ModuleList([FlowDecoderBlock(h, heads, int(m.get("flow_ffn_multiplier", 4)), self.dropout)
                                         for _ in range(int(m.get("flow_layers", 4)))])
        if not self.flow_blocks:
            raise ValueError("flow_layers must be positive")
        self.velocity_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, self.latent_dim))
        self.source_model = (
            ConditionalGaussianLatentSource(
                2*h, self.future_horizon, self.latent_dim,
                hidden_dim=int(m.get("conditional_source_hidden_dim", 256)),
                min_log_std=float(m.get("conditional_source_min_log_std", -4.0)),
                max_log_std=float(m.get("conditional_source_max_log_std", 1.5)),
            ) if self.source_mode == "conditional_gaussian" else nn.Identity()
        )
        self.source_temperature = float(m.get("conditional_source_temperature", 1.0))
        if not math.isfinite(self.source_temperature) or self.source_temperature < 0:
            raise ValueError("conditional_source_temperature must be finite and nonnegative")
        self.frozen_motion = m.get("pretrained_taufree_path") is not None
        self.free_tau_head = None if self.frozen_motion else two_linear(h, h, self.joint_dim)
        self.pretrained_normalizer, self.pretrained_contract, self.wm_normalizer = None, None, None
        self.runtime_normalizer, self._runtime_pretrained_normalizer, self._runtime_wm_motion_normalizer = None, None, None
        self.codec_snapshot = None
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))
        self.register_buffer("codec_ready", torch.tensor(False))
        self.stage = "codec"
        self._adaptation_frozen = set()
        self.set_stage("codec")

    def get_extra_state(self):
        return copy.deepcopy({key:getattr(self, key) for key in
            ("pretrained_normalizer", "pretrained_contract", "wm_normalizer", "codec_snapshot", "stage", "_adaptation_frozen")})

    def set_extra_state(self, state):
        self.runtime_normalizer, self._runtime_pretrained_normalizer, self._runtime_wm_motion_normalizer = None, None, None
        for key, value in copy.deepcopy(state).items():
            setattr(self, key, value)
        self.set_stage(self.stage)

    def _apply(self, fn, recurse=True):
        # Runtime constants are ordinary snapshots, not checkpoint buffers.
        # A device/dtype move requires preparing them again on the new device.
        self.runtime_normalizer, self._runtime_pretrained_normalizer, self._runtime_wm_motion_normalizer = None, None, None
        return super()._apply(fn, recurse=recurse)

    def prepare_runtime_normalizers(self, *, physical=True):
        """Validate/copy normalization constants once after load/device setup."""
        from model.pinn_model.latent_runtime_normalizer import PreparedNormalizer
        if self.training:
            raise ValueError("runtime normalizers require eval mode")
        dims = {key: self.joint_dim for key in (*MOTION_KEYS, "tau")}
        dims["action"] = self.action_dim
        self.runtime_normalizer = (PreparedNormalizer(self.wm_normalizer, dims, self.latent_mean.device)
                                   if physical and self.wm_normalizer is not None else None)
        motion_dims = {key: self.joint_dim for key in MOTION_KEYS}
        self._runtime_wm_motion_normalizer = (
            PreparedNormalizer(self.wm_normalizer, motion_dims, self.latent_mean.device)
            if self.frozen_motion and self.wm_normalizer is not None else None)
        self._runtime_pretrained_normalizer = (
            PreparedNormalizer(self.pretrained_normalizer, motion_dims,
                               self.latent_mean.device) if self.frozen_motion and self.pretrained_normalizer is not None
            else None)
        return self

    def checkpoint_contract(self):
        m, data = self.config.get("model") or {}, self.config.get("dataloader") or {}
        contract = {"model_version": MODEL_VERSION, "schema": 1, "codec": self.codec_contract(),
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
        if self.source_mode == "conditional_gaussian":
            contract["source"] = {"mode": self.source_mode, "condition": "pooled_history_and_action_tokens",
                                  "hidden_dim": self.source_model.network[1].out_features,
                                  "min_log_std": self.source_model.min_log_std,
                                  "max_log_std": self.source_model.max_log_std,
                                  "temperature": self.source_temperature,
                                  "inverse_solver": "fixed_point_heun"}
        return contract

    def codec_contract(self):
        data = self.config.get("dataloader") or {}
        return {"schema": "latent_codec_v1", "joint_dim": self.joint_dim, "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim, "decoder_hidden_dim": self.decoder_hidden_dim, "linear_layers": 2,
                "classes": self.contact_state_count, "input": ["normalized_q", "normalized_tau", "contact_one_hot"],
                "activation": "silu", "future_horizon": self.external_future_horizon,
                "normalization": {k:data.get(k) for k in ("normalize_mode", "normalize_lowdim_keys", "filters")},
                "snapshot_policy": "final_raw", "stats_population": "training_split_future_windows_only",
                "std_floor": float((self.config.get("codec") or {}).get("latent_std_floor", 1e-4))}

    def validate_checkpoint_contract(self, contract):
        if contract != self.checkpoint_contract():
            raise ValueError("latent CARS-WM architecture/data/grid checkpoint contract mismatch")

    def validate_checkpoint(self, checkpoint):
        if checkpoint.get("model_version") != MODEL_VERSION:
            raise ValueError("checkpoint model_version is not latent_carswm_lstm_v1")
        self.validate_checkpoint_contract(checkpoint.get("carswm_contract"))

    def initialize_pretrained_motion(self, **kwargs):
        path = (self.config.get("model") or {}).get("pretrained_taufree_path")
        if path is not None:
            load_motion_checkpoint(self, path, self.config, **kwargs)
            self.set_stage(self.stage)

    def freeze_modules(self, names):
        for name in names:
            if name not in (*self.CODEC_MODULES, *self.CONDITION_MODULES, *self.FLOW_MODULES):
                raise ValueError(f"unknown module {name}")
            self._adaptation_frozen.add(name)
        self.set_stage(self.stage)

    def set_stage(self, stage):
        if stage not in {"codec", "flow"}:
            raise ValueError("model stage must be codec or flow")
        self.stage = stage
        self.requires_grad_(False)
        names = self.CODEC_MODULES if stage == "codec" else (*self.CONDITION_MODULES, *self.FLOW_MODULES, "free_tau_head")
        for name in names:
            module = getattr(self, name)
            if module is not None and name not in self._adaptation_frozen:
                module.requires_grad_(True)
        if self.frozen_motion:
            self.motion_encoder.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        if mode:
            self.runtime_normalizer, self._runtime_pretrained_normalizer, self._runtime_wm_motion_normalizer = None, None, None
        super().train(mode)
        for name in (*self.CODEC_MODULES, *self.CONDITION_MODULES, *self.FLOW_MODULES, "free_tau_head"):
            module = getattr(self, name)
            if module is not None and not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

    def parameter_groups(self):
        return [{"name":name, "params":[p for p in getattr(self, name).parameters() if p.requires_grad]}
                for name in (*self.CODEC_MODULES, *self.CONDITION_MODULES, *self.FLOW_MODULES, "free_tau_head")
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

    def future_input(self, batch):
        batch = self.prepare_batch(batch)
        q, tau, labels = (batch[key] for key in self.TARGET_KEYS)
        if q.shape[1:] != (self.future_horizon, self.joint_dim) or tau.shape != q.shape:
            raise ValueError("future q/tau shape mismatch")
        if labels.shape != (*q.shape[:2], 1) or not torch.isfinite(labels).all():
            raise ValueError("future contact shape or finite-label mismatch")
        if ((labels != labels.round()) | (labels < 0) | (labels >= self.contact_state_count)).any():
            raise ValueError("invalid contact phase")
        return torch.cat((q, tau, F.one_hot(labels[...,0].long(), self.contact_state_count).to(q.dtype)), -1)

    def decode(self, raw_latent):
        logits = self.contact_head(raw_latent)
        return {"q_pred": self.q_head(raw_latent), "tau_pred": self.tau_head(raw_latent),
                "contact_logits": logits, "contact_probability": logits.softmax(-1),
                "contact_state_pred": logits.argmax(-1, keepdim=True)}

    def codec_forward(self, batch):
        z = self.future_encoder(self.future_input(batch))
        return {**self.decode(z), "raw_latent": z, "_prepared_batch": self.prepare_batch(batch)}

    @torch.no_grad()
    def target_latent(self, batch):
        if not self.codec_ready:
            raise RuntimeError("freeze a codec and fit train-only latent statistics before Flow")
        return ((self.future_encoder(self.future_input(batch)).float() - self.latent_mean) / self.latent_std).detach()

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
                if not self.training and self._runtime_wm_motion_normalizer is not None and self._runtime_pretrained_normalizer is not None:
                    raw = self._runtime_wm_motion_normalizer.convert(key, value, inverse=True)
                    value = self._runtime_pretrained_normalizer.convert(key, raw)
                else:
                    raw = convert_scale(key, value, self.wm_normalizer, inverse=True)
                    value = convert_scale(key, raw, self.pretrained_normalizer)
            values.append(value)
        return self._lstm(self.motion_encoder, torch.cat(values, -1))

    def encode_conditions(self, batch, *, auxiliary=False, cache_condition_kv=False):
        if cache_condition_kv and self.training:
            raise ValueError("condition K/V caching requires eval mode")
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
        history_pe = pe("history_grid_positions", motion)
        history = self.history_norm(torch.cat((motion + history_pe + self.modality_embeddings["motion"],
                                               tau + history_pe + self.modality_embeddings["tau"]), 1))
        action = self.action_norm(action + pe("action_grid_positions", action) + self.modality_embeddings["action"])
        future_positions = batch["future_grid_positions"]
        if future_positions.shape != (motion.shape[0], self.future_horizon):
            raise ValueError("future_grid_positions must match predicted horizon")
        encoded = {"history":history, "action":action, "action_padding_mask":~valid,
                   "future_pe":grid_sinusoidal(future_positions, self.hidden_dim, dtype=motion.dtype),
                   "_prepared_batch":batch}
        encoded["source_condition"] = torch.cat(
            (history.mean(1), (action * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)), -1)
        if cache_condition_kv:
            encoded["condition_kv_cache"] = tuple(
                block.prepare_condition_kv(history, action) for block in self.flow_blocks)
        if auxiliary and self.free_tau_head is not None:
            encoded["free_tau_pred"] = self.free_tau_head(motion[:, -1])
        return encoded

    def velocity(self, z, time, encoded, *, time_embedding=None):
        if z.shape[1:] != (self.future_horizon, self.latent_dim):
            raise ValueError("latent shape mismatch")
        if time_embedding is None:
            if not torch.is_tensor(time):
                time = z.new_full((z.shape[0],), float(time))
            elif time.ndim == 0:
                time = time.expand(z.shape[0])
            time_embedding = self.flow_time_embedding(time)
        features = self.latent_projection(z) + encoded["future_pe"] + time_embedding[:,None]
        condition_kv = encoded.get("condition_kv_cache")
        for index, block in enumerate(self.flow_blocks):
            features = block(features, encoded["history"], encoded["action"], encoded["action_padding_mask"],
                             condition_kv=None if condition_kv is None else condition_kv[index])
        return self.velocity_head(features)

    def source_from_noise(self, noise, encoded):
        if self.source_mode == "gaussian":
            return noise
        return self.source_model.transform(encoded["source_condition"], noise,
                                           temperature=self.source_temperature)

    def integrate_latent(self, source, encoded, *, steps, solver, cache_time_embeddings=True):
        """Pure fixed-grid ODE core; suitable for torch.compile after warmup.

        Shape, input and eval-mode checks belong to the public sample boundary.
        Compilation does not encompass the sliding-window LSTMs or preprocessing.
        """
        z = source
        dt = 1.0 / steps
        time_embeddings = (self.flow_time_embedding(z.new_tensor([i*dt for i in range(steps+1)]))
                           if cache_time_embeddings else None)
        for i in range(steps):
            first = self.velocity(z, i*dt, encoded, time_embedding=(
                time_embeddings[i].expand(z.shape[0], -1) if cache_time_embeddings else None))
            if solver == "euler":
                z = z + dt*first
            else:
                second = self.velocity(z + dt*first, (i+1)*dt, encoded, time_embedding=(
                    time_embeddings[i+1].expand(z.shape[0], -1) if cache_time_embeddings else None))
                z = z + dt*0.5*(first+second)
        return z

    def forward(self, batch, *, flow_time=None, source_noise=None):
        if self.stage == "codec":
            return self.codec_forward(batch)
        encoded = self.encode_conditions(batch, auxiliary=True)
        target = self.target_latent(encoded["_prepared_batch"])
        noise = torch.randn_like(target) if source_noise is None else source_noise
        if noise.shape != target.shape:
            raise ValueError("source_noise must match latent future")
        noise = self.source_from_noise(noise, encoded)
        time = torch.rand(target.shape[0], device=target.device) if flow_time is None else torch.as_tensor(flow_time, device=target.device, dtype=target.dtype)
        if time.ndim == 0:
            time = time.expand(target.shape[0])
        time = time.reshape(target.shape[0])
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("flow_time must lie in [0,1]")
        state = (1-time[:,None,None])*noise + time[:,None,None]*target
        return {**encoded, "flow_velocity_pred": self.velocity(state, time, encoded),
                "flow_velocity_target":target-noise, "flow_target_latent":target,
                "flow_source":noise, "flow_state":state, "flow_time":time}

    @torch.no_grad()
    def sample(self, batch, *, num_samples=1, steps=None, solver=None, source_noise=None,
               cache_condition_kv=None, cache_time_embeddings=None, integration_fn=None):
        if not self.codec_ready:
            raise RuntimeError("sampling requires a finalized codec with latent statistics")
        steps = self.flow_inference_steps if steps is None else int(steps)
        solver = self.flow_solver if solver is None else solver
        if steps < 1 or num_samples < 1 or solver not in {"euler", "heun"}:
            raise ValueError("invalid sample count, ODE steps or solver")
        cache_condition_kv = not self.training if cache_condition_kv is None else cache_condition_kv
        cache_time_embeddings = not self.training if cache_time_embeddings is None else cache_time_embeddings
        if self.training and (cache_condition_kv or cache_time_embeddings):
            raise ValueError("sampling caches require eval mode")
        # Encode one immutable snapshot per call; reuse it throughout [0,1].
        encoded = self.encode_conditions(batch, cache_condition_kv=cache_condition_kv)
        b = encoded["history"].shape[0]
        if num_samples > 1:
            for key in ("history", "action", "action_padding_mask", "future_pe"):
                encoded[key] = encoded[key].repeat_interleave(num_samples, 0)
            if cache_condition_kv:
                encoded["condition_kv_cache"] = tuple(
                    tuple(value.repeat_interleave(num_samples, 0) for value in cache)
                    for cache in encoded["condition_kv_cache"])
        shape = (b, num_samples, self.future_horizon, self.latent_dim)
        z = torch.randn(shape, device=encoded["history"].device,
                        dtype=encoded["history"].dtype) if source_noise is None else source_noise
        if z.shape != shape or z.device != encoded["history"].device:
            raise ValueError(f"source_noise must have shape {shape} on the condition device")
        # source_noise is the base epsilon for both source modes.  This allows
        # fixed-noise, paired comparisons against the original checkpoint.
        z = self.source_from_noise(z, encoded).clone().reshape(
            b*num_samples, self.future_horizon, self.latent_dim)
        integrate = self.integrate_latent if integration_fn is None else integration_fn
        z = integrate(z, encoded, steps=steps, solver=solver, cache_time_embeddings=cache_time_embeddings)
        if integration_fn is not None:
            # Compiled/CUDA-graph integrators may reuse their workspace on the
            # next request. Public sampled trajectories own their storage.
            z = z.clone()
        raw = z.float()*self.latent_std + self.latent_mean
        out = {key:value.reshape(b, num_samples, self.future_horizon, -1) for key,value in self.decode(raw).items()}
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
    # Normalized public sampling needs only the frozen NEXT motion conversion.
    # Validate physical q/tau/action statistics when its adapter is prepared.
    return model.to(device).eval().prepare_runtime_normalizers(physical=False), payload
