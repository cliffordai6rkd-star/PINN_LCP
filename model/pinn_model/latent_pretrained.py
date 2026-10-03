"""Strict NEXT LSTM transfer and its numerical input contract."""
import copy
import logging
import math

import torch

from data_process.causal_data_filter import normalize_dataloader_filters
from train.base_trainer import BaseTrainer
from train.nomalizer import Normalizer

log = logging.getLogger(__name__)
MOTION_KEYS = ("q", "dq", "delta_q")


def validate_normalizer(envelope, key, dimension):
    """Validate locally so this family also works with the original Normalizer.

    Some worktrees add Normalizer.validate(); its availability must not become
    an implicit dependency on another user's uncommitted change.
    """
    required = {"gaussian": ("mean", "std"), "limit": ("min", "max"), "quantile": ("q01", "q99")}
    mode = envelope.get("normalize_mode")
    if mode not in required:
        raise ValueError(f"unsupported normalization {mode}")
    eps = float(envelope.get("eps", 1e-6))
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("normalizer eps must be finite and positive")
    stats = envelope.get("stats", {}).get(key, {})
    for name in required[mode]:
        value = stats.get(name)
        if not torch.is_tensor(value) or value.shape != (dimension,) or not torch.isfinite(value).all():
            raise ValueError(f"invalid normalizer {key}.{name}: expected finite per-feature tensor")
    first, second = (stats[name] for name in required[mode])
    if mode == "gaussian" and (second < 0).any():
        raise ValueError(f"negative normalizer std for {key}")
    if mode != "gaussian" and (second < first).any():
        raise ValueError(f"reversed normalizer bounds for {key}")


def normalizer_envelope(normalizer, config):
    data = config.get("dataloader") or {}
    return {"stats": copy.deepcopy(getattr(normalizer, "stats", {})),
            "eps": float(getattr(normalizer, "eps", 1e-6)),
            "normalize_mode": data.get("normalize_mode"),
            "normalize_lowdim_keys": data.get("normalize_lowdim_keys", [])}


def convert_scale(key, value, envelope, *, inverse=False):
    if envelope is None:
        raise ValueError("input normalizer contract is unavailable")
    mode = envelope.get("normalize_mode")
    if mode is None or key not in (envelope.get("normalize_lowdim_keys") or []):
        return value
    if inverse and mode == "quantile":
        raise ValueError("clipped quantile WM inputs cannot be inverted for a pretrained LSTM")
    validate_normalizer(envelope, key, value.shape[-1])
    normalizer = Normalizer(envelope["stats"], eps=envelope.get("eps", 1e-6))
    method = f"{mode}_{'denormalize' if inverse else 'normalize'}"
    return getattr(normalizer, method)(key, value)


def load_motion_checkpoint(model, configured_path, wm_config, *, wm_filters=None,
                           source_preprocessing=None):
    path = BaseTrainer.resolve_resume_checkpoint(configured_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("pretrained checkpoint must contain config and model")
    cm, cd = config.get("model") or {}, config.get("dataloader") or {}
    checks = {"architecture": "lstm", "target_key": "tau", "inputs": list(MOTION_KEYS),
              "history_mode": "stateless_sliding_window", "hidden_dim": 128, "num_layers": 2,
              "output_dim": 7}
    for key, expected in checks.items():
        if cm.get(key) != expected:
            raise ValueError(f"pretrained model.{key}: expected {expected!r}, got {cm.get(key)!r}")
    if model.hidden_dim != 128 or model.joint_dim != 7:
        raise ValueError("NEXT motion transfer requires joint_dim=7, hidden_dim=128")
    for key in MOTION_KEYS:
        if (cm.get("input_dims") or {}).get(key) != 7:
            raise ValueError(f"pretrained input_dims.{key} must explicitly equal 7")
    if cd.get("horizon") != model.history_horizon:
        raise ValueError("pretrained history horizon differs from actual WM LSTM horizon")
    rate = payload.get("sample_rate_hz")
    expected_rate = model.grid.state_rate_hz / model.temporal_stride
    if rate is None or abs(float(rate) - expected_rate) > 1e-6:
        raise ValueError("pretrained sample_rate_hz differs from actual WM LSTM cadence")
    if cd.get("expected_fps", rate) != rate:
        raise ValueError("pretrained configured and saved sample rates disagree")
    if cd.get("pad_history", False):
        raise ValueError("pretrained NEXT history must use real, unpadded windows")
    if "dataloader_filters" not in payload:
        raise ValueError("pretrained checkpoint is missing preprocessing information")
    saved_filters = payload["dataloader_filters"]
    configured_filters = normalize_dataloader_filters(cd)
    wm_filters = wm_filters if wm_filters is not None else normalize_dataloader_filters(
        wm_config.get("dataloader") or {})
    def effective(spec):
        return spec.get("operations", []) if spec.get("enabled", False) else []
    for key in MOTION_KEYS:
        if effective(saved_filters.get(key, {})) != effective(configured_filters.get(key, {})):
            raise ValueError(f"pretrained saved/configured preprocessing disagree for {key}")
        if effective(saved_filters.get(key, {})) != effective(wm_filters.get(key, {})):
            raise ValueError(f"incompatible preprocessing for {key}; denormalization cannot undo filtering/resampling")
        # Source operations already applied before the WM loader are part of
        # the numerical contract too, even if not requested by WM filters.
        if source_preprocessing and source_preprocessing.get(key):
            expected = (config.get("preprocessing_contract") or {}).get("source", {}).get(key)
            if expected != source_preprocessing[key]:
                raise ValueError(f"incompatible source preprocessing for {key}")
    envelope = payload.get("normalizer")
    if not isinstance(envelope, dict) or "stats" not in envelope or "normalize_mode" not in envelope:
        raise ValueError("pretrained checkpoint must save its own normalizer and mode")
    if envelope.get("normalize_mode") != cd.get("normalize_mode") or envelope.get("normalize_lowdim_keys") != cd.get("normalize_lowdim_keys"):
        raise ValueError("pretrained configured/saved normalization contract differs")
    for key in MOTION_KEYS:
        if envelope.get("normalize_mode") is not None:
            if key not in (envelope.get("normalize_lowdim_keys") or []):
                raise ValueError(f"pretrained normalizer does not cover {key}")
            validate_normalizer(envelope, key, 7)
    recurrent = {key.removeprefix("recurrent."):value for key,value in payload["model"].items()
                 if key.startswith("recurrent.")}
    # Strict load checks missing, extra, and all tensor shapes. The NEXT
    # head (which may use 256 hidden units) is deliberately not transferred.
    model.motion_encoder.load_state_dict(recurrent, strict=True)
    model.pretrained_normalizer = copy.deepcopy(envelope)
    model.pretrained_contract = {"selected_file": str(path), "model": checks,
        "input_dims": {key:7 for key in MOTION_KEYS}, "history_horizon": cd["horizon"],
        "sample_rate_hz": rate, "filters": copy.deepcopy(saved_filters),
        "source_preprocessing": copy.deepcopy(source_preprocessing), "snapshot": "model",
        "normalizer": copy.deepcopy(envelope)}
    model.frozen_motion = True
    model.motion_encoder.requires_grad_(False).eval()
    log.info("strict NEXT recurrent transfer selected %s", path)
    return path
