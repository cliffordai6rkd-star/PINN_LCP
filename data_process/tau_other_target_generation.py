"""Build causal tau_other supervision from reusable q/dq/tau episode streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_TIMESTAMP_SCALES = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


@dataclass(frozen=True)
class TauOtherTargetBuildResult:
    tau_other: torch.Tensor
    dq: torch.Tensor
    ddq: torch.Tensor
    tau_g: torch.Tensor


def normalize_tau_other_target_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the checkpointed target-generation contract."""

    raw = config.get("target_generation") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("target_generation must be a mapping")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False}

    allowed = {
        "enabled",
        "method",
        "target_key",
        "timestamp_key",
        "timestamp_unit",
        "source_keys",
        "dq_sign",
        "torque_filter_key",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown target_generation options: {unknown}")

    method = str(raw.get("method", "causal_gravity_residual_v1")).lower()
    if method != "causal_gravity_residual_v1":
        raise ValueError(
            "target_generation.method must be 'causal_gravity_residual_v1'"
        )

    source_keys = raw.get("source_keys") or {"q": "q", "dq": "dq", "tau": "tau"}
    if not isinstance(source_keys, Mapping):
        raise ValueError("target_generation.source_keys must be a mapping")
    missing_sources = sorted({"q", "dq", "tau"} - set(source_keys))
    unknown_sources = sorted(set(source_keys) - {"q", "dq", "tau"})
    if missing_sources or unknown_sources:
        raise ValueError(
            "target_generation.source_keys must contain exactly q, dq, and tau"
        )
    normalized_sources = {key: str(source_keys[key]) for key in ("q", "dq", "tau")}
    if any(not value for value in normalized_sources.values()):
        raise ValueError("target_generation.source_keys values must be non-empty")

    timestamp_unit = str(raw.get("timestamp_unit", "s")).lower()
    if timestamp_unit not in _TIMESTAMP_SCALES:
        raise ValueError(
            "target_generation.timestamp_unit must be one of s, ms, us, or ns"
        )

    dq_sign = raw.get("dq_sign")
    if dq_sign is not None:
        if not isinstance(dq_sign, Sequence) or isinstance(dq_sign, (str, bytes)):
            raise ValueError("target_generation.dq_sign must be a list or null")
        dq_sign = [float(value) for value in dq_sign]
        if not dq_sign or any(value not in {-1.0, 1.0} for value in dq_sign):
            raise ValueError("target_generation.dq_sign must contain only -1 or 1")

    return {
        "enabled": True,
        "method": method,
        "target_key": str(raw.get("target_key", "tau_other")),
        "timestamp_key": str(raw.get("timestamp_key", "timestamp")),
        "timestamp_unit": timestamp_unit,
        "source_keys": normalized_sources,
        "dq_sign": dq_sign,
        "torque_filter_key": str(raw.get("torque_filter_key", "tau")),
    }


def resolve_tau_other_target_generation(
    config: Mapping[str, Any],
    dataloader_filters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach the exact torque source and operations used by the target."""

    normalized = normalize_tau_other_target_generation(config)
    if not normalized["enabled"]:
        return normalized
    filter_key = normalized["torque_filter_key"]
    filter_spec = dataloader_filters.get(filter_key) or {}
    operations = (
        list(filter_spec.get("operations") or [])
        if bool(filter_spec.get("enabled", False))
        else []
    )
    if bool(filter_spec.get("enabled", False)) and operations:
        raise ValueError(
            "gravity tau_other target generation requires measured "
            "observation.torque without a dataloader torque filter"
        )
    physics = config.get("physics") or {}
    pinocchio = physics.get("pinocchio") or {}
    if not isinstance(pinocchio, Mapping):
        raise ValueError("physics.pinocchio must be a mapping")
    data_config = config.get("dataloader") or {}
    lowdim_keys = data_config.get("lowdim_keys") or {}
    measured_tau_source = str(
        lowdim_keys.get(
            normalized["source_keys"]["tau"],
            normalized["source_keys"]["tau"],
        )
    )
    return {
        **normalized,
        "torque_filter_operations": operations,
        "pinocchio": dict(pinocchio),
        "measured_tau_source": measured_tau_source,
        "ddq_source": "unused",
        "residual_formula": "tau_other=tau_measured-tau_g",
    }


def timestamps_to_seconds(values: torch.Tensor, unit: str) -> np.ndarray:
    timestamps = torch.as_tensor(values).detach().cpu().numpy().reshape(-1)
    timestamps = timestamps.astype(np.float64) * _TIMESTAMP_SCALES[unit]
    if not np.isfinite(timestamps).all():
        raise ValueError("target-generation timestamps must be finite")
    return timestamps


def build_causal_tau_other_target(
    *,
    timestamps_s: np.ndarray,
    q: torch.Tensor,
    dq: torch.Tensor,
    tau_measured: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    target_config: Mapping[str, Any],
    dynamics: Any,
) -> TauOtherTargetBuildResult:
    """Derive episode-local residual labels from measured torque and dynamics."""

    q_np = torch.as_tensor(q).detach().cpu().to(torch.float64).numpy()
    dq_np = torch.as_tensor(dq).detach().cpu().to(torch.float64).numpy()
    tau_np = torch.as_tensor(tau_measured).detach().cpu().to(torch.float64).numpy()
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if q_np.ndim != 2 or dq_np.shape != q_np.shape or tau_np.shape != q_np.shape:
        raise ValueError("q, dq, and tau must share shape [frames, joints]")
    if len(timestamps_s) != len(q_np):
        raise ValueError("timestamps must align with q, dq, and tau")
    if not np.isfinite(q_np).all() or not np.isfinite(dq_np).all():
        raise ValueError("causal tau_other generation requires finite q and dq")
    if not np.isfinite(tau_np).all():
        raise ValueError("causal tau_other generation requires finite measured tau")

    joint_count = q_np.shape[1]
    configured_sign = target_config.get("dq_sign")
    dq_sign = (
        np.ones(joint_count, dtype=np.float64)
        if configured_sign is None
        else np.asarray(configured_sign, dtype=np.float64)
    )
    if dq_sign.shape != (joint_count,):
        raise ValueError(
            f"target_generation.dq_sign must have {joint_count} entries"
        )
    dq_corrected = dq_np * dq_sign[None, :]
    tau_other = np.empty_like(tau_np)
    ddq = np.zeros_like(q_np)
    tau_g_all = np.empty_like(tau_np)

    covered = np.zeros(len(q_np), dtype=bool)
    for episode in episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        if start < 0 or stop > len(q_np) or stop - start < 2:
            raise ValueError(
                "target generation requires every episode to contain at least two "
                f"aligned frames, got [{start}, {stop})"
            )
        if covered[start:stop].any():
            raise ValueError("episode metadata overlaps during target generation")
        tau_g = (
            dynamics.gravity_torque(
                torch.as_tensor(q_np[start:stop], dtype=torch.float64)
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        tau_other[start:stop] = tau_np[start:stop] - tau_g
        tau_g_all[start:stop] = tau_g
        covered[start:stop] = True

    if not covered.all():
        missing = int((~covered).sum())
        raise ValueError(f"episode metadata leaves {missing} frames uncovered")
    return TauOtherTargetBuildResult(
        tau_other=torch.as_tensor(tau_other, dtype=torch.float32),
        dq=torch.as_tensor(dq_corrected, dtype=torch.float32),
        ddq=torch.as_tensor(ddq, dtype=torch.float32),
        tau_g=torch.as_tensor(tau_g_all, dtype=torch.float32),
    )
