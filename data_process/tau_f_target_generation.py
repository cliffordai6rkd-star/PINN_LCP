"""Build causal tau_f supervision from reusable q/dq/tau episode streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data_process.causal_data_filter import filter_episode_values
from data_process.offline_tau_labels import (
    KalmanRTSConfig,
    estimate_joint_states_causal,
)


_TIMESTAMP_SCALES = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


@dataclass(frozen=True)
class TauFTargetBuildResult:
    tau_f: torch.Tensor
    dq: torch.Tensor
    ddq: torch.Tensor
    tau_id: torch.Tensor
    tau_id_filtered: torch.Tensor


def normalize_tau_f_target_generation(config: Mapping[str, Any]) -> dict[str, Any]:
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
        "state_estimator",
        "dq_sign",
        "rnea_state_source",
        "torque_filter_key",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown target_generation options: {unknown}")

    method = str(raw.get("method", "causal_rnea_residual_v1")).lower()
    if method != "causal_rnea_residual_v1":
        raise ValueError(
            "target_generation.method must be 'causal_rnea_residual_v1'"
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

    state_estimator = raw.get("state_estimator") or {}
    if not isinstance(state_estimator, Mapping):
        raise ValueError("target_generation.state_estimator must be a mapping")
    estimator_allowed = set(asdict(KalmanRTSConfig()))
    unknown_estimator = sorted(set(state_estimator) - estimator_allowed)
    if unknown_estimator:
        raise ValueError(
            "Unknown target_generation.state_estimator options: "
            f"{unknown_estimator}"
        )
    normalized_estimator = asdict(KalmanRTSConfig(**state_estimator))

    rnea_state_source = str(raw.get("rnea_state_source", "measured")).lower()
    if rnea_state_source not in {"measured", "filtered"}:
        raise ValueError(
            "target_generation.rnea_state_source must be measured or filtered"
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
        "target_key": str(raw.get("target_key", "tau_f")),
        "timestamp_key": str(raw.get("timestamp_key", "timestamp")),
        "timestamp_unit": timestamp_unit,
        "source_keys": normalized_sources,
        "state_estimator": normalized_estimator,
        "dq_sign": dq_sign,
        "rnea_state_source": rnea_state_source,
        "torque_filter_key": str(raw.get("torque_filter_key", "tau")),
    }


def resolve_tau_f_target_generation(
    config: Mapping[str, Any],
    dataloader_filters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach the exact torque operations used for measured tau and RNEA tau."""

    normalized = normalize_tau_f_target_generation(config)
    if not normalized["enabled"]:
        return normalized
    filter_key = normalized["torque_filter_key"]
    filter_spec = dataloader_filters.get(filter_key) or {}
    operations = (
        list(filter_spec.get("operations") or [])
        if bool(filter_spec.get("enabled", False))
        else []
    )
    physics = config.get("physics") or {}
    pinocchio = physics.get("pinocchio") or {}
    if not isinstance(pinocchio, Mapping):
        raise ValueError("physics.pinocchio must be a mapping")
    return {
        **normalized,
        "torque_filter_operations": operations,
        "pinocchio": dict(pinocchio),
        "measured_tau_source": "dataloader_filtered_column",
        "ddq_source": "variable_dt_kalman_forward_filter",
        "residual_formula": "tau_f=tau_filtered-tau_id_filtered",
    }


def timestamps_to_seconds(values: torch.Tensor, unit: str) -> np.ndarray:
    timestamps = torch.as_tensor(values).detach().cpu().numpy().reshape(-1)
    timestamps = timestamps.astype(np.float64) * _TIMESTAMP_SCALES[unit]
    if not np.isfinite(timestamps).all():
        raise ValueError("target-generation timestamps must be finite")
    return timestamps


def build_causal_tau_f_target(
    *,
    timestamps_s: np.ndarray,
    q: torch.Tensor,
    dq: torch.Tensor,
    tau_filtered: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    target_config: Mapping[str, Any],
    dynamics: Any,
) -> TauFTargetBuildResult:
    """Derive episode-local causal acceleration and residual torque labels."""

    q_np = torch.as_tensor(q).detach().cpu().to(torch.float64).numpy()
    dq_np = torch.as_tensor(dq).detach().cpu().to(torch.float64).numpy()
    tau_np = torch.as_tensor(tau_filtered).detach().cpu().to(torch.float64).numpy()
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if q_np.ndim != 2 or dq_np.shape != q_np.shape or tau_np.shape != q_np.shape:
        raise ValueError("q, dq, and tau must share shape [frames, joints]")
    if len(timestamps_s) != len(q_np):
        raise ValueError("timestamps must align with q, dq, and tau")
    if not np.isfinite(q_np).all() or not np.isfinite(dq_np).all():
        raise ValueError("causal tau_f generation requires finite q and dq")
    if not np.isfinite(tau_np).all():
        raise ValueError("causal tau_f generation requires finite filtered tau")

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
    estimator_config = KalmanRTSConfig(**target_config["state_estimator"])
    torque_operations = target_config.get("torque_filter_operations") or []
    tau_f = np.empty_like(tau_np)
    ddq = np.empty_like(q_np)
    tau_id_all = np.empty_like(tau_np)
    tau_id_filtered_all = np.empty_like(tau_np)

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
        episode_timestamps = timestamps_s[start:stop]
        estimate = estimate_joint_states_causal(
            episode_timestamps,
            q_np[start:stop],
            dq_corrected[start:stop],
            estimator_config,
        )
        if estimate.segment_starts != (0,):
            raise ValueError(
                "target generation found an internal timestamp gap; split the "
                "source into separate episodes so Kalman and torque filters reset "
                "at the same boundary"
            )
        ddq[start:stop] = estimate.ddq_filtered
        if target_config["rnea_state_source"] == "filtered":
            q_rnea = estimate.q_filtered
            dq_rnea = estimate.dq_filtered
        else:
            q_rnea = q_np[start:stop]
            dq_rnea = dq_corrected[start:stop]
        tau_id = (
            dynamics.inverse_dynamics(
                torch.as_tensor(q_rnea, dtype=torch.float64),
                torch.as_tensor(dq_rnea, dtype=torch.float64),
                torch.as_tensor(estimate.ddq_filtered, dtype=torch.float64),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        tau_id_filtered = filter_episode_values(
            episode_timestamps,
            tau_id,
            torque_operations,
        )
        tau_id_all[start:stop] = tau_id
        tau_id_filtered_all[start:stop] = tau_id_filtered
        tau_f[start:stop] = tau_np[start:stop] - tau_id_filtered
        covered[start:stop] = True

    if not covered.all():
        missing = int((~covered).sum())
        raise ValueError(f"episode metadata leaves {missing} frames uncovered")
    return TauFTargetBuildResult(
        tau_f=torch.as_tensor(tau_f, dtype=torch.float32),
        dq=torch.as_tensor(dq_corrected, dtype=torch.float32),
        ddq=torch.as_tensor(ddq, dtype=torch.float32),
        tau_id=torch.as_tensor(tau_id_all, dtype=torch.float32),
        tau_id_filtered=torch.as_tensor(tau_id_filtered_all, dtype=torch.float32),
    )
