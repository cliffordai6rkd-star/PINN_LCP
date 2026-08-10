"""Shared causal filtering contract for sequence-model torque targets."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from data_process.offline_tau_labels import (
    causal_median_one_pole_filter,
    causal_trailing_median_filter,
)


def torque_target_filter_config(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    filter_config = (config.get("model") or {}).get("target_filter")
    if filter_config is None:
        return None
    if not isinstance(filter_config, Mapping):
        raise ValueError("model.target_filter must be a mapping or null")
    if not bool(filter_config.get("enabled", True)):
        return None

    median_window = int(filter_config.get("median_window", 1))
    apply_additional_lowpass = bool(
        filter_config.get("apply_additional_lowpass", False)
    )
    cutoff_hz = None
    if apply_additional_lowpass:
        if "cutoff_hz" not in filter_config:
            raise ValueError(
                "model.target_filter.cutoff_hz is required when "
                "apply_additional_lowpass=true"
            )
        cutoff_hz = float(filter_config["cutoff_hz"])
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError(
                "model.target_filter.cutoff_hz must be positive and finite"
            )
    if median_window < 1 or median_window % 2 == 0:
        raise ValueError(
            "model.target_filter.median_window must be a positive odd integer"
        )
    return {
        "cutoff_hz": cutoff_hz,
        "median_window": median_window,
        "apply_additional_lowpass": apply_additional_lowpass,
    }


def filter_torque_target_episode(
    timestamps_s: np.ndarray | torch.Tensor,
    tau_nm: np.ndarray | torch.Tensor,
    filter_config: Mapping[str, Any],
) -> np.ndarray:
    """Apply the configured filter to one episode without crossing boundaries."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau_nm, dtype=np.float64)
    median_window = int(filter_config["median_window"])
    if not bool(filter_config["apply_additional_lowpass"]):
        return causal_trailing_median_filter(
            tau,
            median_window=median_window,
        )
    return causal_median_one_pole_filter(
        timestamps,
        tau,
        cutoff_hz=float(filter_config["cutoff_hz"]),
        median_window=median_window,
    )


def filter_torque_target_dataset(
    timestamps_s: np.ndarray | torch.Tensor,
    tau_nm: np.ndarray | torch.Tensor,
    episode_ranges: Sequence[tuple[int, int]],
    filter_config: Mapping[str, Any],
) -> torch.Tensor:
    """Filter a concatenated dataset while resetting state at every episode."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau_nm, dtype=np.float64)
    if tau.ndim != 2 or len(timestamps) != len(tau):
        raise ValueError("torque targets must have aligned [N] timestamps and [N, D] tau")
    filtered = np.empty_like(tau)
    covered = np.zeros(len(tau), dtype=bool)
    for start, end in episode_ranges:
        start, end = int(start), int(end)
        if start < 0 or end <= start or end > len(tau):
            raise ValueError(f"invalid episode range [{start}, {end})")
        if covered[start:end].any():
            raise ValueError("episode ranges must not overlap")
        filtered[start:end] = filter_torque_target_episode(
            timestamps[start:end],
            tau[start:end],
            filter_config,
        )
        covered[start:end] = True
    if not covered.all():
        raise ValueError("episode ranges must cover every torque target frame")
    return torch.as_tensor(filtered, dtype=torch.float32)
