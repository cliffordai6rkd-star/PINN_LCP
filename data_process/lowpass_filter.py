from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def _as_time_series(values):
    values = np.asarray(values, dtype=np.float64)
    original_shape = values.shape
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    return values, original_shape


def butterworth_lowpass(values, sample_rate_hz, cutoff_hz, order=2, zero_phase=True):
    values_2d, original_shape = _as_time_series(values)
    sample_rate_hz = float(sample_rate_hz)
    cutoff_hz = float(cutoff_hz)

    if values_2d.shape[0] == 0:
        return values_2d.reshape(original_shape)
    if sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")

    nyquist_hz = 0.5 * sample_rate_hz
    if cutoff_hz >= nyquist_hz:
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be smaller than Nyquist frequency {nyquist_hz}"
        )

    b, a = butter(int(order), cutoff_hz / nyquist_hz, btype="low", analog=False)
    if zero_phase:
        padlen = 3 * max(len(a), len(b))
        if values_2d.shape[0] > padlen:
            filtered = filtfilt(b, a, values_2d, axis=0)
        else:
            filtered = lfilter(b, a, values_2d, axis=0)
    else:
        filtered = lfilter(b, a, values_2d, axis=0)

    return filtered.reshape(original_shape)


def central_difference(values, sample_rate_hz):
    values_2d, original_shape = _as_time_series(values)
    if values_2d.shape[0] == 0:
        return values_2d.reshape(original_shape)
    derivative = np.gradient(values_2d, 1.0 / float(sample_rate_hz), axis=0)
    return derivative.reshape(original_shape)


def apply_lowpass_config(signals: dict[str, Any], config: dict[str, Any]):
    filtered = {
        name: np.asarray(values, dtype=np.float64).copy()
        for name, values in signals.items()
    }
    if not config.get("enabled", False):
        return filtered

    sample_rate_hz = float(config["sample_rate_hz"])
    default_order = int(config.get("order", 2))
    default_zero_phase = bool(config.get("zero_phase", True))

    for field_name, field_config in config.get("fields", {}).items():
        if field_name not in filtered or not field_config.get("enabled", False):
            continue

        filtered[field_name] = butterworth_lowpass(
            filtered[field_name],
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=float(field_config["cutoff_hz"]),
            order=int(field_config.get("order", default_order)),
            zero_phase=bool(field_config.get("zero_phase", default_zero_phase)),
        )

    if config.get("derive_v_from_q", False):
        filtered["v"] = central_difference(filtered["q"], sample_rate_hz)
    if config.get("derive_a_from_v", False):
        filtered["a"] = central_difference(filtered["v"], sample_rate_hz)

    return filtered
