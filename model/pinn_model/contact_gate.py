"""Offline contact labels and predicted-contact gating for V4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ContactGateConfig:
    enabled: bool = False
    label_mode: str = "binary"
    metric: str = "force_xyz_l2"
    wrench_key: str = "wrench"
    contact_key: str = "contact"
    force_on_threshold_n: float = 0.5
    force_off_threshold_n: float = 0.3
    signal_on_threshold: float | None = None
    signal_off_threshold: float | None = None
    consecutive_frames: int = 3
    head_hidden_dim: int = 64
    probability_threshold: float = 0.5
    positive_class_weight: float | str = "auto"
    label_cache_path: str | None = None
    contact_threshold: float = 10.0
    precontact_duration_s: float = 1.0
    state_rate_hz: float = 100.0

    @classmethod
    def from_config(cls, config: Mapping):
        values = config.get("contact_gate") or {}
        if str(values.get("label_mode", "binary")).lower() == "three_phase":
            removed = {"phase_label_mode", "precontact_frames", "thresholds", "on_threshold",
                       "off_threshold", "consecutive_frames", "force_on_threshold_n",
                       "force_off_threshold_n", "signal_on_threshold", "signal_off_threshold"} & values.keys()
            if removed:
                raise ValueError(f"Removed dual-threshold three-phase options: {sorted(removed)}; "
                                 "use contact_threshold and precontact_duration_s")
        metric = str(values.get("metric", "force_xyz_l2")).lower()
        threshold_sets = values.get("thresholds") or {}
        metric_thresholds = threshold_sets.get(metric) or {}
        # YAML 1.1 parses unquoted ``on``/``off`` as booleans.
        configured_threshold_on = metric_thresholds.get(
            "on", metric_thresholds.get(True)
        )
        configured_threshold_off = metric_thresholds.get(
            "off", metric_thresholds.get(False)
        )
        configured_on = values.get("on_threshold")
        if configured_on is None:
            configured_on = configured_threshold_on
        configured_off = values.get("off_threshold")
        if configured_off is None:
            configured_off = configured_threshold_off
        positive_weight = values.get("positive_class_weight", "auto")
        if isinstance(positive_weight, str):
            positive_weight = positive_weight.lower()
        else:
            positive_weight = float(positive_weight)
        result = cls(
            enabled=bool(values.get("enabled", False)),
            label_mode=str(values.get("label_mode", "binary")).lower(),
            metric=metric,
            wrench_key=str(values.get("wrench_key", "wrench")),
            contact_key=str(values.get("contact_key", "contact")),
            force_on_threshold_n=float(
                values.get("force_on_threshold_n", 0.5)
            ),
            force_off_threshold_n=float(
                values.get("force_off_threshold_n", 0.3)
            ),
            signal_on_threshold=(
                None
                if configured_on is None
                else float(configured_on)
            ),
            signal_off_threshold=(
                None
                if configured_off is None
                else float(configured_off)
            ),
            consecutive_frames=int(values.get("consecutive_frames", 3)),
            head_hidden_dim=int(values.get("head_hidden_dim", 64)),
            probability_threshold=float(
                values.get("probability_threshold", 0.5)
            ),
            positive_class_weight=positive_weight,
            label_cache_path=(
                None
                if values.get("label_cache_path") is None
                else str(values["label_cache_path"])
            ),
            contact_threshold=float(values.get("contact_threshold", 10.0)),
            precontact_duration_s=float(values.get("precontact_duration_s", 1.0)),
            state_rate_hz=float(
                (config.get("dataloader") or {}).get("high_fps", 100.0)
            ),
        )
        result.validate()
        return result

    def validate(self):
        if self.label_mode not in {"binary", "three_phase"}:
            raise ValueError(
                "contact_gate.label_mode must be 'binary' or 'three_phase'"
            )
        if not math.isfinite(self.contact_threshold) or self.contact_threshold < 0:
            raise ValueError("contact_gate.contact_threshold must be finite and non-negative")
        if not math.isfinite(self.precontact_duration_s) or self.precontact_duration_s < 0:
            raise ValueError("contact_gate.precontact_duration_s must be finite and non-negative")
        if not math.isfinite(self.state_rate_hz) or self.state_rate_hz <= 0.0:
            raise ValueError("dataloader.high_fps must be positive")
        if self.metric not in {"force_xyz_l2", "wrench_l2", "tau_ext_l1", "tau_ext_l2"}:
            raise ValueError(
                "contact_gate.metric must be force_xyz_l2, wrench_l2, tau_ext_l1, or tau_ext_l2"
            )
        if self.label_mode == "binary":
            if self.off_threshold < 0.0:
                raise ValueError("contact_gate.off_threshold must be non-negative")
            if self.on_threshold <= self.off_threshold:
                raise ValueError("contact_gate.on_threshold must be greater than off_threshold")
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError(
                "contact_gate.probability_threshold must be in (0, 1)"
            )
        if self.label_mode == "binary" and self.consecutive_frames < 1:
            raise ValueError("contact_gate.consecutive_frames must be positive")
        if self.head_hidden_dim < 1:
            raise ValueError("contact_gate.head_hidden_dim must be positive")
        if not self.wrench_key or not self.contact_key:
            raise ValueError("contact_gate wrench_key/contact_key must not be empty")
        weight = self.positive_class_weight
        if weight != "auto" and (not isinstance(weight, float) or weight <= 0.0):
            raise ValueError(
                "contact_gate.positive_class_weight must be 'auto' or positive"
            )

    @property
    def on_threshold(self) -> float:
        return (
            self.force_on_threshold_n
            if self.signal_on_threshold is None
            else self.signal_on_threshold
        )

    @property
    def off_threshold(self) -> float:
        return (
            self.force_off_threshold_n
            if self.signal_off_threshold is None
            else self.signal_off_threshold
        )


def hysteresis_binary_mask(
    signal: torch.Tensor,
    *,
    on_threshold: float,
    off_threshold: float,
    consecutive_frames: int,
    backfill: bool = True,
) -> torch.Tensor:
    """Convert one scalar episode timeline into a stable binary contact mask."""

    values = torch.as_tensor(signal)
    if values.ndim != 1:
        raise ValueError(f"hysteresis signal must have shape [T], got {values.shape}")
    if on_threshold <= off_threshold:
        raise ValueError("on_threshold must be greater than off_threshold")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be positive")

    mask = torch.zeros_like(values, dtype=torch.float32)
    contact = False
    candidate_count = 0
    for index in range(values.shape[0]):
        value = float(values[index])
        if not contact:
            if value >= on_threshold:
                candidate_count += 1
            else:
                candidate_count = 0
            if candidate_count >= consecutive_frames:
                contact = True
                start = index - consecutive_frames + 1 if backfill else index
                mask[start : index + 1] = 1.0
                candidate_count = 0
        else:
            mask[index] = 1.0
            if value <= off_threshold:
                candidate_count += 1
            else:
                candidate_count = 0
            if candidate_count >= consecutive_frames:
                contact = False
                if backfill:
                    start = index - consecutive_frames + 1
                    mask[start : index + 1] = 0.0
                candidate_count = 0
    return mask


def contact_labels_from_wrench(
    wrench: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    """Create [T, 1] labels while resetting hysteresis at episode boundaries."""

    values = torch.as_tensor(wrench)
    if values.ndim != 2 or values.shape[-1] != 6:
        raise ValueError(f"wrench must have shape [T, 6], got {values.shape}")
    labels = torch.zeros((values.shape[0], 1), dtype=torch.float32)
    for start, end in episode_bounds:
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > values.shape[0]:
            raise ValueError(f"invalid episode bounds [{start}, {end})")
        force_norm = torch.linalg.vector_norm(values[start:end, :3], dim=-1)
        labels[start:end, 0] = hysteresis_binary_mask(
            force_norm,
            on_threshold=config.on_threshold,
            off_threshold=config.off_threshold,
            consecutive_frames=config.consecutive_frames,
            backfill=True,
        )
    return labels


def batched_contact_phase_mask(signal, *, contact_threshold=10., precontact_duration_s=1., state_rate_hz=100.):
    """Single threshold with a future alignment prefix, for uniform rollout grids."""
    values = torch.as_tensor(signal).detach()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("batched contact signal must be finite [B,T]")
    if not math.isfinite(contact_threshold) or contact_threshold < 0:
        raise ValueError("contact_threshold must be finite and non-negative")
    if (not math.isfinite(precontact_duration_s) or precontact_duration_s < 0
            or not math.isfinite(state_rate_hz) or state_rate_hz <= 0):
        raise ValueError("invalid contact timing")
    frames = int(math.floor(precontact_duration_s*state_rate_hz+1e-9))
    contact = values > contact_threshold
    padded = torch.nn.functional.pad(contact[:, None].float(), (0, frames))
    upcoming = torch.nn.functional.max_pool1d(padded, frames+1, stride=1)[:, 0].bool()
    return torch.where(contact, 2., upcoming.float())


def contact_phase_labels_from_wrench(
    wrench: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    """Create [T, 1] three-phase labels from the measured wrench force."""

    values = torch.as_tensor(wrench)
    if values.ndim != 2 or values.shape[-1] != 6:
        raise ValueError(f"wrench must have shape [T, 6], got {values.shape}")
    if config.metric == "force_xyz_l2":
        signal = torch.linalg.vector_norm(values[:, :3], dim=-1)
    elif config.metric == "wrench_l2":
        signal = torch.linalg.vector_norm(values, dim=-1)
    else:
        raise ValueError(
            "tau_ext_l1 labels require a separate tau_ext tensor; use "
            "contact_phase_labels_from_signal"
        )
    return contact_phase_labels_from_signal(signal, episode_bounds, config)


def contact_phase_labels_from_signal(
    signal: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
    *, timestamps_s=None, valid_mask=None, max_gap_s=0.03,
) -> torch.Tensor:
    """Strict > threshold contact; preceding one second is alignment, contact wins.

    Labels are -1 for unknown context. Prefixes never cross episode boundaries,
    invalid rows, or recording gaps. Actual timestamps define seconds when given.
    """
    values = torch.as_tensor(signal).detach().cpu()
    if values.ndim != 1:
        raise ValueError(f"signal must have shape [T], got {values.shape}")
    valid = torch.isfinite(values)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool).cpu()
        if mask.shape != values.shape:
            raise ValueError("valid_mask must align with signal")
        valid &= mask
    times = (torch.arange(len(values), dtype=torch.float64)/config.state_rate_hz
             if timestamps_s is None else torch.as_tensor(timestamps_s, dtype=torch.float64).cpu())
    if times.shape != values.shape or not torch.isfinite(times).all():
        raise ValueError("timestamps must be finite and align with signal")
    labels = torch.full((len(values), 1), -1., dtype=torch.float32)
    for start, end in episode_bounds:
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(values):
            raise ValueError(f"invalid episode bounds [{start}, {end})")
        active = valid[start:end].numpy()
        t = times[start:end].numpy()
        breaks = np.r_[True, (np.diff(t) <= 0) | (np.diff(t) > max_gap_s+1e-9)]
        segment_starts = np.flatnonzero(active & (breaks | ~np.r_[False, active[:-1]]))
        segment_ends = np.flatnonzero(active & (np.r_[breaks[1:], True] | ~np.r_[active[1:], False]))+1
        for lo, hi in zip(segment_starts, segment_ends):
            contact = (values[start+lo:start+hi] > config.contact_threshold).numpy()
            phase = np.zeros(hi-lo, dtype=np.float32)
            onsets = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
            for onset in onsets:
                left = np.searchsorted(t[lo:hi], t[lo+onset]-config.precontact_duration_s-1e-9)
                phase[left:onset] = 1.
            phase[contact] = 2.
            labels[start+lo:start+hi, 0] = torch.from_numpy(phase)
    return labels


def probability_contact_mask(
    probability: torch.Tensor,
    config: ContactGateConfig,
) -> torch.Tensor:
    values = torch.as_tensor(probability)
    original_shape = values.shape
    if values.ndim == 2 and values.shape[-1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(
            f"contact probability must have shape [T] or [T, 1], got {original_shape}"
        )
    return (values >= config.probability_threshold).to(torch.float32)[:, None]


def save_contact_label_cache(
    path: Path,
    labels: torch.Tensor,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
):
    values = torch.as_tensor(labels).detach().cpu().numpy().astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError(f"contact labels must have shape [T, 1], got {values.shape}")
    bounds = np.asarray(episode_bounds, dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("episode_bounds must have shape [E, 2]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        contact=values,
        episode_bounds=bounds,
        force_on_threshold_n=np.asarray(config.on_threshold),
        force_off_threshold_n=np.asarray(config.off_threshold),
        consecutive_frames=np.asarray(config.consecutive_frames, dtype=np.int64),
    )
    temporary.replace(destination)


def load_contact_label_cache(
    path: Path,
    *,
    flow_length: int,
    episode_bounds: Sequence[tuple[int, int]],
    config: ContactGateConfig,
) -> torch.Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"contact label cache does not exist: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "contact",
            "episode_bounds",
            "force_on_threshold_n",
            "force_off_threshold_n",
            "consecutive_frames",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"contact label cache is missing fields: {missing}")
        labels = np.asarray(payload["contact"], dtype=np.float32)
        cached_bounds = np.asarray(payload["episode_bounds"], dtype=np.int64)
        cached_on = float(payload["force_on_threshold_n"])
        cached_off = float(payload["force_off_threshold_n"])
        cached_frames = int(payload["consecutive_frames"])

    expected_bounds = np.asarray(episode_bounds, dtype=np.int64)
    if labels.shape != (flow_length, 1):
        raise ValueError(
            "contact label cache length mismatch: "
            f"{labels.shape} vs {(flow_length, 1)}"
        )
    if not np.array_equal(cached_bounds, expected_bounds):
        raise ValueError("contact label cache episode bounds do not match dataset")
    if (
        cached_on != config.on_threshold
        or cached_off != config.off_threshold
        or cached_frames != config.consecutive_frames
    ):
        raise ValueError(
            "contact label cache thresholds do not match config: "
            f"cache=({cached_on}, {cached_off}, {cached_frames}) "
            f"config=({config.force_on_threshold_n}, "
            f"{config.force_off_threshold_n}, {config.consecutive_frames})"
        )
    return torch.from_numpy(labels.copy())
