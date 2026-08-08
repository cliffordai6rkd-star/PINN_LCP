"""Offline contact labels and predicted-contact gating for V4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ContactGateConfig:
    enabled: bool = False
    wrench_key: str = "wrench"
    contact_key: str = "contact"
    force_on_threshold_n: float = 0.5
    force_off_threshold_n: float = 0.3
    consecutive_frames: int = 3
    head_hidden_dim: int = 64
    probability_threshold: float = 0.5
    positive_class_weight: float | str = "auto"
    label_cache_path: str | None = None

    @classmethod
    def from_config(cls, config: Mapping):
        values = config.get("contact_gate") or {}
        positive_weight = values.get("positive_class_weight", "auto")
        if isinstance(positive_weight, str):
            positive_weight = positive_weight.lower()
        else:
            positive_weight = float(positive_weight)
        result = cls(
            enabled=bool(values.get("enabled", False)),
            wrench_key=str(values.get("wrench_key", "wrench")),
            contact_key=str(values.get("contact_key", "contact")),
            force_on_threshold_n=float(
                values.get("force_on_threshold_n", 0.5)
            ),
            force_off_threshold_n=float(
                values.get("force_off_threshold_n", 0.3)
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
        )
        result.validate()
        return result

    def validate(self):
        if self.force_off_threshold_n < 0.0:
            raise ValueError("contact_gate.force_off_threshold_n must be non-negative")
        if self.force_on_threshold_n <= self.force_off_threshold_n:
            raise ValueError(
                "contact_gate.force_on_threshold_n must be greater than "
                "force_off_threshold_n"
            )
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError(
                "contact_gate.probability_threshold must be in (0, 1)"
            )
        if self.consecutive_frames < 1:
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
            on_threshold=config.force_on_threshold_n,
            off_threshold=config.force_off_threshold_n,
            consecutive_frames=config.consecutive_frames,
            backfill=True,
        )
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
        force_on_threshold_n=np.asarray(config.force_on_threshold_n),
        force_off_threshold_n=np.asarray(config.force_off_threshold_n),
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
        cached_on != config.force_on_threshold_n
        or cached_off != config.force_off_threshold_n
        or cached_frames != config.consecutive_frames
    ):
        raise ValueError(
            "contact label cache thresholds do not match config: "
            f"cache=({cached_on}, {cached_off}, {cached_frames}) "
            f"config=({config.force_on_threshold_n}, "
            f"{config.force_off_threshold_n}, {config.consecutive_frames})"
        )
    return torch.from_numpy(labels.copy())
