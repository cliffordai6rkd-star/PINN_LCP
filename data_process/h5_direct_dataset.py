"""Direct, episode-aware H5 loading without resampling or interpolation.

The training datasets use this module when ``dataloader.backend=h5``.  Every
configured field is read at the row selected by the H5 master timestamp.  The
loader deliberately has no alignment fallback: fields with a different length
must be fixed at collection time or handled by a task-specific dataset.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


log = logging.getLogger(__name__)

_EPISODE_INDEX_PATTERN = re.compile(r"(?:^|_)episode_(\d+)(?:_|\.|$)")
_TIMESTAMP_SCALES_TO_SECONDS = {
    "s": 1.0,
    "ms": 1.0e-3,
    "us": 1.0e-6,
    "ns": 1.0e-9,
}


def load_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Direct H5 dataset loading requires h5py. Activate the PINN "
            "training environment or install h5py there."
        ) from exc
    return h5py


def discover_h5_files(
    root: Path | str,
    patterns: Sequence[str] = ("*.h5", "*.hdf5"),
    max_episodes: int | None = None,
) -> list[Path]:
    """Return a stable, de-duplicated list of episode files."""

    root = Path(root).expanduser()
    if root.is_file():
        files = [root]
    else:
        files = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not files:
        raise FileNotFoundError(
            f"No H5 episode files found under {root} with patterns={list(patterns)}"
        )
    if max_episodes is not None:
        max_episodes = int(max_episodes)
        if max_episodes <= 0:
            raise ValueError("dataloader.max_episodes must be positive or null")
        files = files[:max_episodes]
    return files


def episode_index_from_path(path: Path, fallback: int | None = None) -> int:
    """Use the raw episode number in the filename, not its sorted position."""

    match = _EPISODE_INDEX_PATTERN.search(path.name)
    if match is not None:
        return int(match.group(1))
    if fallback is None:
        raise ValueError(
            f"Cannot infer episode index from {path.name!r}; expected "
            "a name containing episode_<integer>."
        )
    return int(fallback)


def timestamp_scale_to_seconds(unit: str) -> float:
    unit = str(unit).lower()
    try:
        return _TIMESTAMP_SCALES_TO_SECONDS[unit]
    except KeyError as exc:
        raise ValueError(
            "timestamp unit must be one of s, ms, us, ns, "
            f"got {unit!r}"
        ) from exc


def read_h5_array(h5_file: Any, path: str):
    if path not in h5_file:
        raise KeyError(f"H5 file {h5_file.filename} is missing dataset {path!r}")
    value = h5_file[path]
    if not hasattr(value, "shape"):
        raise TypeError(f"H5 path {path!r} in {h5_file.filename} is not a dataset")
    return value[...]


def _read_field(h5_file: Any, spec: str | Mapping[str, Any]):
    if isinstance(spec, str):
        return read_h5_array(h5_file, spec)
    if not isinstance(spec, Mapping):
        raise TypeError(f"H5 field spec must be a path or mapping, got {spec!r}")

    if "path" in spec:
        unknown = set(spec) - {"path", "dtype", "transform"}
        if unknown:
            raise ValueError(f"H5 path field has unknown options: {sorted(unknown)}")
        value = read_h5_array(h5_file, str(spec["path"]))
        transform = spec.get("transform")
        if transform is None:
            return value
        if transform != "ee_pose_matrix_to_quaternion":
            raise ValueError(f"unsupported direct H5 transform {transform!r}")
        from data_process.tool.ee_pose_matrix_to_quaternion import pose_to_quat7

        if value.shape[-2:] != (4, 4):
            raise ValueError(
                "ee_pose_matrix_to_quaternion expects (...,4,4), "
                f"got {value.shape}"
            )
        import numpy as np

        flat = value.reshape(-1, 4, 4)
        converted = np.stack([pose_to_quat7(pose) for pose in flat], axis=0)
        return converted.reshape(value.shape[:-2] + (7,))

    operation = str(spec.get("operation", "")).lower()
    paths = spec.get("paths")
    if operation != "subtract" or not isinstance(paths, (list, tuple)) or len(paths) != 2:
        raise ValueError(
            "Derived H5 fields currently require "
            "{operation: subtract, paths: [minuend, subtrahend]}"
        )
    unknown = set(spec) - {"operation", "paths", "dtype"}
    if unknown:
        raise ValueError(f"H5 subtract field has unknown options: {sorted(unknown)}")
    return read_h5_array(h5_file, str(paths[0])) - read_h5_array(
        h5_file, str(paths[1])
    )


class TensorColumnDataset(torch.utils.data.Dataset):
    """Small HuggingFace-like tensor table used by the existing trainers."""

    def __init__(self, columns: Mapping[str, torch.Tensor], selected=None):
        self._columns = dict(columns)
        self._selected = tuple(selected) if selected is not None else None
        lengths = {int(value.shape[0]) for value in self._columns.values()}
        if len(lengths) != 1:
            raise ValueError(f"tensor columns have inconsistent lengths: {lengths}")
        self._length = lengths.pop() if lengths else 0

    @property
    def column_names(self) -> list[str]:
        return list(self._columns)

    def with_format(self, _format=None, columns=None, output_all_columns=False):
        del _format, output_all_columns
        if columns is None:
            selected = None
        else:
            selected = tuple(columns)
            missing = sorted(set(selected) - set(self._columns))
            if missing:
                raise KeyError(f"tensor table is missing columns: {missing}")
        return TensorColumnDataset(self._columns, selected=selected)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        names = self._selected or tuple(self._columns)
        return {name: self._columns[name][index] for name in names}


class DirectH5EpisodeDataset(torch.utils.data.Dataset):
    """Concatenate aligned H5 episodes while retaining exact boundaries."""

    def __init__(
        self,
        *,
        root: Path | str,
        fields: Mapping[str, str | Mapping[str, Any]],
        timestamp_path: str,
        timestamp_output_key: str = "timestamp",
        timestamp_unit: str = "us",
        timestamp_output_unit: str = "s",
        patterns: Sequence[str] = ("*.h5", "*.hdf5"),
        max_episodes: int | None = None,
        expected_fps: float | None = None,
        max_cadence_error_s: float = 1.0e-6,
    ):
        if not fields:
            raise ValueError("dataloader.h5_fields must not be empty")
        if timestamp_output_key in fields:
            raise ValueError(
                f"timestamp output key {timestamp_output_key!r} conflicts with h5_fields"
            )
        self.root = Path(root)
        self.files = discover_h5_files(self.root, patterns, max_episodes)
        self.fields = dict(fields)
        self.timestamp_path = str(timestamp_path)
        self.timestamp_output_key = str(timestamp_output_key)
        self.timestamp_unit = str(timestamp_unit).lower()
        self.timestamp_output_unit = str(timestamp_output_unit).lower()
        self.timestamp_scale_s = timestamp_scale_to_seconds(self.timestamp_unit)
        self.timestamp_output_scale_s = timestamp_scale_to_seconds(
            self.timestamp_output_unit
        )
        self.expected_fps = (
            None if expected_fps is None else float(expected_fps)
        )
        self.max_cadence_error_s = float(max_cadence_error_s)
        if self.expected_fps is not None and self.expected_fps <= 0.0:
            raise ValueError("dataloader.expected_fps must be positive or null")
        if self.max_cadence_error_s < 0.0:
            raise ValueError("dataloader.max_cadence_error_s must be non-negative")

        columns, episodes = self._load()
        self.hf_dataset = TensorColumnDataset(columns)
        self.meta = SimpleNamespace(episodes=episodes)

    def _load(self):
        h5py = load_h5py()
        buffers = {name: [] for name in self.fields}
        timestamp_buffers = []
        episodes = []
        offset = 0
        seen_episode_indices = set()

        for position, path in enumerate(self.files):
            episode_index = episode_index_from_path(path, fallback=position)
            if episode_index in seen_episode_indices:
                raise ValueError(f"duplicate H5 episode index {episode_index}")
            seen_episode_indices.add(episode_index)

            with h5py.File(path, "r") as h5_file:
                raw_timestamps = read_h5_array(h5_file, self.timestamp_path)
                timestamps = torch.as_tensor(raw_timestamps).reshape(-1)
                if timestamps.numel() == 0:
                    raise ValueError(f"H5 episode {path} has no timestamps")
                timestamps_s = timestamps.to(dtype=torch.float64) * self.timestamp_scale_s
                deltas_s = torch.diff(timestamps_s)
                if deltas_s.numel() and torch.any(deltas_s <= 0.0):
                    raise ValueError(
                        f"H5 timestamps must be strictly increasing in {path}"
                    )
                if self.expected_fps is not None and deltas_s.numel():
                    expected_dt_s = 1.0 / self.expected_fps
                    max_error_s = float((deltas_s - expected_dt_s).abs().max())
                    if max_error_s > self.max_cadence_error_s:
                        raise ValueError(
                            f"H5 episode {path.name} is not uniformly sampled at "
                            f"{self.expected_fps:g} Hz: max cadence error is "
                            f"{max_error_s:.9f} s, allowed "
                            f"{self.max_cadence_error_s:.9f} s. Direct H5 mode "
                            "does not resample."
                        )

                frame_count = int(timestamps.numel())
                for output_key, spec in self.fields.items():
                    array = _read_field(h5_file, spec)
                    if getattr(array, "ndim", 0) == 0 or int(array.shape[0]) != frame_count:
                        raise ValueError(
                            f"H5 field {output_key!r} in {path.name} has shape "
                            f"{getattr(array, 'shape', None)}, expected first dimension "
                            f"{frame_count} from {self.timestamp_path!r}. Direct H5 "
                            "mode will not align or resample mismatched fields."
                        )
                    dtype = spec.get("dtype") if isinstance(spec, Mapping) else None
                    tensor = torch.as_tensor(array)
                    if dtype is not None:
                        torch_dtype = getattr(torch, str(dtype), None)
                        if torch_dtype is None:
                            raise ValueError(f"unknown torch dtype {dtype!r}")
                        tensor = tensor.to(dtype=torch_dtype)
                    elif tensor.is_floating_point():
                        tensor = tensor.to(dtype=torch.float32)
                    buffers[output_key].append(tensor.contiguous())

            output_ratio = self.timestamp_scale_s / self.timestamp_output_scale_s
            rounded_ratio = round(output_ratio)
            if (
                not timestamps.is_floating_point()
                and abs(output_ratio - rounded_ratio) < 1.0e-12
            ):
                output_timestamps = timestamps.to(dtype=torch.int64) * int(
                    rounded_ratio
                )
            else:
                output_timestamps = (
                    timestamps.to(dtype=torch.float64) * output_ratio
                )
            timestamp_buffers.append(output_timestamps.contiguous())
            episodes.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": offset,
                    "dataset_to_index": offset + frame_count,
                    "length": frame_count,
                    "h5_path": str(path),
                }
            )
            offset += frame_count

        columns = {
            name: torch.cat(values, dim=0).contiguous()
            for name, values in buffers.items()
        }
        columns[self.timestamp_output_key] = torch.cat(
            timestamp_buffers, dim=0
        ).contiguous()
        log.info(
            "direct H5 dataset loaded: root=%s episodes=%d frames=%d fields=%s",
            self.root,
            len(episodes),
            offset,
            sorted(columns),
        )
        return columns, episodes

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        return self.hf_dataset[index]
