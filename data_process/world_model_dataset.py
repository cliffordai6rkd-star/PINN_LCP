"""Training dataset for the torque state world model.

The preferred H5 backend consumes an already uniform, shared state timeline
without running a dataset converter.  The legacy LeRobot packed backend remains
available for old experiments.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from model.pinn_model.contact_gate import (
    ContactGateConfig,
    contact_labels_from_wrench,
)
from train.nomalizer import Normalizer


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TorqueWorldModelDataset requires the lerobot package."
        ) from exc
    return LeRobotDataset


class TorqueWorldModelDataset(torch.utils.data.Dataset):
    """Build causal high-rate samples with a held low-rate action plan.

    A sample is anchored at an image timestamp ``t``.  Its q/tau history ends
    at ``t`` while the action chunk starts at ``t + inference_delay``.  The
    chunk is selected from the most recent expert refresh and is held until
    the next refresh.  Relative target poses are recomputed from the current
    high-rate EE pose for every sample, never from future q labels.
    """

    DEFAULT_HIGH_KEYS = {
        "q": "observation.joint",
        "dq": "observation.velocity",
        "ddq": "observation.acceleration",
        "tau": "observation.torque",
        "wrench": "observation.wrench_ext",
        "reference_pose": "reference.ee_pose",
    }
    DEFAULT_H5_HIGH_FIELDS = {
        "q": "teleop/q_follower",
        "dq": "teleop/dq_follower",
        "ddq": "teleop/ddq_follower",
        "tau": "teleop/tau_follower",
        "wrench": "teleop/wrench_ext",
        "reference_pose": {
            "path": "teleop/ee_pose_follower",
            "transform": "ee_pose_matrix_to_quaternion",
        },
    }

    def __init__(self, config, normalizer=None, compute_normalizer=False):
        self.config = config
        self.data_config = config.get("dataloader") or {}
        self.backend = str(self.data_config.get("backend", "lerobot")).lower()
        self.repo_id = self.data_config.get("repo_id")
        root = self.data_config.get("root")
        self.root = Path(root) if root is not None else None
        if self.root is None:
            raise ValueError("dataloader.root is required")
        if self.backend not in {"lerobot", "h5"}:
            raise ValueError(
                "dataloader.backend must be 'lerobot' or 'h5', "
                f"got {self.backend!r}"
            )
        if self.backend == "lerobot":
            if not self.repo_id:
                raise ValueError(
                    "dataloader.repo_id is required for backend=lerobot"
                )
            LeRobotDataset = _load_lerobot_dataset_class()
            self.source_dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root,
                video_backend=self.data_config.get("video_backend", "torchcodec"),
                download_videos=False,
            )
            self.stats_dataset = self.source_dataset.hf_dataset
        else:
            self.source_dataset = None
            self.stats_dataset = None

        self.history_horizon = int(
            self.data_config.get(
                "state_history_horizon",
                self.data_config.get("history_horizon", 50),
            )
        )
        self.future_horizon = int(
            self.data_config.get(
                "prediction_horizon",
                self.data_config.get("future_horizon", 40),
            )
        )
        self.high_fps = int(self.data_config.get("high_fps", 80))
        self.expert_fps = float(self.data_config.get("expert_fps", 4.0))
        self.action_horizon = int(
            self.data_config.get(
                "action_chunk_horizon",
                self.data_config.get("action_horizon", 8),
            )
        )
        self.inference_delay_s = float(
            self.data_config.get("inference_delay_s", 0.0)
        )
        self.action_condition_mode = str(
            self.data_config.get("action_condition_mode", "relative_pose")
        ).lower()
        self.action_resample = str(
            self.data_config.get("action_resample", "pose")
        ).lower()
        self.pad_history = bool(self.data_config.get("pad_history", True))
        self.pad_future = bool(self.data_config.get("pad_future", False))
        self.high_timestamp_key = str(
            self.data_config.get("high_timestamp_key", "timing.high_timestamp_ns")
        )
        self.anchor_timestamp_key = str(
            self.data_config.get(
                "anchor_timestamp_key", "timing.anchor_timestamp_ns"
            )
        )
        self.high_keys = {
            **self.DEFAULT_HIGH_KEYS,
            **(self.data_config.get("high_keys") or {}),
        }
        self.h5_high_fields = {
            **self.DEFAULT_H5_HIGH_FIELDS,
            **(self.data_config.get("h5_high_fields") or {}),
        }
        self._validate_config()

        if self.backend == "h5":
            (
                self.high_tensors,
                self.high_timestamps,
                self.anchor_timestamps,
                self.episodes,
            ) = self._load_h5_columns()
        else:
            self.high_tensors, self.high_timestamps, self.anchor_timestamps = (
                self._load_columns()
            )
            self.episodes = self._build_virtual_episodes()
        self.dataset = SimpleNamespace(meta=SimpleNamespace(episodes=self.episodes))

        self.contact_gate_config = ContactGateConfig.from_config(config)
        self._build_contact_labels()

        self.valid_indices = []
        self.raw_idx_to_episode_start = {}
        self.raw_idx_to_episode_end = {}
        self.raw_idx_to_episode = {}
        self._build_valid_indices()

        self.normalize_mode = self.data_config.get("normalize_mode", "gaussian")
        self.normalize_lowdim_keys = list(
            self.data_config.get(
                "normalize_lowdim_keys",
                ["q", "tau", "dq", "ddq", "wrench", "target_relative_pose"],
            )
        )
        self.normalizer = normalizer
        self.is_normalize = normalizer is not None
        if compute_normalizer:
            self.fit_normalizer(range(len(self.valid_indices)))

    def _validate_config(self):
        positive = {
            "state_history_horizon": self.history_horizon,
            "prediction_horizon": self.future_horizon,
            "high_fps": self.high_fps,
            "action_chunk_horizon": self.action_horizon,
        }
        invalid = [key for key, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"dataloader dimensions must be positive: {invalid}")
        if self.expert_fps <= 0.0:
            raise ValueError("dataloader.expert_fps must be positive")
        if self.inference_delay_s < 0.0:
            raise ValueError("dataloader.inference_delay_s must be non-negative")
        if self.action_condition_mode not in {"relative_pose", "absolute_pose"}:
            raise ValueError(
                "dataloader.action_condition_mode must be relative_pose or absolute_pose"
            )
        if self.action_resample not in {"pose", "nearest", "previous"}:
            raise ValueError(
                "dataloader.action_resample must be pose, nearest, or previous"
            )
        sampling_dt = (
            ((self.config.get("model") or {}).get("state_estimator") or {}).get(
                "sampling_dt"
            )
        )
        if sampling_dt is not None:
            expected_dt = 1.0 / self.high_fps
            if abs(float(sampling_dt) - expected_dt) > 1.0e-9:
                raise ValueError(
                    "model.state_estimator.sampling_dt must equal "
                    f"1/dataloader.high_fps ({expected_dt:g}), got "
                    f"{float(sampling_dt):g}"
                )

    @staticmethod
    def _as_tensor(value, dtype=None):
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def _load_columns(self):
        dataset_keys = list(dict.fromkeys(self.high_keys.values()))
        dataset_keys.extend((self.high_timestamp_key, self.anchor_timestamp_key))
        formatted = self.stats_dataset.with_format(
            "torch", columns=dataset_keys, output_all_columns=False
        )
        columns = formatted[:]
        high_tensors = {}
        source_rows = None
        block_size = None
        for key, dataset_key in self.high_keys.items():
            if dataset_key not in columns:
                raise KeyError(f"dual-rate dataset is missing feature {dataset_key!r}")
            value = self._as_tensor(columns[dataset_key], dtype=torch.float32)
            if value.ndim != 3:
                raise ValueError(
                    f"high-rate feature {dataset_key!r} must have [N,S,D], "
                    f"got {tuple(value.shape)}"
                )
            if source_rows is None:
                source_rows, block_size = int(value.shape[0]), int(value.shape[1])
            elif tuple(value.shape[:2]) != (source_rows, block_size):
                raise ValueError(
                    "high-rate fields must share [N,S]; "
                    f"{dataset_key!r} has {tuple(value.shape[:2])}, expected "
                    f"{(source_rows, block_size)}"
                )
            high_tensors[key] = value.reshape(-1, value.shape[-1]).contiguous()

        high_ts = self._as_tensor(columns[self.high_timestamp_key]).reshape(-1)
        anchor_ts = self._as_tensor(columns[self.anchor_timestamp_key]).reshape(-1)
        if high_ts.numel() != source_rows * block_size:
            raise ValueError("timing.high_timestamp_ns does not match packed rows")
        if anchor_ts.numel() != source_rows:
            raise ValueError("timing.anchor_timestamp_ns does not match packed rows")
        high_ts = high_ts.to(dtype=torch.int64).contiguous()
        anchor_ts = anchor_ts.to(dtype=torch.int64).contiguous()
        dt_ns = int(round(1.0e9 / self.high_fps))
        # LeRobot concatenates episodes, while each converted episode has its
        # own physical clock origin. Validate monotonicity and cadence within
        # episodes only; a timestamp reset at an episode boundary is expected.
        for episode in self.source_dataset.meta.episodes:
            row_start = int(episode["dataset_from_index"])
            row_end = int(episode["dataset_to_index"])
            episode_start = row_start * block_size
            episode_end = row_end * block_size
            episode_ts = high_ts[episode_start:episode_end]
            if episode_ts.numel() > 1:
                actual = torch.diff(episode_ts)
                if torch.any(actual <= 0):
                    raise ValueError(
                        "timing.high_timestamp_ns must be strictly increasing "
                        "within each episode"
                    )
                if torch.any(torch.abs(actual - dt_ns) > 1):
                    raise ValueError(
                        "timing.high_timestamp_ns is not uniform at configured "
                        "high_fps within an episode"
                    )
        self.packed_window_size = block_size
        self.high_dt_ns = dt_ns
        self.action_period_ns = int(round(1.0e9 / self.expert_fps))
        self.inference_delay_ns = int(round(self.inference_delay_s * 1.0e9))
        self.action_condition_horizon = int(
            round((self.action_horizon - 1) * self.high_fps / self.expert_fps)
        ) + 1
        return high_tensors, high_ts, anchor_ts

    def _load_h5_columns(self):
        """Load native uniform H5 rows exactly once, without time alignment."""

        from data_process.h5_direct_dataset import (
            DirectH5EpisodeDataset,
            load_h5py,
            read_h5_array,
            timestamp_scale_to_seconds,
        )

        timestamp_path = str(
            self.data_config.get("h5_timestamp_path", "teleop/timestamp_us")
        )
        timestamp_unit = str(self.data_config.get("h5_timestamp_unit", "us"))
        anchor_timestamp_path = str(
            self.data_config.get(
                "h5_anchor_timestamp_path", "cameras/wrist/timestamp_us"
            )
        )
        anchor_timestamp_unit = str(
            self.data_config.get("h5_anchor_timestamp_unit", timestamp_unit)
        )
        patterns = tuple(
            self.data_config.get("h5_patterns", ["*.h5", "*.hdf5"])
        )
        direct = DirectH5EpisodeDataset(
            root=self.root,
            fields=self.h5_high_fields,
            timestamp_path=timestamp_path,
            timestamp_output_key="__h5_timestamp_ns",
            timestamp_unit=timestamp_unit,
            timestamp_output_unit="ns",
            patterns=patterns,
            max_episodes=self.data_config.get("max_episodes"),
            expected_fps=self.high_fps,
            max_cadence_error_s=float(
                self.data_config.get("max_cadence_error_s", 1.0e-6)
            ),
        )
        columns = direct.hf_dataset[:]
        high_tensors = {
            key: self._as_tensor(columns[key], dtype=torch.float32).contiguous()
            for key in self.h5_high_fields
        }
        high_ts = self._as_tensor(
            columns["__h5_timestamp_ns"], dtype=torch.int64
        ).reshape(-1).contiguous()

        h5py = load_h5py()
        anchor_scale_to_ns = timestamp_scale_to_seconds(
            anchor_timestamp_unit
        ) / 1.0e-9
        rounded_anchor_scale = round(anchor_scale_to_ns)
        anchor_buffers = []
        episodes = []
        anchor_offset = 0
        for episode, path in zip(direct.meta.episodes, direct.files):
            with h5py.File(path, "r") as h5_file:
                raw_anchor = torch.as_tensor(
                    read_h5_array(h5_file, anchor_timestamp_path)
                ).reshape(-1)
            if raw_anchor.numel() == 0:
                raise ValueError(f"H5 episode {path.name} has no action anchors")
            if (
                not raw_anchor.is_floating_point()
                and abs(anchor_scale_to_ns - rounded_anchor_scale) < 1.0e-12
            ):
                anchor_ns = raw_anchor.to(dtype=torch.int64) * int(
                    rounded_anchor_scale
                )
            else:
                anchor_ns = (
                    raw_anchor.to(dtype=torch.float64) * anchor_scale_to_ns
                ).round().to(dtype=torch.int64)
            if anchor_ns.numel() > 1 and torch.any(torch.diff(anchor_ns) <= 0):
                raise ValueError(
                    f"H5 action anchors must be strictly increasing in {path.name}"
                )
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            episode_high_ts = high_ts[start:end]
            if int(anchor_ns[-1]) < int(episode_high_ts[0]) or int(
                anchor_ns[0]
            ) > int(episode_high_ts[-1]):
                raise ValueError(
                    f"H5 action-anchor clock does not overlap the state clock in "
                    f"{path.name}"
                )
            episodes.append(
                {
                    **dict(episode),
                    "anchor_from_index": anchor_offset,
                    "anchor_to_index": anchor_offset + int(anchor_ns.numel()),
                }
            )
            anchor_buffers.append(anchor_ns.contiguous())
            anchor_offset += int(anchor_ns.numel())

        anchor_ts = torch.cat(anchor_buffers, dim=0).contiguous()
        self.packed_window_size = 1
        self.high_dt_ns = int(round(1.0e9 / self.high_fps))
        self.action_period_ns = int(round(1.0e9 / self.expert_fps))
        self.inference_delay_ns = int(round(self.inference_delay_s * 1.0e9))
        self.action_condition_horizon = int(
            round((self.action_horizon - 1) * self.high_fps / self.expert_fps)
        ) + 1
        self.source_dataset = direct
        self.stats_dataset = direct.hf_dataset
        return high_tensors, high_ts, anchor_ts, episodes

    def _build_virtual_episodes(self):
        episodes = []
        for episode in self.source_dataset.meta.episodes:
            row_start = int(episode["dataset_from_index"])
            row_end = int(episode["dataset_to_index"])
            high_start = row_start * self.packed_window_size
            high_end = row_end * self.packed_window_size
            episodes.append(
                {
                    **dict(episode),
                    "source_dataset_from_index": row_start,
                    "source_dataset_to_index": row_end,
                    "dataset_from_index": high_start,
                    "dataset_to_index": high_end,
                }
            )
        return episodes

    def _build_contact_labels(self):
        bounds = [
            (int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            for ep in self.episodes
        ]
        if self.contact_gate_config.enabled:
            self.contact = contact_labels_from_wrench(
                self.high_tensors["wrench"], bounds, self.contact_gate_config
            )
        else:
            self.contact = torch.zeros(
                (self.high_tensors["wrench"].shape[0], 1), dtype=torch.float32
            )

    def _episode_for_index(self, high_idx):
        cached = self.raw_idx_to_episode.get(int(high_idx))
        if cached is not None:
            return cached
        for episode in self.episodes:
            if int(episode["dataset_from_index"]) <= high_idx < int(
                episode["dataset_to_index"]
            ):
                return episode
        raise IndexError(f"high-rate index {high_idx} is outside all episodes")

    def _refresh_time(self, high_idx, episode):
        episode_start = int(episode["dataset_from_index"])
        if self.backend == "h5":
            anchor_origin = int(
                self.anchor_timestamps[int(episode["anchor_from_index"])]
            )
        else:
            anchor_origin = int(
                self.anchor_timestamps[int(episode["source_dataset_from_index"])]
            )
        current_time = int(self.high_timestamps[high_idx])
        if current_time < anchor_origin:
            return anchor_origin
        periods = (current_time - anchor_origin) // self.action_period_ns
        return anchor_origin + periods * self.action_period_ns

    def _sample_indices(self, times, episode):
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        source_times = self.high_timestamps[start:end]
        times = torch.as_tensor(times, dtype=torch.int64)
        right = torch.searchsorted(source_times, times, right=False)
        right = right.clamp(0, source_times.numel() - 1)
        left = (right - 1).clamp(0, source_times.numel() - 1)
        if self.action_resample == "previous":
            indices = left
            exact = source_times[right] == times
            indices = torch.where(exact, right, indices)
        else:
            left_distance = (times - source_times[left]).abs()
            right_distance = (source_times[right] - times).abs()
            indices = torch.where(right_distance < left_distance, right, left)
        selected_times = source_times[indices]
        if torch.any(times < source_times[0]) or torch.any(times > source_times[-1]):
            raise IndexError("action target lies outside episode high-rate timeline")
        return indices + start, selected_times

    def _sample_reference_poses(self, times, episode):
        if self.action_resample != "pose":
            indices, _ = self._sample_indices(times, episode)
            return self.high_tensors["reference_pose"].index_select(0, indices)

        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        source_times = self.high_timestamps[start:end]
        query = torch.as_tensor(times, dtype=torch.int64)
        if torch.any(query < source_times[0]) or torch.any(query > source_times[-1]):
            raise IndexError("action target lies outside episode high-rate timeline")
        right = torch.searchsorted(source_times, query, right=False)
        right = right.clamp(0, source_times.numel() - 1)
        left = (right - 1).clamp(0, source_times.numel() - 1)
        left_time = source_times[left]
        right_time = source_times[right]
        denominator = (right_time - left_time).to(dtype=torch.float64)
        alpha = torch.where(
            denominator > 0,
            (query - left_time).to(dtype=torch.float64) / denominator,
            torch.zeros_like(denominator),
        ).to(dtype=torch.float32)
        poses = self.high_tensors["reference_pose"][start:end]
        left_pose = poses.index_select(0, left)
        right_pose = poses.index_select(0, right)
        position = torch.lerp(
            left_pose[:, :3],
            right_pose[:, :3],
            alpha[:, None],
        )
        quaternion = self._quat_slerp(
            left_pose[:, 3:],
            right_pose[:, 3:],
            alpha,
        )
        return torch.cat((position, quaternion), dim=-1)

    @staticmethod
    def _quat_slerp(left, right, alpha):
        left = torch.nn.functional.normalize(left, dim=-1)
        right = torch.nn.functional.normalize(right, dim=-1)
        dot = (left * right).sum(dim=-1, keepdim=True)
        right = torch.where(dot < 0.0, -right, right)
        dot = dot.abs().clamp(max=1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        alpha = alpha[:, None]
        spherical = (
            torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(1e-8) * left
            + torch.sin(alpha * theta) / sin_theta.clamp_min(1e-8) * right
        )
        linear = torch.lerp(left, right, alpha)
        quaternion = torch.where(dot > 0.9995, linear, spherical)
        quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
        return torch.where(quaternion[:, 3:4] < 0.0, -quaternion, quaternion)

    @staticmethod
    def _quat_inverse(quaternion):
        inverse = quaternion.clone()
        inverse[..., :3] = -inverse[..., :3]
        return torch.nn.functional.normalize(inverse, dim=-1)

    @staticmethod
    def _quat_multiply(first, second):
        ax, ay, az, aw = first.unbind(dim=-1)
        bx, by, bz, bw = second.unbind(dim=-1)
        return torch.stack(
            (
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ),
            dim=-1,
        )

    @staticmethod
    def _quat_rotate(quaternion, vector):
        """Rotate vectors by xyzw quaternions without leaving Torch."""
        q_xyz = quaternion[..., :3]
        q_w = quaternion[..., 3:4]
        cross = torch.linalg.cross(q_xyz, vector, dim=-1)
        return vector + 2.0 * (
            q_w * cross + torch.linalg.cross(q_xyz, cross, dim=-1)
        )

    def _relative_pose(self, current, targets):
        current_position = current[:3]
        current_quaternion = torch.nn.functional.normalize(current[3:], dim=-1)
        target_quaternion = torch.nn.functional.normalize(targets[..., 3:], dim=-1)
        inverse = self._quat_inverse(current_quaternion).expand_as(target_quaternion)
        relative_position = self._quat_rotate(
            inverse,
            targets[..., :3] - current_position,
        )
        relative_quaternion = torch.nn.functional.normalize(
            self._quat_multiply(inverse, target_quaternion), dim=-1
        )
        sign = torch.where(
            relative_quaternion[..., 3:4] < 0,
            -torch.ones_like(relative_quaternion[..., 3:4]),
            torch.ones_like(relative_quaternion[..., 3:4]),
        )
        return torch.cat(
            (relative_position, relative_quaternion * sign),
            dim=-1,
        )

    def _action_for_anchor(self, high_idx, episode):
        refresh_time = self._refresh_time(high_idx, episode)
        target_times = torch.arange(self.action_horizon, dtype=torch.int64)
        target_times = (
            refresh_time
            + self.inference_delay_ns
            + target_times * self.action_period_ns
        )
        action_chunk = self._sample_reference_poses(target_times, episode)

        current_time = int(self.high_timestamps[high_idx])
        condition_times = (
            current_time
            + torch.arange(self.action_condition_horizon, dtype=torch.int64)
            * self.high_dt_ns
        )
        relative_times = condition_times - target_times[0]
        slots = torch.div(relative_times, self.action_period_ns, rounding_mode="floor")
        slots = slots.clamp(0, self.action_horizon - 1)
        condition_abs = action_chunk.index_select(0, slots)
        current_pose = self.high_tensors["reference_pose"][high_idx]
        if self.action_condition_mode == "relative_pose":
            condition = self._relative_pose(current_pose, condition_abs)
        else:
            condition = condition_abs.clone()
        mask = (condition_times >= target_times[0]) & (
            condition_times <= target_times[-1]
        )
        # Once a chunk has reached its final waypoint, keep that waypoint as
        # the held target until the next expert refresh instead of emitting an
        # all-invalid attention sequence.
        if not bool(mask.any()):
            mask[-1] = True
        return {
            "refresh_time": torch.tensor(refresh_time, dtype=torch.int64),
            "target_times": target_times,
            "action_chunk": action_chunk,
            "condition_abs": condition_abs,
            "condition": condition,
            "condition_mask": mask.to(dtype=torch.float32),
        }

    def _build_valid_indices(self):
        for episode in self.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            if self.backend == "h5":
                first_anchor = self.anchor_timestamps[
                    int(episode["anchor_from_index"])
                ]
                anchor_start = start + int(
                    torch.searchsorted(
                        self.high_timestamps[start:end], first_anchor
                    ).item()
                )
            else:
                anchor_start = start + self.packed_window_size - 1
            if not self.pad_history:
                anchor_start += self.history_horizon - 1
            last = end - self.future_horizon - 1
            for high_idx in range(anchor_start, max(anchor_start, last + 1)):
                try:
                    self._action_for_anchor(high_idx, episode)
                except IndexError:
                    continue
                self.valid_indices.append(high_idx)
                self.raw_idx_to_episode_start[high_idx] = start
                self.raw_idx_to_episode_end[high_idx] = end
                self.raw_idx_to_episode[high_idx] = episode

    @property
    def flow_length(self):
        return int(self.high_timestamps.numel())

    @property
    def horizon(self):
        return max(self.history_horizon, self.future_horizon)

    def __len__(self):
        return len(self.valid_indices)

    def covered_raw_indices(self, sample_indices):
        covered = set()
        for sample_idx in sample_indices:
            high_idx = self.valid_indices[int(sample_idx)]
            start = self.raw_idx_to_episode_start[high_idx]
            end = self.raw_idx_to_episode_end[high_idx]
            covered.update(range(max(start, high_idx - self.history_horizon + 1), high_idx + 1))
            covered.update(range(high_idx + 1, min(end, high_idx + self.future_horizon + 1)))
        return sorted(covered)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
        self.is_normalize = normalizer is not None

    def fit_normalizer(self, sample_indices):
        if self.normalize_mode is None:
            return
        sample_indices = list(sample_indices)
        if not sample_indices:
            raise ValueError("cannot fit normalizer with no samples")
        stats = {}
        covered = torch.as_tensor(
            self.covered_raw_indices(sample_indices), dtype=torch.long
        )
        for key in self.normalize_lowdim_keys:
            if key == "target_relative_pose":
                stats[key] = self._target_pose_statistics(sample_indices)
                continue
            if key not in self.high_tensors:
                raise KeyError(f"normalization key {key!r} missing from dataset")
            values = self.high_tensors[key].index_select(0, covered)
            stats[key] = self._tensor_statistics(values)
        self.set_normalizer(Normalizer(stats))

    @staticmethod
    def _tensor_statistics(values):
        tensor = values.reshape(-1, values.shape[-1]).to(dtype=torch.float32)
        if tensor.shape[0] == 0:
            raise ValueError("cannot compute statistics from an empty tensor")
        return {
            "mean": tensor.mean(dim=0),
            "std": tensor.std(dim=0, unbiased=False),
            "min": tensor.min(dim=0).values,
            "max": tensor.max(dim=0).values,
            "q01": torch.quantile(tensor, 0.01, dim=0),
            "q99": torch.quantile(tensor, 0.99, dim=0),
        }

    def _target_pose_statistics(self, sample_indices):
        """Stream target-pose moments without materializing all A-step chunks."""
        quantile_budget = int(
            self.data_config.get("normalizer_quantile_samples", 200_000)
        )
        if quantile_budget <= 0:
            raise ValueError("dataloader.normalizer_quantile_samples must be positive")
        estimated_samples = max(
            1,
            quantile_budget // max(self.action_condition_horizon, 1),
        )
        quantile_positions = set(
            torch.linspace(
                0,
                len(sample_indices) - 1,
                steps=min(len(sample_indices), estimated_samples),
            )
            .round()
            .to(dtype=torch.long)
            .tolist()
        )
        count = 0
        total = None
        total_square = None
        minimum = None
        maximum = None
        quantile_values = []
        for position, sample_idx in enumerate(sample_indices):
            high_idx = self.valid_indices[int(sample_idx)]
            episode = self._episode_for_index(high_idx)
            action = self._action_for_anchor(high_idx, episode)
            values = action["condition"][action["condition_mask"].bool()]
            if values.numel() == 0:
                continue
            values64 = values.to(dtype=torch.float64)
            batch_sum = values64.sum(dim=0)
            batch_square = values64.square().sum(dim=0)
            batch_min = values.min(dim=0).values
            batch_max = values.max(dim=0).values
            total = batch_sum if total is None else total + batch_sum
            total_square = (
                batch_square
                if total_square is None
                else total_square + batch_square
            )
            minimum = batch_min if minimum is None else torch.minimum(minimum, batch_min)
            maximum = batch_max if maximum is None else torch.maximum(maximum, batch_max)
            count += values.shape[0]
            if position in quantile_positions:
                quantile_values.append(values)
        if count == 0 or total is None:
            raise ValueError("no valid target_relative_pose tokens for normalization")
        mean64 = total / count
        variance64 = (total_square / count - mean64.square()).clamp_min(0.0)
        quantile_tensor = torch.cat(quantile_values, dim=0)[:quantile_budget]
        return {
            "mean": mean64.to(dtype=torch.float32),
            "std": variance64.sqrt().to(dtype=torch.float32),
            "min": minimum.to(dtype=torch.float32),
            "max": maximum.to(dtype=torch.float32),
            "q01": torch.quantile(quantile_tensor, 0.01, dim=0),
            "q99": torch.quantile(quantile_tensor, 0.99, dim=0),
        }

    def _normalize(self, key, value):
        if not self.is_normalize:
            return value
        if key not in self.normalizer.stats:
            return value
        if self.normalize_mode == "gaussian":
            return self.normalizer.gaussian_normalize(key, value)
        if self.normalize_mode == "limit":
            return self.normalizer.limit_normalize(key, value)
        if self.normalize_mode == "quantile":
            return self.normalizer.quantile_normalize(key, value)
        raise ValueError(f"unknown normalize_mode {self.normalize_mode!r}")

    def _build_sample(self, high_idx):
        episode = self._episode_for_index(high_idx)
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        history = torch.arange(
            high_idx - self.history_horizon + 1, high_idx + 1, dtype=torch.long
        ).clamp_min(start)
        future = torch.arange(
            high_idx + 1, high_idx + self.future_horizon + 1, dtype=torch.long
        )
        if self.pad_future:
            future = future.clamp_max(end - 1)
        elif int(future[-1]) >= end:
            raise IndexError("future window crosses episode boundary")

        action = self._action_for_anchor(high_idx, episode)
        sample = {
            "sample_idx": torch.tensor(high_idx, dtype=torch.long),
            "history_indices": history,
            "future_indices": future,
            "history_timestamp_ns": self.high_timestamps.index_select(0, history),
            "future_timestamp_ns": self.high_timestamps.index_select(0, future),
            "action_update_timestamp_ns": action["refresh_time"],
            "action_chunk_timestamp_ns": action["target_times"],
            "expert_action_chunk_abs": action["action_chunk"],
            "target_pose_abs": action["condition_abs"],
            "target_relative_pose_raw": action["condition"],
            "target_relative_pose_mask": action["condition_mask"],
        }
        for key, values in self.high_tensors.items():
            if key == "reference_pose":
                continue
            sample[key] = values.index_select(0, history)
            sample[f"{key}_future"] = values.index_select(0, future)
            sample[f"{key}_future_raw"] = sample[f"{key}_future"].clone()
        sample["reference_pose"] = self.high_tensors["reference_pose"].index_select(0, history)
        sample["contact_future"] = self.contact.index_select(0, future)
        for key in self.normalize_lowdim_keys:
            source_key = (
                "target_relative_pose_raw"
                if key == "target_relative_pose"
                else key
            )
            if source_key in sample:
                sample[key] = self._normalize(key, sample[source_key])
            future_key = f"{key}_future"
            if future_key in sample:
                sample[future_key] = self._normalize(key, sample[future_key])
        if "target_relative_pose" not in sample:
            sample["target_relative_pose"] = sample["target_relative_pose_raw"]
        return sample

    def __getitem__(self, index):
        high_idx = self.valid_indices[int(index)]
        return self._build_sample(high_idx)
