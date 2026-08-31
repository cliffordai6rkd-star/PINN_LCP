"""Training dataset for the torque state world model.

The preferred H5 backend consumes an already uniform, shared state timeline
without running a dataset converter.  The legacy LeRobot packed backend remains
available for old experiments.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from data_process.causal_data_filter import (
    filter_episode_values,
    normalize_dataloader_filters,
)
from model.pinn_model.contact_gate import (
    ContactGateConfig,
    contact_labels_from_wrench,
    contact_phase_labels_from_wrench,
    contact_phase_labels_from_signal,
)
from train.nomalizer import Normalizer


ACTION_CONDITION_FEATURES = (
    "absolute_pose",
    "current_ee_pose",
    "relative_pose",
)

V3_STATE_FIELDS = {
    "q": "observation.joint",
    "dq": "observation.velocity",
    "delta_q": "observation.delta_q",
    "tau": "observation.torque",
    "tau_ext": "observation.tau_ext",
    "action": "action.joint",
}


def compose_action_condition(
    current_pose: torch.Tensor,
    target_pose: torch.Tensor,
    features,
) -> torch.Tensor:
    """Compose absolute/current/relative action tokens in a fixed order."""

    if current_pose.shape[-1] != 7 or target_pose.shape[-1] != 7:
        raise ValueError("pose tensors must have final dimension 7")
    batched = target_pose.ndim == 3
    if batched:
        if current_pose.ndim != 2 or current_pose.shape[0] != target_pose.shape[0]:
            raise ValueError("batched current_pose must have shape [B, 7]")
        batch_size, action_horizon = target_pose.shape[:2]
        current_tokens = current_pose[:, None, :].expand(-1, action_horizon, -1)
    else:
        if target_pose.ndim != 2:
            raise ValueError("target_pose must have shape [A, 7] or [B, A, 7]")
        if current_pose.ndim == 1:
            current_pose = current_pose[None, :]
        if current_pose.shape[0] != 1:
            raise ValueError("unbatched current_pose must contain one pose")
        current_tokens = current_pose[None, :, :].expand(1, target_pose.shape[0], -1)
        target_pose = target_pose[None, ...]

    current_quaternion = torch.nn.functional.normalize(current_tokens[..., 3:], dim=-1)
    target_quaternion = torch.nn.functional.normalize(target_pose[..., 3:], dim=-1)
    inverse = TorqueWorldModelDataset._quat_inverse(current_quaternion)
    relative_position = TorqueWorldModelDataset._quat_rotate(
        inverse,
        target_pose[..., :3] - current_tokens[..., :3],
    )
    relative_quaternion = torch.nn.functional.normalize(
        TorqueWorldModelDataset._quat_multiply(inverse, target_quaternion), dim=-1
    )
    sign = torch.where(
        relative_quaternion[..., 3:4] < 0,
        -torch.ones_like(relative_quaternion[..., 3:4]),
        torch.ones_like(relative_quaternion[..., 3:4]),
    )
    relative_pose = torch.cat(
        (relative_position, relative_quaternion * sign), dim=-1
    )
    absolute_pose = target_pose.clone()
    absolute_pose[..., 3:] = target_quaternion
    current_pose_tokens = current_tokens.clone()
    current_pose_tokens[..., 3:] = current_quaternion
    values = {
        "absolute_pose": absolute_pose,
        "current_ee_pose": current_pose_tokens,
        "relative_pose": relative_pose,
    }
    try:
        result = torch.cat([values[str(feature)] for feature in features], dim=-1)
    except KeyError as exc:
        raise ValueError(
            f"unsupported action condition feature {exc.args[0]!r}; "
            f"choose from {ACTION_CONDITION_FEATURES}"
        ) from exc
    return result if batched else result[0]


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

    A sample is anchored at the latest valid high-rate state timestamp ``t``.
    Its state history ends at ``t`` while a direct action chunk is sampled at
    25 Hz from the configured token offset after the latest expert refresh
    (offset 1 by default). State windows remain on the configured high-rate
    timeline (normally 100 Hz).
    """

    DEFAULT_HIGH_KEYS = {
        "q": "observation.joint",
        "dq": "observation.velocity",
        "delta_q": "observation.delta_q",
        "tau": "observation.torque",
        "tau_ext": "observation.tau_ext",
        "action": "action.joint",
    }
    DEFAULT_H5_HIGH_FIELDS = {
        "q": "teleop/q_follower",
        "dq": "teleop/dq_follower",
        "delta_q": {
            "operation": "subtract",
            "paths": ["teleop/q_cmd", "teleop/q_follower"],
        },
        "tau": "teleop/tau_follower",
        "tau_ext": "teleop/tau_ext_cal",
        "action": "teleop/q_cmd",
    }

    def __init__(self, config, normalizer=None, compute_normalizer=False):
        self.config = config
        self.data_config = config.get("dataloader") or {}
        configured_train_data = config.get("train_data")
        if isinstance(configured_train_data, list):
            configured_train_data = {"sources": configured_train_data}
        if configured_train_data is None:
            configured_train_data = {}
        if not isinstance(configured_train_data, dict):
            raise TypeError("train_data must be a mapping or a list of source mappings")
        self.train_data_config = configured_train_data
        declared_format = str(self.train_data_config.get("format", "")).lower()
        self.v3_only = bool(
            self.data_config.get(
                "v3_only",
                config.get("wm_v3_only", declared_format == "lerobot_v3"),
            )
        )
        configured_sources = self.train_data_config.get("sources")
        if configured_sources:
            source_format = str(
                self.train_data_config.get("format", "h5_v3")
            ).lower()
            if source_format not in {"h5_v3", "lerobot_v3"}:
                raise ValueError(
                    "train_data.sources requires train_data.format=h5_v3 "
                    "or lerobot_v3"
                )
            if source_format == "h5_v3":
                configured_v3_fields = self.train_data_config.get("v3_fields") or {}
                missing_v3 = sorted(set(V3_STATE_FIELDS) - set(configured_v3_fields))
                if missing_v3:
                    raise ValueError(
                        "train_data.v3_fields is missing canonical fields: "
                        f"{missing_v3}"
                    )
                wrong_v3 = {
                    key: configured_v3_fields[key]
                    for key, expected in V3_STATE_FIELDS.items()
                    if str(configured_v3_fields[key]) != expected
                }
                if wrong_v3:
                    raise ValueError(
                        "train_data.v3_fields must use the unified V3 names: "
                        f"{wrong_v3}"
                    )
        # SWM always uses the direct dataset action contract.
        self.direct_action = True
        source_format = str(self.train_data_config.get("format", "")).lower()
        self.backend = str(
            self.train_data_config.get(
                "backend",
                (
                    "h5"
                    if self.train_data_config.get("sources")
                    and source_format != "lerobot_v3"
                    else self.data_config.get("backend", "lerobot")
                ),
            )
        ).lower()
        if self.v3_only and self.backend != "lerobot":
            raise ValueError(
                "WM v3-only dataloader requires dataloader.backend=lerobot and "
                "a converted LeRobot v3 dataset. Run h5_v3_wm.py first."
            )
        self.repo_id = self.data_config.get("repo_id")
        root = self.data_config.get("root")
        self.lerobot_source_specs = []
        if self.backend == "lerobot" and configured_sources:
            self.lerobot_source_specs = self._resolve_lerobot_sources(
                configured_sources
            )
            if self.lerobot_source_specs:
                root = self.lerobot_source_specs[0]["root"]
                self.repo_id = self.lerobot_source_specs[0]["repo_id"]
        elif configured_sources:
            first_source = configured_sources[0]
            if isinstance(first_source, (str, Path)):
                root = self.train_data_config.get("root") or first_source
            elif isinstance(first_source, dict):
                root = self.train_data_config.get("root") or first_source.get(
                    "root", first_source.get("path")
                )
        self.root = Path(root) if root is not None else None
        if self.root is None:
            raise ValueError("dataloader.root is required")
        if self.backend not in {"lerobot", "h5"}:
            raise ValueError(
                "dataloader.backend must be 'lerobot' or 'h5', "
                f"got {self.backend!r}"
            )
        if self.backend == "lerobot":
            if not self.lerobot_source_specs:
                if not self.repo_id:
                    raise ValueError(
                        "dataloader.repo_id is required for backend=lerobot"
                    )
                self.lerobot_source_specs = [
                    {"repo_id": str(self.repo_id), "root": self.root, "name": str(self.repo_id)}
                ]
            LeRobotDataset = _load_lerobot_dataset_class()
            self.source_datasets = [
                LeRobotDataset(
                    repo_id=spec["repo_id"],
                    root=spec["root"],
                    video_backend=self.data_config.get("video_backend", "torchcodec"),
                    download_videos=False,
                )
                for spec in self.lerobot_source_specs
            ]
            self.source_dataset = self.source_datasets[0]
            self.stats_dataset = self.source_dataset.hf_dataset
        else:
            self.source_datasets = []
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
                self.data_config.get(
                    "action_condition_horizon",
                    self.data_config.get("action_horizon", 8),
                ),
            )
        )
        self.configured_action_condition_horizon = int(
            self.data_config.get("action_condition_horizon", self.action_horizon)
        )
        # Optional per-rollout action plans for OPD.  Entry zero is the
        # ordinary action chunk at the sampled state; later entries are
        # re-anchored chunks for successive high-rate rollout states.
        self.action_rollout_horizon = int(
            self.data_config.get("action_rollout_horizon", 0)
        )
        if self.action_rollout_horizon < 0:
            raise ValueError("dataloader.action_rollout_horizon must be non-negative")
        # The state row at t already reflects the currently held command.
        # Action token 0 therefore starts at the next expert refresh by
        # default.  Keep the offset configurable for VLA contracts whose
        # emitted chunk starts at another relative token.
        self.action_start_offset = int(
            self.data_config.get("action_start_offset", 1)
        )
        self.inference_delay_s = float(
            self.data_config.get("inference_delay_s", 0.0)
        )
        self.action_condition_mode = str(
            self.data_config.get("action_condition_mode", "direct")
        ).lower()
        configured_features = self.data_config.get("action_condition_features")
        if configured_features is None:
            configured_features = []
        self.action_condition_features = tuple(str(value) for value in configured_features)
        self.current_pose_source = str(
            self.data_config.get("current_ee_pose_source", "dataset")
        ).lower()
        self.action_resample = str(
            self.data_config.get("action_resample", "pose")
        ).lower()
        # The WM v3 converter samples each action anchor from the first command
        # at-or-after that camera timestamp (``next``), then holds it until the
        # next anchor. Keep action alignment independent from legacy state
        # resampling.
        self.action_alignment = str(
            self.train_data_config.get(
                "action_alignment",
                "next"
                if self.train_data_config.get("sources")
                else (
                    "nearest"
                    if self.action_resample == "pose"
                    else self.action_resample
                ),
            )
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
        # h5_v3_wm stores the held action's *compressed* 25 Hz index on every
        # 100 Hz state row.  Consuming that index is important: reconstructing
        # a chunk from ``state_time + n * 40 ms`` can select the wrong command
        # when camera and teleop clocks are not perfectly aligned.
        self.action_index_key = str(
            self.data_config.get("action_index_key", "timing.action_index")
        )
        self.action_anchor_timestamp_key = str(
            self.data_config.get(
                "action_anchor_timestamp_key",
                "timing.action_anchor_timestamp_ns",
            )
        )
        self.enforce_uniform_high_timestamps = bool(
            self.data_config.get("enforce_uniform_high_timestamps", False)
        )
        self.max_high_timestamp_jitter_s = float(
            self.data_config.get("max_high_timestamp_jitter_s", 1.0e-6)
        )
        configured_high_keys = self.train_data_config.get(
            "v3_fields",
            self.train_data_config.get(
                "high_keys", self.data_config.get("high_keys")
            ),
        ) or {}
        configured_h5_fields = self.train_data_config.get(
            "raw_h5_fields",
            self.train_data_config.get(
                "h5_high_fields", self.data_config.get("h5_high_fields")
            ),
        ) or {}
        self.high_keys = self._merged_optional_mapping(
            self.DEFAULT_HIGH_KEYS,
            configured_high_keys,
        )
        self.h5_high_fields = self._merged_optional_mapping(
            self.DEFAULT_H5_HIGH_FIELDS,
            configured_h5_fields,
        )
        if self.direct_action:
            for legacy_key in ("ddq", "wrench", "reference_pose"):
                if legacy_key not in configured_high_keys:
                    self.high_keys.pop(legacy_key, None)
            # External torque is only needed to build the optional contact
            # labels.  Do not make contact-disabled datasets carry a field
            # they cannot use.
            if not (config.get("contact_gate") or {}).get("enabled", False) and "tau_ext" not in configured_high_keys:
                self.high_keys.pop("tau_ext", None)
            for legacy_key in ("ddq", "wrench", "reference_pose"):
                if legacy_key not in configured_h5_fields:
                    self.h5_high_fields.pop(legacy_key, None)
            if not (config.get("contact_gate") or {}).get("enabled", False) and "tau_ext" not in configured_h5_fields:
                self.h5_high_fields.pop("tau_ext", None)
            action_key = self.data_config.get("action_key", "action.joint")
            self.high_keys.setdefault("action", action_key)
            self.h5_high_fields.setdefault(
                "action", self.data_config.get("h5_action_field", "teleop/q_cmd")
            )
        if self.direct_action:
            self.high_keys.setdefault("delta_q", "observation.delta_q")
            self.h5_high_fields.setdefault(
                "delta_q",
                {
                    "operation": "subtract",
                    "paths": ["teleop/q_cmd", "teleop/q_follower"],
                },
            )
        configured_high_keys = self.data_config.get("high_keys") or {}
        if (config.get("contact_gate") or {}).get("enabled", False):
            self.high_keys.setdefault("tau_ext", "observation.tau_ext")
            self.h5_high_fields.setdefault("tau_ext", "teleop/tau_ext_cal")
        if configured_high_keys.get("tau_ext") is not None:
            self.h5_high_fields.setdefault("tau_ext", "teleop/tau_ext_cal")
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
            if self._combined_episodes is not None:
                self.episodes = self._combined_episodes
            else:
                self.episodes = self._build_virtual_episodes()
        self.filter_config = normalize_dataloader_filters(
            self.data_config,
            {key: key for key in self.high_keys},
        )
        self._apply_v3_filters()
        self._build_action_tables()
        self.dataset = SimpleNamespace(meta=SimpleNamespace(episodes=self.episodes))
        self.current_ee_pose = None
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
                ["q", "dq", "delta_q", "tau", "action"],
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
            "action_condition_horizon": self.configured_action_condition_horizon,
        }
        invalid = [key for key, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"dataloader dimensions must be positive: {invalid}")
        if self.expert_fps <= 0.0:
            raise ValueError("dataloader.expert_fps must be positive")
        if self.inference_delay_s < 0.0:
            raise ValueError("dataloader.inference_delay_s must be non-negative")
        if self.action_start_offset < 0:
            raise ValueError("dataloader.action_start_offset must be non-negative")
        if self.max_high_timestamp_jitter_s < 0.0:
            raise ValueError(
                "dataloader.max_high_timestamp_jitter_s must be non-negative"
            )
        if self.action_condition_mode not in {
            "direct",
        }:
            raise ValueError(
                "dataloader.action_condition_mode must be 'direct'"
            )
        if not self.direct_action:
            raise ValueError(
                "TorqueWorldModelDataset requires a direct action field"
            )
        if self.action_condition_features and any(
            str(value) not in ACTION_CONDITION_FEATURES
            for value in self.action_condition_features
        ):
            raise ValueError("pose action_condition_features are unsupported for SWM; use direct action")
        if self.action_resample not in {"pose", "nearest", "previous"}:
            raise ValueError(
                "dataloader.action_resample must be pose, nearest, or previous"
            )
        if self.action_alignment not in {"nearest", "previous", "next"}:
            raise ValueError(
                "train_data.action_alignment must be 'previous', 'next', or 'nearest'"
            )
        if self.v3_only and not self.train_data_config.get("sources"):
            if str(self.data_config.get("repo_id", "")).strip() == "":
                raise ValueError(
                    "v3-only WM dataloader requires dataloader.repo_id"
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

    @staticmethod
    def _merged_optional_mapping(defaults, overrides):
        merged = dict(defaults)
        for key, value in overrides.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        return merged

    def _resolve_lerobot_sources(self, sources):
        """Normalize multi-source LeRobot v3 entries into repo/root pairs.

        A source is normally a mapping with ``repo_id`` and ``root``.  For
        convenience, a string is treated as a repo id and resolved below the
        optional ``train_data.root`` directory.
        """

        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("train_data.sources must be a non-empty list")
        base_root = self.train_data_config.get("root")
        specs = []
        for position, source in enumerate(sources):
            if isinstance(source, (str, Path)):
                source_text = str(source)
                source_path = Path(source_text).expanduser()
                repo_id = source_path.name if source_path.name else source_text
                root = source_path
                if base_root is not None and not source_path.is_absolute():
                    root = Path(base_root).expanduser() / source_path
            elif isinstance(source, dict):
                repo_id = source.get("repo_id", source.get("name"))
                root_value = source.get("root", source.get("path"))
                if repo_id is None and root_value is not None:
                    repo_id = Path(str(root_value)).name
                if root_value is None and base_root is not None and repo_id is not None:
                    root_value = Path(base_root).expanduser() / str(repo_id)
                root = None if root_value is None else Path(root_value).expanduser()
            else:
                raise TypeError(
                    "each train_data.sources entry must be a string or mapping"
                )
            if not repo_id or root is None:
                raise ValueError(
                    "each LeRobot v3 source requires repo_id and root (or path)"
                )
            specs.append(
                {
                    "repo_id": str(repo_id),
                    "root": root,
                    "name": str(source.get("name", repo_id))
                    if isinstance(source, dict)
                    else str(repo_id),
                    "position": position,
                }
            )
        return specs

    def _load_columns(self):
        """Load one or more LeRobot v3 datasets onto one logical timeline."""

        self._combined_episodes = None
        if len(getattr(self, "source_datasets", [])) <= 1:
            return self._load_single_lerobot_columns(
                self.stats_dataset,
                self.source_dataset,
            )

        loaded = []
        for source_position, source_dataset in enumerate(self.source_datasets):
            high_tensors, high_ts, anchor_ts = self._load_single_lerobot_columns(
                source_dataset.hf_dataset,
                source_dataset,
            )
            loaded.append(
                {
                    "source_position": source_position,
                    "dataset": source_dataset,
                    "high_tensors": high_tensors,
                    "high_ts": high_ts,
                    "anchor_ts": anchor_ts,
                    "action_indices": self.action_indices,
                    "action_anchor_timestamps": self.action_anchor_timestamps,
                    "packed_window_size": self.packed_window_size,
                }
            )

        block_sizes = {int(item["packed_window_size"]) for item in loaded}
        if len(block_sizes) != 1:
            raise ValueError(
                "all LeRobot v3 sources must use the same packed window size; "
                f"got {sorted(block_sizes)}"
            )
        block_size = block_sizes.pop()
        keys = tuple(self.high_keys)
        if any(
            set(item["high_tensors"]) != set(keys)
            for item in loaded
        ):
            raise ValueError("all LeRobot v3 sources must expose the same configured fields")

        combined_tensors = {
            key: torch.cat([item["high_tensors"][key] for item in loaded], dim=0)
            for key in keys
        }
        combined_high_ts = torch.cat([item["high_ts"] for item in loaded], dim=0)
        combined_anchor_ts = torch.cat([item["anchor_ts"] for item in loaded], dim=0)
        combined_action_indices = torch.cat(
            [item["action_indices"] for item in loaded], dim=0
        )
        combined_action_anchor_timestamps = torch.cat(
            [item["action_anchor_timestamps"] for item in loaded], dim=0
        )

        episodes = []
        high_offset = 0
        source_row_offset = 0
        for item in loaded:
            source_dataset = item["dataset"]
            source_name = self.lerobot_source_specs[item["source_position"]]["name"]
            source_rows = int(item["high_ts"].numel() // block_size)
            for local_episode_position, episode in enumerate(source_dataset.meta.episodes):
                row_start = int(episode["dataset_from_index"])
                row_end = int(episode["dataset_to_index"])
                episodes.append(
                    {
                        **dict(episode),
                        "episode_index": len(episodes),
                        "source_episode_index": int(
                            episode.get("episode_index", local_episode_position)
                        ),
                        "source_index": int(item["source_position"]),
                        "source_name": source_name,
                        "source_dataset_from_index": source_row_offset + row_start,
                        "source_dataset_to_index": source_row_offset + row_end,
                        "dataset_from_index": high_offset + row_start * block_size,
                        "dataset_to_index": high_offset + row_end * block_size,
                    }
                )
            high_offset += int(item["high_ts"].numel())
            source_row_offset += source_rows

        self.packed_window_size = block_size
        self.high_dt_ns = int(round(1.0e9 / self.high_fps))
        self.action_period_ns = int(round(1.0e9 / self.expert_fps))
        self.inference_delay_ns = int(round(self.inference_delay_s * 1.0e9))
        self.action_condition_horizon = self.configured_action_condition_horizon
        self.action_indices = combined_action_indices
        self.action_anchor_timestamps = combined_action_anchor_timestamps
        self._combined_episodes = episodes
        self.source_dataset = SimpleNamespace(meta=SimpleNamespace(episodes=episodes))
        self.stats_dataset = SimpleNamespace(
            column_names=list(combined_tensors),
            hf_dataset=None,
        )
        return combined_tensors, combined_high_ts, combined_anchor_ts

    def _load_single_lerobot_columns(self, stats_dataset, source_dataset):
        dataset_keys = list(dict.fromkeys(self.high_keys.values()))
        dataset_keys.extend((self.high_timestamp_key, self.anchor_timestamp_key))
        # These timing columns are optional for older LeRobot exports.  They
        # are included in the first read when available and silently omitted
        # by the fallback path when an old table has no such columns.
        dataset_keys.extend(
            key
            for key in (self.action_index_key, self.action_anchor_timestamp_key)
            if key not in dataset_keys
        )
        try:
            formatted = stats_dataset.with_format(
                "torch", columns=dataset_keys, output_all_columns=False
            )
        except KeyError:
            # Permit the ingestion-time delta_q fallback above when an older
            # v3 table has no materialized observation.delta_q column.
            formatted = stats_dataset.with_format(
                "torch", columns=None, output_all_columns=True
            )
        columns = formatted[:]
        high_tensors = {}
        source_rows = None
        block_size = None
        for key, dataset_key in self.high_keys.items():
            if dataset_key not in columns:
                # Older v3 exports did not materialize observation.delta_q.
                # When the same row contains the command action and measured
                # q, form the dataset command error once at ingestion time.
                # This is distinct from (and never replaces) predicted
                # delta_q supervision.
                if key == "delta_q":
                    action_source = self.high_keys.get("action", "action.joint")
                    q_source = self.high_keys.get("q", "observation.joint")
                    if action_source in columns and q_source in columns:
                        columns[dataset_key] = self._as_tensor(
                            columns[action_source], dtype=torch.float32
                        ) - self._as_tensor(columns[q_source], dtype=torch.float32)
                    else:
                        raise KeyError(f"dual-rate dataset is missing feature {dataset_key!r}")
                else:
                    raise KeyError(f"dual-rate dataset is missing feature {dataset_key!r}")
            value = self._as_tensor(columns[dataset_key], dtype=torch.float32)
            if value.ndim == 2:
                current_source_rows = int(value.shape[0])
                current_block_size = 1
                flattened_value = value.contiguous()
            elif value.ndim == 3:
                current_source_rows = int(value.shape[0])
                current_block_size = int(value.shape[1])
                flattened_value = value.reshape(-1, value.shape[-1]).contiguous()
            else:
                raise ValueError(
                    f"high-rate feature {dataset_key!r} must have [N,D] or [N,S,D], "
                    f"got {tuple(value.shape)}"
                )
            if source_rows is None:
                source_rows = current_source_rows
                block_size = current_block_size
            elif (current_source_rows, current_block_size) != (
                source_rows,
                block_size,
            ):
                raise ValueError(
                    "high-rate fields must share [N] for raw rows or [N,S] "
                    f"for packed rows; {dataset_key!r} has "
                    f"{(current_source_rows, current_block_size)}, expected "
                    f"{(source_rows, block_size)}"
                )
            high_tensors[key] = flattened_value

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
        max_jitter_ns = int(round(self.max_high_timestamp_jitter_s * 1.0e9))
        for episode in source_dataset.meta.episodes:
            row_start = int(episode["dataset_from_index"])
            row_end = int(episode["dataset_to_index"])
            episode_start = row_start * block_size
            episode_end = row_end * block_size
            episode_ts = high_ts[episode_start:episode_end]
            if episode_ts.numel() > 1:
                actual = torch.diff(episode_ts)
                if torch.any(actual <= 0):
                    raise ValueError(
                        f"{self.high_timestamp_key} must be strictly increasing "
                        "within each episode"
                    )
                if self.enforce_uniform_high_timestamps and torch.any(
                    torch.abs(actual - dt_ns) > max_jitter_ns
                ):
                    raise ValueError(
                        f"{self.high_timestamp_key} is not uniform at configured "
                        "high_fps within an episode"
                    )
        self.packed_window_size = block_size
        self.high_dt_ns = dt_ns
        self.action_period_ns = int(round(1.0e9 / self.expert_fps))
        self.inference_delay_ns = int(round(self.inference_delay_s * 1.0e9))
        self.action_condition_horizon = self.configured_action_condition_horizon
        self.action_indices = self._optional_timing_column(
            columns.get(self.action_index_key),
            source_rows=source_rows,
            block_size=block_size,
            name=self.action_index_key,
            dtype=torch.int64,
        )
        self.action_anchor_timestamps = self._optional_timing_column(
            columns.get(self.action_anchor_timestamp_key),
            source_rows=source_rows,
            block_size=block_size,
            name=self.action_anchor_timestamp_key,
            dtype=torch.int64,
        )
        if self.v3_only and (
            self.action_indices is None or self.action_anchor_timestamps is None
        ):
            raise KeyError(
                "v3-only WM data must contain timing.action_index and "
                "timing.action_anchor_timestamp_ns from h5_v3_wm"
            )
        return high_tensors, high_ts, anchor_ts

    def _apply_v3_filters(self):
        """Apply pending causal history-only filters before normalization."""

        pending = {}
        for key, spec in self.filter_config.items():
            operations = spec["operations"]
            prefix = spec["dataset_preprocessed_operations"]
            remaining = operations[len(prefix) :]
            if spec["enabled"] and remaining:
                if key not in self.high_tensors:
                    raise KeyError(
                        f"dataloader.filters.{key} refers to a missing V3 field"
                    )
                pending[key] = remaining
        if not pending:
            return

        for episode in self.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            timestamps_s = (
                self.high_timestamps[start:end].to(dtype=torch.float64)
                - self.high_timestamps[start].to(dtype=torch.float64)
            ) / 1.0e9
            for key, operations in pending.items():
                filtered = filter_episode_values(
                    timestamps_s,
                    self.high_tensors[key][start:end],
                    operations,
                )
                self.high_tensors[key][start:end] = torch.as_tensor(
                    filtered,
                    dtype=self.high_tensors[key].dtype,
                )

    def _optional_timing_column(
        self,
        value,
        *,
        source_rows,
        block_size,
        name,
        dtype,
    ):
        """Flatten an optional packed timing column to the high-rate rows."""

        if value is None:
            return None
        tensor = self._as_tensor(value)
        expected = int(source_rows) * int(block_size)
        if tensor.ndim == 1:
            flat = tensor
        elif tensor.ndim in (2, 3):
            # [N, 1] and [N, S, 1] are both emitted by LeRobot depending on
            # the feature writer/version.
            if tensor.ndim == 2 and tensor.shape[-1] == 1:
                flat = tensor[:, 0]
            elif tensor.ndim == 3 and tensor.shape[-1] == 1:
                flat = tensor[..., 0].reshape(-1)
            else:
                flat = tensor.reshape(-1)
        else:
            raise ValueError(f"{name} must be a scalar timing column, got {tuple(tensor.shape)}")
        if flat.numel() != expected:
            raise ValueError(
                f"{name} has {flat.numel()} values, expected {expected} "
                "for the packed high-rate timeline"
            )
        flat = flat.to(dtype=dtype).contiguous()
        if dtype == torch.int64 and torch.any(flat < 0):
            raise ValueError(f"{name} must contain non-negative indices/timestamps")
        return flat

    def _load_h5_columns(self):
        """Load native uniform H5 rows exactly once, without time alignment."""

        from data_process.h5_direct_dataset import (
            DirectH5EpisodeDataset,
            V3H5Collection,
            load_h5py,
            read_h5_array,
            timestamp_scale_to_seconds,
        )

        timestamp_path = str(
            self.train_data_config.get(
                "timestamp_path",
                self.data_config.get("h5_timestamp_path", "teleop/timestamp_us"),
            )
        )
        timestamp_unit = str(
            self.train_data_config.get(
                "timestamp_unit", self.data_config.get("h5_timestamp_unit", "us")
            )
        )
        anchor_timestamp_path = str(
            self.train_data_config.get(
                "anchor_timestamp_path",
                self.data_config.get(
                    "h5_anchor_timestamp_path", "cameras/wrist/timestamp_us"
                ),
            )
        )
        anchor_timestamp_unit = str(
            self.train_data_config.get(
                "anchor_timestamp_unit",
                self.data_config.get("h5_anchor_timestamp_unit", timestamp_unit),
            )
        )
        patterns = tuple(
            self.train_data_config.get(
                "patterns", self.data_config.get("h5_patterns", ["*.h5", "*.hdf5"])
            )
        )
        sources = self.train_data_config.get("sources")
        if sources:
            cache_config = self.train_data_config.get("cache") or {}
            collection = V3H5Collection(
                sources=sources,
                source_base_root=self.train_data_config.get("root"),
                fields=self.h5_high_fields,
                timestamp_path=timestamp_path,
                timestamp_unit=timestamp_unit,
                anchor_timestamp_path=anchor_timestamp_path,
                anchor_timestamp_unit=anchor_timestamp_unit,
                patterns=patterns,
                expected_fps=self.train_data_config.get("expected_fps", self.high_fps),
                max_cadence_error_s=float(
                    self.train_data_config.get(
                        "max_cadence_error_s",
                        self.data_config.get("max_cadence_error_s", 1.0e-6),
                    )
                ),
                cache_mode=str(cache_config.get("mode", "ram")),
                cache_root=cache_config.get("root"),
                cache_rebuild=bool(cache_config.get("rebuild", False)),
                cache_chunk_rows=int(cache_config.get("chunk_rows", 4096)),
            )
            columns = collection.columns
            high_timestamps = collection.high_timestamps
            anchor_ts = collection.anchor_timestamps
            episodes = collection.episodes
            self.files = collection.files
            self.source_dataset = SimpleNamespace(meta=SimpleNamespace(episodes=episodes))
            self.stats_dataset = SimpleNamespace(
                column_names=list(columns),
                hf_dataset=None,
            )
            self.packed_window_size = 1
            self.high_dt_ns = int(round(1.0e9 / self.high_fps))
            self.action_period_ns = int(round(1.0e9 / self.expert_fps))
            self.inference_delay_ns = int(round(self.inference_delay_s * 1.0e9))
            self.action_condition_horizon = self.configured_action_condition_horizon
            self.action_indices = None
            self.action_anchor_timestamps = None
            return columns, high_timestamps, anchor_ts, episodes

        direct = DirectH5EpisodeDataset(
            root=self.root,
            fields=self.h5_high_fields,
            timestamp_path=timestamp_path,
            timestamp_output_key="__h5_timestamp_ns",
            timestamp_unit=timestamp_unit,
            timestamp_output_unit="ns",
            patterns=patterns,
            max_episodes=self.data_config.get("max_episodes"),
            expected_fps=self.train_data_config.get("expected_fps", self.high_fps),
            max_cadence_error_s=float(
                self.train_data_config.get(
                    "max_cadence_error_s",
                    self.data_config.get("max_cadence_error_s", 1.0e-6),
                )
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
        self.action_condition_horizon = self.configured_action_condition_horizon
        # Native H5 episodes do not carry the compressed action-index column
        # emitted by h5_v3_wm.  Their camera anchors are retained and used by
        # _action_for_anchor as the equivalent action table.
        self.action_indices = None
        self.action_anchor_timestamps = None
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

    def _build_action_tables(self):
        """Index the direct action table for each episode when v3 metadata exists.

        ``h5_v3_wm`` holds one action value on many 100 Hz rows and records a
        compressed ``timing.action_index`` for the held value.  A table of the
        first row for each index lets sampling use exact consecutive 25 Hz
        tokens without guessing from the state clock.
        """

        self._action_tables = {}
        if self.action_indices is None:
            return
        if "action" not in self.high_tensors:
            raise KeyError("timing.action_index is present but the action field is missing")
        for episode in self.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            indices = self.action_indices[start:end]
            if indices.numel() == 0:
                continue
            if torch.any(indices[1:] < indices[:-1]):
                raise ValueError(
                    f"{self.action_index_key} must be non-decreasing within each episode"
                )
            unique, inverse = torch.unique_consecutive(indices, return_inverse=True)
            # The first occurrence is the row carrying the action token.  A
            # Python loop is intentional here: this runs once per episode and
            # keeps the indexing behavior explicit for irregular hold lengths.
            first_rows = []
            for position in range(unique.numel()):
                first_rows.append(int(torch.nonzero(inverse == position, as_tuple=False)[0]))
            rows = torch.as_tensor(first_rows, dtype=torch.long) + start
            if self.action_anchor_timestamps is not None:
                times = self.action_anchor_timestamps.index_select(0, rows)
            else:
                times = self.high_timestamps.index_select(0, rows)
            self._action_tables[id(episode)] = {
                "indices": unique.to(dtype=torch.long),
                "rows": rows,
                "times": times.to(dtype=torch.int64),
            }

    def _indexed_action_for_anchor(self, high_idx, episode):
        """Return an exact consecutive action chunk from v3 action indices."""

        table = self._action_tables.get(id(episode))
        if table is None:
            return None
        current_index = int(self.action_indices[int(high_idx)])
        positions = torch.searchsorted(
            table["indices"], torch.tensor(current_index, dtype=torch.long)
        )
        if int(positions) >= table["indices"].numel() or int(table["indices"][positions]) != current_index:
            raise IndexError("state row refers to an unknown action index")

        start_position = int(positions) + self.action_start_offset
        if start_position >= table["indices"].numel():
            raise IndexError("future action chunk crosses the available action table")
        if self.inference_delay_ns:
            desired_time = int(table["times"][start_position]) + self.inference_delay_ns
            start_position = int(
                torch.searchsorted(
                    table["times"], torch.tensor(desired_time, dtype=torch.int64), right=False
                )
            )
        end_position = start_position + self.action_condition_horizon
        if end_position > table["indices"].numel():
            raise IndexError("future action chunk crosses the available action table")
        rows = table["rows"][start_position:end_position]
        chunk = self.high_tensors["action"].index_select(0, rows)
        times = table["times"][start_position:end_position]
        return {
            "refresh_time": times[0],
            "target_times": times,
            "action_chunk": chunk,
            "condition_abs": chunk,
            "condition": chunk.clone(),
            "condition_mask": torch.ones(
                self.action_condition_horizon, dtype=torch.float32
            ),
            "action_index": table["indices"][start_position:end_position],
        }

    def _action_rollout_for_anchor(self, high_idx, episode):
        """Return re-anchored direct action chunks for OPD rollout states.

        The first chunk corresponds to ``high_idx``.  Each subsequent chunk is
        sampled using the same action-anchor contract as ``_action_for_anchor``
        at ``high_idx + offset``.  The v3 path is vectorized over offsets so
        enabling this optional field does not add one Python search per token.
        """

        horizon = self.action_rollout_horizon
        if horizon <= 0:
            return None

        end = int(episode["dataset_to_index"])
        state_indices = torch.arange(
            int(high_idx), int(high_idx) + horizon, dtype=torch.long
        )
        if int(state_indices[-1]) >= end:
            raise IndexError("action rollout crosses the episode boundary")

        table = self._action_tables.get(id(episode))
        if table is None:
            # Native H5/legacy fallback.  These datasets are much smaller and
            # use irregular timestamp anchors, so retain the exact helper.
            chunks = [
                self._action_for_anchor(int(state_index), episode)["condition"]
                for state_index in state_indices.tolist()
            ]
            return torch.stack(chunks, dim=0)

        current_indices = self.action_indices.index_select(0, state_indices)
        positions = torch.searchsorted(table["indices"], current_indices)
        if torch.any(positions >= table["indices"].numel()):
            raise IndexError("state row refers to an unknown action index")
        if torch.any(table["indices"].index_select(0, positions) != current_indices):
            raise IndexError("state row refers to an unknown action index")

        starts = positions + self.action_start_offset
        if torch.any(starts >= table["indices"].numel()):
            raise IndexError("future action chunk crosses the available action table")
        if self.inference_delay_ns:
            desired_times = table["times"].index_select(0, starts) + self.inference_delay_ns
            starts = torch.searchsorted(table["times"], desired_times, right=False)
            if torch.any(starts >= table["indices"].numel()):
                raise IndexError("future action chunk crosses the available action table")

        token_offsets = torch.arange(
            self.action_condition_horizon, dtype=torch.long
        )
        chunk_positions = starts[:, None] + token_offsets[None, :]
        if torch.any(chunk_positions >= table["indices"].numel()):
            raise IndexError("future action chunk crosses the available action table")
        rows = table["rows"].index_select(0, chunk_positions.reshape(-1))
        return self.high_tensors["action"].index_select(0, rows).reshape(
            horizon, self.action_condition_horizon, -1
        )

    def _build_contact_labels(self):
        bounds = [
            (int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
            for ep in self.episodes
        ]
        if self.contact_gate_config.enabled:
            if self.contact_gate_config.label_mode == "three_phase":
                if self.contact_gate_config.metric in {"tau_ext_l1", "tau_ext_l2"}:
                    if "tau_ext" not in self.high_tensors:
                        raise KeyError(
                            "contact_gate.metric=tau_ext_l1 requires a high_keys "
                            "entry named tau_ext (offline tau_ext_cal), not "
                            "observation.torque"
                        )
                    tau_ext = self.high_tensors["tau_ext"]
                    if not torch.is_tensor(tau_ext):
                        tau_ext = tau_ext.to_tensor()
                    if self.contact_gate_config.metric == "tau_ext_l2":
                        signal = torch.linalg.vector_norm(tau_ext, dim=-1)
                    else:
                        signal = tau_ext.abs().sum(dim=-1)
                    self.contact = contact_phase_labels_from_signal(
                        signal, bounds, self.contact_gate_config
                    )
                else:
                    self.contact = contact_phase_labels_from_wrench(
                        self.high_tensors["wrench"], bounds, self.contact_gate_config
                    )
            else:
                self.contact = contact_labels_from_wrench(
                    self.high_tensors["wrench"], bounds, self.contact_gate_config
                )
        else:
            self.contact = torch.zeros(
                (self.high_tensors["q"].shape[0], 1), dtype=torch.float32
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

    def _sample_indices(self, times, episode, *, resample=None):
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        source_times = self.high_timestamps[start:end]
        times = torch.as_tensor(times, dtype=torch.int64)
        right = torch.searchsorted(source_times, times, right=False)
        right = right.clamp(0, source_times.numel() - 1)
        left = (right - 1).clamp(0, source_times.numel() - 1)
        resample = self.action_resample if resample is None else str(resample)
        if resample == "previous":
            indices = left
            exact = source_times[right] == times
            indices = torch.where(exact, right, indices)
        elif resample == "next":
            indices = right
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

    def _sample_action_values(self, times, episode):
        """Sample direct command values on the high-rate state timeline."""

        if "action" not in self.high_tensors:
            raise KeyError(
                "direct action conditioning requires a high-rate 'action' field"
            )
        indices, _ = self._sample_indices(
            times, episode, resample=self.action_alignment
        )
        return self.high_tensors["action"].index_select(0, indices)

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
        indexed = self._indexed_action_for_anchor(high_idx, episode)
        if indexed is not None:
            return indexed

        # Native H5 mode has the camera-anchor table but no compressed action
        # index.  Use the actual recorded anchors (rather than a synthetic
        # 40 ms grid) to define the chunk start and its timestamps.
        if self.backend == "h5":
            anchor_start = int(episode["anchor_from_index"])
            anchor_end = int(episode["anchor_to_index"])
            anchors = self.anchor_timestamps[anchor_start:anchor_end]
            current_time = self.high_timestamps[int(high_idx)]
            current_position = int(
                torch.searchsorted(anchors, current_time, right=True).clamp(
                    max=anchors.numel()
                )
            ) - 1
            current_position = max(current_position, 0)
            current_position += self.action_start_offset
            if current_position >= anchors.numel():
                raise IndexError("future action chunk crosses the action-anchor table")
            if self.inference_delay_ns:
                desired_time = int(anchors[current_position]) + self.inference_delay_ns
                current_position = int(
                    torch.searchsorted(
                        anchors,
                        torch.tensor(desired_time, dtype=torch.int64),
                        right=False,
                    )
                )
            end_position = current_position + self.action_condition_horizon
            if end_position > anchors.numel():
                raise IndexError("future action chunk crosses the action-anchor table")
            target_times = anchors[current_position:end_position]
            action_chunk = self._sample_action_values(target_times, episode)
            return {
                "refresh_time": target_times[0],
                "target_times": target_times,
                "action_chunk": action_chunk,
                "condition_abs": action_chunk,
                "condition": action_chunk.clone(),
                "condition_mask": torch.ones(
                    self.action_condition_horizon, dtype=torch.float32
                ),
                "action_index": torch.arange(
                    current_position,
                    end_position,
                    dtype=torch.long,
                ),
            }

        refresh_time = self._refresh_time(high_idx, episode)
        target_times = torch.arange(self.action_condition_horizon, dtype=torch.int64)
        target_times = (
            refresh_time
            + self.inference_delay_ns
            + (target_times + self.action_start_offset) * self.action_period_ns
        )
        if not self.direct_action:
            raise ValueError(
                "TorqueWorldModelDataset requires direct action conditioning; "
                "set dataloader.action_key (for example action.joint)."
            )
        # The action horizon is 8 tokens at 25 Hz while state history and
        # prediction windows remain on the 100 Hz timeline.
        action_chunk = self._sample_action_values(target_times, episode)
        condition_abs = action_chunk
        condition = action_chunk.clone()
        mask = torch.ones(self.action_condition_horizon, dtype=torch.bool)
        return {
            "refresh_time": target_times[0],
            "target_times": target_times,
            "action_chunk": action_chunk,
            "condition_abs": condition_abs,
            "condition": condition,
            "condition_mask": mask.to(dtype=torch.float32),
            "action_index": torch.full(
                (self.action_condition_horizon,), -1, dtype=torch.long
            ),
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
                    if self.action_rollout_horizon:
                        self._action_rollout_for_anchor(high_idx, episode)
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

    def future_contact_phase(self, high_idx, *, reduction="max"):
        """Summarize contact phase in the supervised future window.

        Phase ids are ordered ``free < align < contact``.  The default
        ``max`` reduction marks a sample by the most advanced phase that can
        occur in its target window, which directly increases coverage of
        impending contact events without changing the current-state history
        distribution.
        """

        if reduction != "max":
            raise ValueError("future contact reduction must be 'max'")
        high_idx = int(high_idx)
        episode = self._episode_for_index(high_idx)
        end = min(
            int(episode["dataset_to_index"]),
            high_idx + self.future_horizon + 1,
        )
        start = high_idx + 1
        if start >= end:
            raise IndexError("future contact window is empty")
        labels = self.contact[start:end].reshape(-1).round().to(dtype=torch.long)
        return int(labels.max().item())

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
            if key == "action":
                if self.direct_action and key == "action":
                    values = []
                    for sample_idx in sample_indices:
                        high_idx = self.valid_indices[int(sample_idx)]
                        episode = self._episode_for_index(high_idx)
                        action = self._action_for_anchor(high_idx, episode)
                        values.append(action["condition"])
                    stats[key] = self._tensor_statistics(torch.cat(values, dim=0))
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
            "action_chunk_index": action["action_index"],
            "expert_action_chunk_abs": action["action_chunk"],
            "action_raw": action["condition"],
            "action_mask": action["condition_mask"],
        }
        if self.action_rollout_horizon:
            action_rollout = self._action_rollout_for_anchor(high_idx, episode)
            sample["action_rollout"] = self._normalize("action", action_rollout)
            sample["action_rollout_mask"] = torch.ones(
                self.action_rollout_horizon,
                self.configured_action_condition_horizon,
                dtype=torch.float32,
            )
        for key, values in self.high_tensors.items():
            if key in {"reference_pose", "action", "tau_ext"}:
                continue
            sample[key] = values.index_select(0, history)
            sample[f"{key}_future"] = values.index_select(0, future)
            sample[f"{key}_future_raw"] = sample[f"{key}_future"].clone()
        if self.current_ee_pose is not None:
            sample["current_ee_pose"] = self.current_ee_pose.index_select(0, history)
            sample["current_ee_pose_future"] = self.current_ee_pose.index_select(0, future)
        if "reference_pose" in self.high_tensors:
            sample["reference_pose"] = self.high_tensors["reference_pose"].index_select(0, history)
        sample["contact_future"] = self.contact.index_select(0, future)
        sample["contact"] = self.contact.index_select(0, history)
        for key in self.normalize_lowdim_keys:
            source_key = (
                "action_raw"
                if key == "action"
                else key
            )
            if source_key in sample:
                sample[key] = self._normalize(key, sample[source_key])
            future_key = f"{key}_future"
            if future_key in sample:
                sample[future_key] = self._normalize(key, sample[future_key])
        if "action" not in sample:
            sample["action"] = sample["action_raw"]
        return sample

    def __getitem__(self, index):
        high_idx = self.valid_indices[int(index)]
        return self._build_sample(high_idx)
