from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from data_process.world_model_dataset import TorqueWorldModelDataset


def _write_direct_world_model_h5(path, frames=16, fps=4):
    timestamps_us = 1_000_000 + (
        np.arange(frames, dtype=np.int64) * int(round(1_000_000 / fps))
    )
    values = np.arange(frames * 2, dtype=np.float64).reshape(frames, 2)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    poses[:, 0, 3] = np.arange(frames, dtype=np.float64)
    with h5py.File(path, "w") as h5_file:
        teleop = h5_file.create_group("teleop")
        teleop.create_dataset("timestamp_us", data=timestamps_us)
        teleop.create_dataset("q_follower", data=values)
        teleop.create_dataset("dq_follower", data=values + 100.0)
        teleop.create_dataset("ddq_follower", data=values + 200.0)
        teleop.create_dataset("tau_follower", data=values + 300.0)
        teleop.create_dataset(
            "wrench_ext", data=np.zeros((frames, 6), dtype=np.float64)
        )
        teleop.create_dataset("ee_pose_follower", data=poses)
        camera = h5_file.create_group("cameras").create_group("wrist")
        camera.create_dataset(
            "timestamp_us", data=timestamps_us[3::4]
        )


def test_world_model_can_read_uniform_h5_without_lerobot_or_resampling(tmp_path):
    _write_direct_world_model_h5(tmp_path / "episode_0007_direct.h5")
    config = {
        "dataloader": {
            "backend": "h5",
            "root": str(tmp_path),
            "high_fps": 4,
            "expert_fps": 1,
            "action_chunk_horizon": 2,
            "inference_delay_s": 0.0,
            "state_history_horizon": 3,
            "prediction_horizon": 2,
            "normalize_mode": None,
            "action_condition_mode": "relative_pose",
        },
        "contact_gate": {"enabled": False},
    }

    dataset = TorqueWorldModelDataset(config)
    sample = dataset[0]

    assert dataset.backend == "h5"
    assert dataset.episodes[0]["episode_index"] == 7
    assert dataset.high_timestamps.dtype == torch.int64
    assert dataset.high_timestamps.tolist() == (
        1_000_000_000 + torch.arange(16) * 250_000_000
    ).tolist()
    assert sample["q"].shape == (3, 2)
    assert sample["tau_future"].shape == (2, 2)
    assert sample["target_relative_pose"].shape == (5, 7)
    torch.testing.assert_close(
        dataset.high_tensors["reference_pose"][:, 3:],
        torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(16, 4),
    )


class _FakeHFDataset:
    def __init__(self, columns):
        self.columns = columns

    def with_format(self, *args, **kwargs):
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.columns
        return {key: value[index] for key, value in self.columns.items()}


class _FakeLeRobotDataset:
    columns = None
    episodes = None

    def __init__(self, **kwargs):
        del kwargs
        self.hf_dataset = _FakeHFDataset(self.columns)
        self.meta = SimpleNamespace(episodes=self.episodes)


def _packed_values(rows, block, dim, offset=0.0):
    value = torch.zeros(rows, block, dim)
    for row in range(rows):
        for sub in range(block):
            value[row, sub] = offset + row * block + sub
    return value


def _make_dataset(normalize_mode=None, compute_normalizer=False):
    rows = 4
    block = 4
    # Every block ends at an image anchor.  The virtual high-rate timeline is
    # therefore 0.25 s before the first anchor, then advances continuously.
    high_ts = torch.arange(rows * block, dtype=torch.int64) * 250_000_000
    anchor_ts = high_ts.reshape(rows, block)[:, -1]
    reference = _packed_values(rows, block, 7)
    reference[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    _FakeLeRobotDataset.columns = {
        "observation.joint": _packed_values(rows, block, 2),
        "observation.velocity": _packed_values(rows, block, 2, 100.0),
        "observation.acceleration": _packed_values(rows, block, 2, 200.0),
        "observation.torque": _packed_values(rows, block, 2, 300.0),
        "observation.wrench_ext": _packed_values(rows, block, 6, 400.0),
        "reference.ee_pose": reference,
        "timing.high_timestamp_ns": high_ts.reshape(rows, block, 1),
        "timing.anchor_timestamp_ns": anchor_ts.reshape(rows, 1),
    }
    _FakeLeRobotDataset.episodes = [
        {"dataset_from_index": 0, "dataset_to_index": rows}
    ]
    config = {
        "dataloader": {
            "repo_id": "fake",
            "root": "fake",
            "high_fps": 4,
            "expert_fps": 1,
            "action_chunk_horizon": 2,
            "inference_delay_s": 0.25,
            "state_history_horizon": 3,
            "prediction_horizon": 2,
            "normalize_mode": normalize_mode,
            "action_condition_mode": "relative_pose",
        },
        "contact_gate": {"enabled": False},
    }
    with patch(
        "data_process.world_model_dataset._load_lerobot_dataset_class",
        return_value=_FakeLeRobotDataset,
    ):
        return TorqueWorldModelDataset(
            config,
            compute_normalizer=compute_normalizer,
        )


def test_dataset_keeps_action_chunk_sparse_and_state_high_rate():
    dataset = _make_dataset()
    sample = dataset[0]

    assert sample["q"].shape == (3, 2)
    assert sample["tau_future"].shape == (2, 2)
    assert sample["expert_action_chunk_abs"].shape == (2, 7)
    assert sample["target_relative_pose"].shape == (
        dataset.action_condition_horizon,
        7,
    )
    assert sample["action_chunk_timestamp_ns"].dtype == torch.int64
    assert sample["history_timestamp_ns"].dtype == torch.int64


def test_action_chunk_is_held_until_expert_refresh_and_then_updates():
    dataset = _make_dataset()
    # Valid samples start at the first image anchor (high index 3).  The
    # refresh period is one second, i.e. four high-rate ticks.
    first = dataset.valid_indices.index(3)
    held = dataset[first]
    held_late = dataset[dataset.valid_indices.index(6)]
    refreshed = dataset[dataset.valid_indices.index(7)]

    torch.testing.assert_close(
        held["expert_action_chunk_abs"], held_late["expert_action_chunk_abs"]
    )
    assert not torch.equal(
        held["expert_action_chunk_abs"], refreshed["expert_action_chunk_abs"]
    )
    assert held["action_update_timestamp_ns"].item() == 750_000_000
    assert refreshed["action_update_timestamp_ns"].item() == 1_750_000_000


def test_relative_target_uses_current_pose_at_anchor_not_future_pose():
    dataset = _make_dataset()
    sample = dataset[dataset.valid_indices.index(3)]

    # The first target starts at 1.0 s and the current pose is at 0.75 s.  The
    # relative xyz is therefore computed from the anchor pose, not the target.
    target_x = sample["target_pose_abs"][0, 0]
    current_x = dataset.high_tensors["reference_pose"][3, 0]
    assert sample["target_relative_pose_raw"][0, 0].item() == target_x.item() - current_x.item()


def test_relative_pose_translation_is_expressed_in_current_ee_frame():
    dataset = _make_dataset()
    half_sqrt_two = 2.0**-0.5
    current = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two]
    )
    target = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, half_sqrt_two, half_sqrt_two]]
    )

    relative = dataset._relative_pose(current, target)

    # A +x displacement in the base frame is -y in an EE frame rotated +90 deg.
    torch.testing.assert_close(
        relative[0, :3],
        torch.tensor([0.0, -1.0, 0.0]),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        relative[0, 3:],
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
        rtol=0.0,
    )


def test_dataset_accepts_per_episode_timestamp_resets():
    rows = 4
    block = 4
    episode_timestamps = torch.arange(2 * block, dtype=torch.int64) * 250_000_000
    high_ts = episode_timestamps.repeat(2)
    anchor_ts = torch.full((rows,), 750_000_000, dtype=torch.int64)
    reference = _packed_values(rows, block, 7)
    reference[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    _FakeLeRobotDataset.columns = {
        "observation.joint": _packed_values(rows, block, 2),
        "observation.velocity": _packed_values(rows, block, 2),
        "observation.acceleration": _packed_values(rows, block, 2),
        "observation.torque": _packed_values(rows, block, 2),
        "observation.wrench_ext": _packed_values(rows, block, 6),
        "reference.ee_pose": reference,
        "timing.high_timestamp_ns": high_ts.reshape(rows, block, 1),
        "timing.anchor_timestamp_ns": anchor_ts.reshape(rows, 1),
    }
    _FakeLeRobotDataset.episodes = [
        {"dataset_from_index": 0, "dataset_to_index": 2},
        {"dataset_from_index": 2, "dataset_to_index": 4},
    ]
    config = {
        "dataloader": {
            "repo_id": "fake",
            "root": "fake",
            "high_fps": 4,
            "expert_fps": 1,
            "action_chunk_horizon": 2,
            "inference_delay_s": 0.0,
            "state_history_horizon": 3,
            "prediction_horizon": 2,
            "normalize_mode": None,
        },
        "contact_gate": {"enabled": False},
    }
    with patch(
        "data_process.world_model_dataset._load_lerobot_dataset_class",
        return_value=_FakeLeRobotDataset,
    ):
        dataset = TorqueWorldModelDataset(config)
    assert len(dataset) > 0
    assert len(dataset.episodes) == 2


def test_dataset_concatenates_multiple_lerobot_v3_sources(monkeypatch):
    rows = 2
    block = 2

    def make_columns(offset):
        high_ts = torch.arange(rows * block, dtype=torch.int64) * 500_000_000
        anchor_ts = high_ts.reshape(rows, block)[:, -1]
        action_index = torch.arange(rows * block, dtype=torch.int64).reshape(
            rows, block, 1
        )
        action_anchor = high_ts.reshape(rows, block, 1)
        return {
            "observation.joint": _packed_values(rows, block, 2, offset),
            "observation.velocity": _packed_values(rows, block, 2, offset + 100),
            "observation.delta_q": _packed_values(rows, block, 2, offset + 200),
            "observation.torque": _packed_values(rows, block, 2, offset + 300),
            "action.joint": _packed_values(rows, block, 2, offset + 400),
            "timing.state_timestamp_ns": high_ts.reshape(rows, block, 1),
            "timing.anchor_timestamp_ns": anchor_ts.reshape(rows, 1),
            "timing.action_index": action_index,
            "timing.action_anchor_timestamp_ns": action_anchor,
        }

    columns_by_repo = {
        "insert_usb_lerobot_v3": make_columns(0),
        "push_button_lerobot_v3": make_columns(1000),
    }

    class MultiSourceHFDataset(_FakeHFDataset):
        pass

    class MultiSourceLeRobotDataset:
        def __init__(self, *, repo_id, root, **kwargs):
            del root, kwargs
            self.hf_dataset = MultiSourceHFDataset(columns_by_repo[repo_id])
            self.meta = SimpleNamespace(
                episodes=[
                    {
                        "episode_index": 7,
                        "dataset_from_index": 0,
                        "dataset_to_index": rows,
                    }
                ]
            )

    monkeypatch.setattr(
        "data_process.world_model_dataset._load_lerobot_dataset_class",
        lambda: MultiSourceLeRobotDataset,
    )
    config = {
        "wm_v3_only": True,
        "train_data": {
            "format": "lerobot_v3",
            "sources": [
                {
                    "repo_id": "insert_usb_lerobot_v3",
                    "root": "/tmp/insert_usb_lerobot_v3",
                },
                {
                    "repo_id": "push_button_lerobot_v3",
                    "root": "/tmp/push_button_lerobot_v3",
                },
            ],
        },
        "dataloader": {
            "backend": "lerobot",
            "high_fps": 2,
            "expert_fps": 1,
            "state_history_horizon": 1,
            "prediction_horizon": 1,
            "action_condition_horizon": 1,
            "action_start_offset": 1,
            "high_timestamp_key": "timing.state_timestamp_ns",
            "anchor_timestamp_key": "timing.anchor_timestamp_ns",
            "action_index_key": "timing.action_index",
            "action_anchor_timestamp_key": "timing.action_anchor_timestamp_ns",
            "normalize_mode": None,
            "pad_history": True,
            "pad_future": False,
        },
        "contact_gate": {"enabled": False},
    }

    dataset = TorqueWorldModelDataset(config)

    assert len(dataset.source_datasets) == 2
    assert dataset.high_tensors["q"].shape[0] == rows * block * 2
    assert len(dataset.episodes) == 2
    assert [episode["episode_index"] for episode in dataset.episodes] == [0, 1]
    assert [episode["source_name"] for episode in dataset.episodes] == [
        "insert_usb_lerobot_v3",
        "push_button_lerobot_v3",
    ]
    assert dataset.episodes[1]["dataset_from_index"] == rows * block
    assert dataset.episodes[1]["source_dataset_from_index"] == rows
    assert len(dataset) == 4
    first = dataset[0]
    second = dataset[2]
    assert first["q"].flatten()[0].item() == 1.0
    assert second["q"].flatten()[0].item() == 1001.0


def test_normalizer_streams_target_pose_and_uses_training_frames():
    dataset = _make_dataset(
        normalize_mode="gaussian",
        compute_normalizer=True,
    )
    assert set(dataset.normalizer.stats) == {
        "q",
        "tau",
        "dq",
        "ddq",
        "wrench",
        "target_relative_pose",
    }
    sample = dataset[0]
    assert torch.isfinite(sample["q"]).all()
    assert torch.isfinite(sample["target_relative_pose"]).all()


def test_pose_action_resampling_interpolates_between_high_rate_ticks():
    dataset = _make_dataset()
    dataset.action_resample = "pose"
    episode = dataset.episodes[0]

    pose = dataset._sample_reference_poses(
        torch.tensor([875_000_000], dtype=torch.int64),
        episode,
    )

    torch.testing.assert_close(pose[0, :3], torch.full((3,), 3.5))
    torch.testing.assert_close(
        pose[0, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0])
    )
