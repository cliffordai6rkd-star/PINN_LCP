from types import SimpleNamespace
from unittest.mock import patch

import torch

from data_process.world_model_dataset import TorqueWorldModelDataset


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
