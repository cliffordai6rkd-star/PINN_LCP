from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_process.tool.h5_2_lerobotev3 import H5Dataset, build_conversion_spec


def _dataset() -> H5Dataset:
    return H5Dataset(".", h5py=object(), np=np)


def test_conversion_spec_requires_explicit_fps() -> None:
    with pytest.raises(ValueError, match="define a positive integer 'fps'"):
        build_conversion_spec(
            {
                "master_timestamp_path": "teleop/timestamp_us",
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [1],
                        "h5_path": "teleop/state",
                    }
                },
            }
        )


def test_uniform_timestamps_use_configured_fps_and_master_bounds() -> None:
    master = np.asarray([1_000_000, 1_071_000, 1_129_000, 1_210_000], dtype=np.int64)

    result = _dataset().uniform_timestamps(master, "camera/timestamp_us", fps=10)

    np.testing.assert_array_equal(
        result,
        np.asarray([1_000_000, 1_100_000, 1_200_000], dtype=np.int64),
    )


def test_nearest_alignment_selects_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)
    dataset = _dataset()

    assert dataset._history_index_from_timestamps(timestamps, 200) == 1
    assert dataset._history_index_from_timestamps(timestamps, 299) == 1


def test_nearest_alignment_rejects_target_before_first_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)

    with pytest.raises(ValueError, match="No historical sample"):
        _dataset()._history_index_from_timestamps(timestamps, 99)


def test_past_window_ends_at_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300], dtype=np.int64)
    dataset = _dataset()

    np.testing.assert_array_equal(
        dataset._nearest_past_window_indices(timestamps, 299, window_size=3),
        np.asarray([0, 0, 1]),
    )
    np.testing.assert_array_equal(
        dataset._nearest_past_window_indices(timestamps, 350, window_size=3),
        np.asarray([0, 1, 2]),
    )


def test_future_window_starts_at_latest_historical_sample() -> None:
    timestamps = np.asarray([100, 200, 300, 400], dtype=np.int64)
    dataset = _dataset()

    np.testing.assert_array_equal(
        dataset._nearest_future_window_indices(timestamps, 299, window_size=3),
        np.asarray([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        dataset._nearest_future_window_indices(timestamps, 350, window_size=3),
        np.asarray([2, 3, 3]),
    )


def test_ee_pose_transform_converts_single_matrix_to_quat7() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    result = _dataset()._ee_pose_matrix_to_quaternion(pose)

    assert result.shape == (7,)
    np.testing.assert_allclose(result, [1, 2, 3, 0, 0, 0, 1], atol=1e-6)


def test_ee_pose_transform_converts_pose_window_to_quat7_window() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 8, axis=0)
    poses[:, 0, 3] = np.arange(8, dtype=np.float32)

    result = _dataset()._ee_pose_matrix_to_quaternion(poses)

    assert result.shape == (8, 7)
    np.testing.assert_allclose(result[:, 0], np.arange(8), atol=1e-6)
    np.testing.assert_allclose(result[:, 3:], np.asarray([[0, 0, 0, 1]] * 8), atol=1e-6)


def test_conversion_spec_keeps_transform_out_of_lerobot_metadata() -> None:
    spec = build_conversion_spec(
        {
            "fps": 10,
            "master_timestamp_path": "camera/timestamp_us",
            "features": {
                "action.ee_pose": {
                    "dtype": "float32",
                    "shape": [8, 7],
                    "h5_path": "teleop/ee_pose_follower",
                    "timestamp_path": "teleop/timestamp_us",
                    "align": "nearest_future_window",
                    "window_size": 8,
                    "transform": "ee_pose_matrix_to_quaternion",
                }
            },
        }
    )

    assert spec["mappings"][0]["transform"] == "ee_pose_matrix_to_quaternion"
    assert "transform" not in spec["lerobot_features"]["action.ee_pose"]
    assert spec["lerobot_features"]["action.ee_pose"]["shape"] == (8, 7)


@pytest.mark.parametrize(
    ("align", "window_size", "expected_x"),
    [
        ("nearest", 1, np.asarray(1.0, dtype=np.float32)),
        ("nearest_future_window", 3, np.asarray([1.0, 2.0, 2.0], dtype=np.float32)),
    ],
)
def test_pose_transform_runs_after_configured_alignment(
    align: str,
    window_size: int,
    expected_x: np.ndarray,
) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0)
    poses[:, 0, 3] = np.arange(3, dtype=np.float32)
    cache = {
        "datasets": {
            "teleop/ee_pose": poses,
            "teleop/timestamp_us": np.asarray([100, 200, 300], dtype=np.int64),
        },
        "timestamps": {
            "teleop/timestamp_us": np.asarray([100, 200, 300], dtype=np.int64),
        },
    }
    mapping = {
        "h5_paths": ["teleop/ee_pose"],
        "timestamp_path": "teleop/timestamp_us",
        "align": align,
        "window_size": window_size,
        "transform": "ee_pose_matrix_to_quaternion",
        "feature_spec": {"dtype": "float32"},
    }

    result = _dataset()._read_mapped_value(
        h5_file=None,
        mapping=mapping,
        frame_idx=0,
        target_t=250,
        h5_path=Path("episode.h5"),
        cache=cache,
    )

    np.testing.assert_allclose(result[..., 0], expected_x, atol=1e-6)
    expected_quaternion = np.broadcast_to(
        np.asarray([0, 0, 0, 1], dtype=np.float32),
        result[..., 3:].shape,
    )
    np.testing.assert_allclose(result[..., 3:], expected_quaternion, atol=1e-6)
