from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_process.tool.h5_2_lerobotev3 import (
    H5Dataset,
    LeRobotV3Dataset,
    build_conversion_spec,
)


def _dataset() -> H5Dataset:
    return H5Dataset(".", h5py=object(), np=np)


def _dual_rate_spec() -> dict:
    return build_conversion_spec(
        {
            "fps": 10,
            "master_timestamp_path": "teleop/timestamp_us",
            "timeline": {
                "mode": "dual_rate",
                "high_fps": 20,
                "low_fps": 10,
                "action_horizon": 3,
                "max_gap_s": 0.051,
                "store_timestamps": True,
            },
            "features": {
                "observation.images.wrist": {
                    "dtype": "video",
                    "shape": [2, 2, 3],
                    "h5_path": "camera/frames",
                    "timestamp_path": "camera/timestamp_us",
                    "grid": "low_anchor",
                    "resample": "previous",
                    "allow_stale": True,
                    "emit_age_key": "wrist",
                },
                "observation.joint": {
                    "dtype": "float32",
                    "shape": [2, 1],
                    "h5_path": "teleop/q",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "high_past",
                    "resample": "linear",
                },
                "action.joint": {
                    "dtype": "float32",
                    "shape": [3, 1],
                    "h5_path": "teleop/q",
                    "timestamp_path": "teleop/timestamp_us",
                    "grid": "low_future",
                    "resample": "linear",
                },
            },
        }
    )


def _dual_rate_cache() -> tuple[dict, dict]:
    spec = _dual_rate_spec()
    teleop_us = np.arange(1_000_000, 1_600_001, 50_000, dtype=np.int64)
    camera_us = np.asarray([1_050_000, 1_250_000, 1_450_000], dtype=np.int64)
    datasets = {
        "teleop/timestamp_us": teleop_us,
        "teleop/q": ((teleop_us - 1_000_000) * 1e-6)[:, None],
        "camera/timestamp_us": camera_us,
        "camera/frames": np.stack(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        ),
    }
    timestamps = {
        "teleop/timestamp_us": teleop_us,
        "camera/timestamp_us": camera_us,
    }
    cache = _dataset()._build_dual_rate_episode_cache(
        dataset_cache=datasets,
        timestamp_cache=timestamps,
        mappings=spec["mappings"],
        master_timestamp_path=spec["master_timestamp_path"],
        h5_path=Path("episode.h5"),
        timeline=spec["timeline"],
    )
    return spec, cache


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


def test_pchip_alignment_requires_a_bounded_gap() -> None:
    shape_meta = {
        "fps": 100,
        "master_timestamp_path": "teleop/timestamp_us",
        "features": {
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "pchip",
            }
        },
    }

    with pytest.raises(ValueError, match="needs max_gap_s"):
        build_conversion_spec(shape_meta)


def test_pchip_resampling_preserves_knots_and_fills_missing_grid_points() -> None:
    source_times = np.asarray([0.00, 0.01, 0.03, 0.04], dtype=np.float64)
    source_values = (2.0 * source_times + 1.0)[:, None]
    targets = np.arange(5, dtype=np.float64) * 0.01

    result = _dataset()._resample_values(
        source_values,
        source_times=source_times,
        targets=targets,
        method="pchip",
        transform=None,
    )

    np.testing.assert_allclose(result[:, 0], 2.0 * targets + 1.0)
    np.testing.assert_array_equal(result[[0, 1, 3, 4]], source_values)


def test_pchip_alignment_rejects_a_gap_above_the_configured_limit() -> None:
    with pytest.raises(ValueError, match="exceeds max_gap_s"):
        _dataset()._validate_resample_targets(
            source_times=np.asarray([0.0, 0.06]),
            targets=np.asarray([0.01]),
            method="pchip",
            max_gap_s=0.05,
            feature_name="observation.joint",
            h5_path=Path("episode.h5"),
        )


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


def test_conversion_spec_keeps_subtract_combination_out_of_metadata() -> None:
    spec = build_conversion_spec(
        {
            "fps": 100,
            "master_timestamp_path": "teleop/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [2],
                    "h5_paths": ["teleop/q_cmd", "teleop/q"],
                    "combine": "subtract",
                }
            },
        }
    )

    assert spec["mappings"][0]["combine"] == "subtract"
    assert "combine" not in spec["lerobot_features"]["observation.delta_q"]


def test_subtract_combination_computes_first_h5_path_minus_second() -> None:
    mapping = {
        "lerobot_key": "observation.delta_q",
        "h5_paths": ["teleop/q_cmd", "teleop/q"],
        "timestamp_path": "teleop/timestamp_us",
        "resample": "linear",
        "transform": None,
        "combine": "subtract",
        "feature_spec": {"dtype": "float32", "shape": (2,)},
    }
    cache = {
        "datasets": {
            "teleop/q_cmd": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "teleop/q": np.asarray([[0.5, 1.5], [2.0, 3.5]]),
        },
        "timestamp_seconds": {
            "teleop/timestamp_us": np.asarray([0.0, 1.0]),
        },
    }

    result = _dataset()._resample_mapping(
        mapping,
        targets=np.asarray([0.0, 0.5, 1.0]),
        cache=cache,
    )

    np.testing.assert_allclose(
        result,
        np.asarray([[0.5, 0.5], [0.75, 0.5], [1.0, 0.5]], dtype=np.float32),
    )


def _delta_q_source_spec() -> dict:
    return build_conversion_spec(
        {
            "fps": 100,
            "master_timestamp_path": "state/timestamp_us",
            "features": {
                "observation.delta_q": {
                    "dtype": "float32",
                    "shape": [1],
                    "sources": [
                        {
                            "h5_path": "control/q_cmd",
                            "timestamp_path": "control/timestamp_us",
                            "align": "previous",
                            "allow_stale": True,
                        },
                        {
                            "h5_path": "state/q_measured",
                            "timestamp_path": "state/timestamp_us",
                            "align": "pchip",
                            "max_gap_s": 0.05,
                        },
                    ],
                    "combine": "subtract",
                }
            },
        }
    )


def test_delta_q_resamples_command_zoh_and_state_continuously_before_subtracting():
    spec = _delta_q_source_spec()
    mapping = spec["mappings"][0]
    command_times = np.asarray([0.00, 0.03, 0.07])
    state_times = np.asarray([0.00, 0.02, 0.04, 0.06, 0.08])
    targets = np.arange(9, dtype=np.float64) * 0.01
    cache = {
        "datasets": {
            "control/q_cmd": np.asarray([[0.1], [0.2], [0.4]]),
            "state/q_measured": (0.05 + 0.5 * state_times)[:, None],
        },
        "timestamp_seconds": {
            "control/timestamp_us": command_times,
            "state/timestamp_us": state_times,
        },
    }

    result = _dataset()._resample_mapping(mapping, targets, cache)

    command_zoh = np.asarray(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4]
    )
    measured_continuous = 0.05 + 0.5 * targets
    np.testing.assert_allclose(
        result[:, 0],
        command_zoh - measured_continuous,
        atol=1.0e-7,
    )
    assert mapping["sources"][0]["method"] == "previous"
    assert mapping["sources"][1]["method"] == "pchip"
    assert "sources" not in spec["lerobot_features"]["observation.delta_q"]


def test_delta_q_zoh_rejects_target_before_first_command_without_future_leakage():
    mapping = _delta_q_source_spec()["mappings"][0]
    cache = {
        "datasets": {
            "control/q_cmd": np.asarray([[0.1], [0.2]]),
            "state/q_measured": np.asarray([[0.0], [0.1]]),
        },
        "timestamp_seconds": {
            "control/timestamp_us": np.asarray([0.01, 0.02]),
            "state/timestamp_us": np.asarray([0.00, 0.02]),
        },
    }

    with pytest.raises(ValueError, match="outside the source timestamp range"):
        _dataset()._resample_mapping(
            mapping,
            targets=np.asarray([0.00]),
            cache=cache,
        )


def test_dual_rate_spec_generates_timing_and_image_age_features() -> None:
    spec = _dual_rate_spec()

    assert spec["timeline"]["high_window_size"] == 2
    assert spec["lerobot_features"]["timing.high_timestamp_ns"]["shape"] == (2, 1)
    assert spec["lerobot_features"]["timing.action_timestamp_ns"]["shape"] == (3, 1)
    assert spec["lerobot_features"]["wrist"]["shape"] == (1,)


def test_dual_rate_grids_are_uniform_contiguous_and_interpolated() -> None:
    spec, cache = _dual_rate_cache()

    assert cache["target_timestamps"].shape == (4,)
    np.testing.assert_array_equal(
        np.diff(cache["anchor_timestamp_ns"]),
        np.full(3, 100_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.diff(cache["high_timestamp_ns"], axis=1),
        np.full((4, 1), 50_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        cache["high_timestamp_ns"][1:, 0] - cache["high_timestamp_ns"][:-1, -1],
        np.full(3, 50_000_000, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.diff(cache["action_timestamp_ns"], axis=1),
        np.full((4, 2), 100_000_000, dtype=np.int64),
    )

    first = _dataset()._read_dual_rate_frame(0, spec["mappings"], cache)
    np.testing.assert_allclose(first["observation.joint"][:, 0], [0.05, 0.10])
    np.testing.assert_allclose(first["action.joint"][:, 0], [0.10, 0.20, 0.30])


def test_dual_rate_images_repeat_latest_history_without_future_leakage() -> None:
    spec, cache = _dual_rate_cache()
    frames = [
        _dataset()._read_dual_rate_frame(index, spec["mappings"], cache)
        for index in range(4)
    ]

    assert [int(frame["observation.images.wrist"][0, 0, 0]) for frame in frames] == [
        10,
        10,
        20,
        20,
    ]
    np.testing.assert_allclose(
        [float(frame["wrist"][0]) for frame in frames],
        [0.05, 0.15, 0.05, 0.15],
        atol=1e-6,
    )
    image_cache = cache["resampled"]["observation.images.wrist"]
    source_s = cache["timestamp_seconds"]["camera/timestamp_us"]
    selected_s = source_s[image_cache["indices"]]
    assert np.all(selected_s <= cache["target_timestamps"])


def test_dual_rate_rejects_large_numeric_interpolation_gap() -> None:
    with pytest.raises(ValueError, match="exceeds max_gap_s"):
        _dataset()._validate_resample_targets(
            source_times=np.asarray([1.0, 1.2]),
            targets=np.asarray([1.1]),
            method="linear",
            max_gap_s=0.051,
            feature_name="observation.joint",
            h5_path=Path("episode.h5"),
        )


def test_pose_resampling_uses_slerp_and_preserves_unit_quaternion() -> None:
    poses = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )

    result = _dataset()._resample_values(
        poses,
        source_times=np.asarray([0.0, 2.0]),
        targets=np.asarray([1.0]),
        method="pose",
        transform=None,
    )

    np.testing.assert_allclose(result[0, :3], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(
        result[0, 3:], [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-7
    )
    np.testing.assert_allclose(np.linalg.norm(result[0, 3:]), 1.0, atol=1e-7)


def test_lerobot_adapter_finalizes_buffered_metadata() -> None:
    class FakeDataset:
        finalized = False

        @classmethod
        def create(cls, **_kwargs):
            return cls()

        def finalize(self):
            self.finalized = True

    adapter = LeRobotV3Dataset(
        FakeDataset,
        repo_id="local/test",
        root=Path("unused"),
        fps=10,
        features={},
        no_videos=True,
    )

    adapter.finalize()

    assert adapter.dataset.finalized is True


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
