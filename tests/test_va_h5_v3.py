from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from data_process.tool import VA_h5_v3 as va_converter
from data_process.tool.VA_h5_v3 import (
    VAH5Dataset,
    build_conversion_spec,
    homogeneous_pose_to_xyz_quat_xyzw,
)


def _shape_meta() -> dict:
    return {
        "fps": 25,
        "master_timestamp_path": "cameras/wrist/timestamp_us",
        "master_timeline": {
            "max_gap_s": 0.0001,
            "store_timestamps": True,
        },
        "features": {
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [1, 1, 3],
                "h5_path": "cameras/wrist/frames",
                "align": "index",
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": [1, 1, 3],
                "h5_path": "cameras/side/frames",
                "align": "index",
            },
            "observation.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
            },
            "action.joint": {
                "dtype": "float32",
                "shape": [1],
                "h5_path": "teleop/q",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
            },
            "action.ee_pose": {
                "dtype": "float32",
                "shape": [7],
                "h5_path": "teleop/ee_pose",
                "timestamp_path": "teleop/timestamp_us",
                "align": "previous",
                "transform": "ee_pose_matrix_to_quaternion",
            },
        },
    }


def _write_episode(
    path: Path,
    *,
    side_rows: int = 3,
    master_timestamps: tuple[int, ...] = (100, 200, 300),
) -> None:
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset(
            "cameras/wrist/timestamp_us", data=np.asarray(master_timestamps)
        )
        h5_file.create_dataset(
            "cameras/wrist/frames",
            data=(
                np.asarray([1, 2, 3], dtype=np.uint8)
                .reshape(3, 1, 1, 1)
                .repeat(3, axis=-1)
            ),
        )
        h5_file.create_dataset(
            "cameras/side/frames",
            data=(
                np.arange(10, 10 + side_rows, dtype=np.uint8)
                .reshape(side_rows, 1, 1, 1)
                .repeat(3, axis=-1)
            ),
        )
        h5_file.create_dataset(
            "teleop/timestamp_us", data=np.asarray([50, 120, 190, 250, 310])
        )
        h5_file.create_dataset(
            "teleop/q",
            data=np.asarray([[0], [1], [2], [3], [4]], dtype=np.float32),
        )
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
        poses[:, 0, 3] = np.arange(5, dtype=np.float32)
        h5_file.create_dataset("teleop/ee_pose", data=poses)


def test_homogeneous_pose_conversion_is_xyz_quaternion_xyzw() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    pose[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = homogeneous_pose_to_xyz_quat_xyzw(pose, np)

    assert result.shape == (7,)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        result[3:], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1.0e-6
    )


def test_homogeneous_pose_conversion_rejects_nonstandard_bottom_row() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[3, 0] = 1.0

    with pytest.raises(ValueError, match="bottom row"):
        homogeneous_pose_to_xyz_quat_xyzw(pose, np)


def test_camera_rows_are_indexed_and_numeric_values_are_causal(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path)
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )
        frames = [
            dataset.read_frame(
                h5_file,
                index,
                spec["mappings"],
                h5_path,
                spec["master_timestamp_path"],
                cache,
            )
            for index in range(3)
        ]

    assert [
        int(frame["observation.images.wrist"][0, 0, 0]) for frame in frames
    ] == [1, 2, 3]
    assert [
        int(frame["observation.images.side"][0, 0, 0]) for frame in frames
    ] == [10, 11, 12]
    np.testing.assert_array_equal(
        [frame["observation.joint"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        [frame["action.joint"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        [frame["action.ee_pose"][0] for frame in frames], [0, 2, 3]
    )
    np.testing.assert_array_equal(
        frames[0]["action.ee_pose"][3:], [0, 0, 0, 1]
    )
    np.testing.assert_array_equal(
        [frame["timing.master_timestamp_ns"][0] for frame in frames],
        [100_000, 200_000, 300_000],
    )


def test_action_chunk_shape_is_rejected() -> None:
    config = _shape_meta()
    config["features"]["action.joint"]["shape"] = [8, 1]

    with pytest.raises(ValueError, match=r"one \[Da\] vector"):
        build_conversion_spec(config)


def test_unmatched_terminal_camera_row_is_truncated_by_index(tmp_path: Path) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path, side_rows=2)
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )

    np.testing.assert_array_equal(cache["selected_master_indices"], [0, 1])
    assert cache["dropped_media_tail_rows"] == 1


def test_leading_camera_row_without_history_is_dropped_without_index_shift(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "episode.h5"
    _write_episode(h5_path, master_timestamps=(10, 100, 200))
    spec = build_conversion_spec(_shape_meta())
    dataset = VAH5Dataset(tmp_path, h5py=h5py, np=np)

    with dataset.open_episode(h5_path) as h5_file:
        cache = dataset.build_episode_cache(
            h5_file,
            spec["mappings"],
            spec["master_timestamp_path"],
            spec["fps"],
            h5_path,
            master_timeline=spec["master_timeline"],
        )
        first = dataset.read_frame(
            h5_file,
            0,
            spec["mappings"],
            h5_path,
            spec["master_timestamp_path"],
            cache,
        )

    np.testing.assert_array_equal(cache["selected_master_indices"], [1, 2])
    assert int(first["observation.images.wrist"][0, 0, 0]) == 2
    assert int(first["observation.images.side"][0, 0, 0]) == 11
    assert float(first["action.joint"][0]) == 0.0


@pytest.fixture
def conversion_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_path, output_path = tmp_path / "input", tmp_path / "output"
    input_path.mkdir()
    _write_episode(input_path / "episode.h5")
    config = _shape_meta()
    config["io"] = {"input": str(input_path), "output": str(output_path)}
    frames, created_roots = [], []

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            root = kwargs["root"]
            root.mkdir(parents=True)
            created_roots.append(root)
            return cls()

        def add_frame(self, frame, task):
            frames.append(frame)

        def save_episode(self, task):
            pass

        def finalize(self):
            pass

    monkeypatch.setattr(va_converter, "load_shape_meta", lambda _: config)
    monkeypatch.setattr(va_converter, "load_h5py", lambda: h5py)
    monkeypatch.setattr(
        va_converter, "load_conversion_deps", lambda: (h5py, np, FakeLeRobotDataset)
    )
    args = argparse.Namespace(
        config=tmp_path / "config.yaml", input=None, output=None, overwrite=False
    )
    return args, input_path, output_path, frames, created_roots


@pytest.mark.parametrize("flag", [[], ["--overwrite"]])
def test_overwrite_cli_flag(flag: list[str], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["VA_h5_v3.py", "-c", "config.yaml", *flag])
    assert va_converter.parse_args().overwrite is bool(flag)


@pytest.mark.parametrize("output_kind", ["directory", "file"])
def test_overwrite_replaces_existing_output(conversion_setup, output_kind: str):
    args, _, output_path, frames, created_roots = conversion_setup
    if output_kind == "directory":
        (output_path / "nested").mkdir(parents=True)
        (output_path / "nested" / "old.txt").write_text("old")
    else:
        output_path.write_text("old")
    args.overwrite = True

    va_converter.run_conversion(args)

    assert created_roots == [output_path]
    assert output_path.is_dir() and not (output_path / "nested").exists()
    assert len(frames) == 3


def test_existing_output_is_preserved_without_overwrite(conversion_setup):
    args, _, output_path, _, created_roots = conversion_setup
    output_path.mkdir()
    marker = output_path / "old.txt"
    marker.write_text("keep")
    del args.overwrite  # Also preserve callers constructing older Namespaces.

    with pytest.raises(FileExistsError, match="--overwrite"):
        va_converter.run_conversion(args)

    assert marker.read_text() == "keep"
    assert created_roots == []


def test_overwrite_uses_output_override(conversion_setup):
    args, _, output_path, frames, created_roots = conversion_setup
    output_path.write_text("keep configured output")
    args.output = output_path.parent / "override"
    args.output.mkdir()
    marker = args.output / "old.txt"
    marker.write_text("old")
    args.overwrite = True

    va_converter.run_conversion(args)

    assert output_path.read_text() == "keep configured output"
    assert not marker.exists() and created_roots == [args.output]
    assert len(frames) == 3


def test_overwrite_preserves_output_when_input_is_missing(conversion_setup):
    args, input_path, output_path, _, created_roots = conversion_setup
    (input_path / "episode.h5").unlink()
    output_path.write_text("keep")
    args.overwrite = True

    with pytest.raises(FileNotFoundError):
        va_converter.run_conversion(args)

    assert output_path.read_text() == "keep"
    assert created_roots == []


@pytest.mark.parametrize("ancestor", [False, True])
def test_overwrite_refuses_to_delete_input(conversion_setup, ancestor: bool):
    args, input_path, _, _, created_roots = conversion_setup
    args.output = input_path.parent if ancestor else input_path
    args.overwrite = True

    with pytest.raises(ValueError):
        va_converter.run_conversion(args)

    assert (input_path / "episode.h5").is_file()
    assert created_roots == []


def test_inspect_only_ignores_overwrite(conversion_setup, monkeypatch):
    args, _, output_path, _, created_roots = conversion_setup
    output_path.write_text("keep")
    monkeypatch.setattr(
        sys, "argv", ["VA_h5_v3.py", "-c", str(args.config), "--inspect-only", "--overwrite"]
    )

    va_converter.main()

    assert output_path.read_text() == "keep"
    assert created_roots == []
