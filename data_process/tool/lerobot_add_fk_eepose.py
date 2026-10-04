"""Add URDF forward-kinematics xyz + quaternion poses to a LeRobot v3 dataset.

Example (positions in metres, joints in radians, quaternion order xyzw):
    python data_process/tool/lerobot_add_fk_eepose.py \
        --input-root ../xarm_ws/runs/erase_board_lerobotv3 \
        --urdf ../xarm_ws/gello_teleop/models/xarm7_dynamics.urdf \
        --overwrite

Updates parquet columns, feature metadata, and global/per-episode statistics
in place. Files are staged before replacement; no backups are created.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_process.tool.VA_h5_v3 import homogeneous_pose_to_xyz_quat_xyzw
from data_process.tool.lerobot_recompute_acceleration import (
    commit_staged_files,
    feature_stats,
    load_json,
    scalar_column_to_numpy,
    vector_column_to_numpy,
    write_json_temp,
    write_parquet_temp,
)


def compute_eeposes(
    joints: Any,
    urdf_path: Path,
    *,
    base_frame: str = "link_base",
    frame_name: str = "link_eef",
) -> np.ndarray:
    import pinocchio as pin

    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 7 or not len(joints):
        raise ValueError(f"Joint values must have nonempty shape (N, 7), got {joints.shape}.")
    if not np.isfinite(joints).all():
        raise ValueError("Joint values contain non-finite values.")
    if not Path(urdf_path).is_file():
        raise FileNotFoundError(f"Missing URDF: {urdf_path}")

    model = pin.buildModelFromUrdf(str(urdf_path))
    joint_names = [f"joint{i}" for i in range(1, 8)]
    if model.nq != 7 or model.nv != 7 or set(model.names[1:]) != set(joint_names):
        raise ValueError("URDF must contain exactly seven scalar joints named joint1..joint7.")
    for name in joint_names:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"URDF joint {name!r} must have one configuration coordinate.")
    for name in (base_frame, frame_name):
        if not model.existFrame(name):
            raise ValueError(f"URDF does not contain frame {name!r}.")

    base_id, frame_id = model.getFrameId(base_frame), model.getFrameId(frame_name)
    joint_indexes = [model.joints[model.getJointId(name)].idx_q for name in joint_names]
    data = model.createData()
    configuration = np.zeros(model.nq)
    transforms = np.empty((len(joints), 4, 4), dtype=np.float64)
    for index, values in enumerate(joints):
        configuration[joint_indexes] = values
        pin.framesForwardKinematics(model, data, configuration)
        transforms[index] = (data.oMf[base_id].inverse() * data.oMf[frame_id]).homogeneous
    return homogeneous_pose_to_xyz_quat_xyzw(transforms, np)


def append_pose_column(table: pa.Table, key: str, values: np.ndarray) -> pa.Table:
    array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), 7)
    if key in table.column_names:
        table = table.set_column(table.column_names.index(key), key, array)
    else:
        table = table.append_column(key, array)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        payload = json.loads(metadata[b"huggingface"].decode("utf-8"))
        payload["info"]["features"][key] = {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": 7,
            "_type": "List",
        }
        payload.pop("fingerprint", None)
        metadata[b"huggingface"] = json.dumps(payload).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
    return table


def add_fk_eepose(
    root: Path,
    urdf_path: Path,
    *,
    joint_key: str = "observation.joint",
    pose_key: str = "action.eepose",
    base_frame: str = "link_base",
    frame_name: str = "link_eef",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    info_path, stats_path = root / "meta/info.json", root / "meta/stats.json"
    info, stats = load_json(info_path), load_json(stats_path)
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("Expected a LeRobot v3 dataset.")
    if joint_key not in info["features"]:
        raise KeyError(f"Missing source feature {joint_key!r}.")
    if tuple(info["features"][joint_key].get("shape", ())) != (7,):
        raise ValueError(f"{joint_key} must have shape [7].")
    if pose_key == joint_key:
        raise ValueError("The pose key must differ from the source joint key.")
    if not overwrite and (pose_key in info["features"] or pose_key in stats):
        raise FileExistsError(f"Feature {pose_key!r} already exists. Use --overwrite.")

    data_paths = sorted((root / "data").rglob("*.parquet"))
    episode_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not data_paths or not episode_paths:
        raise FileNotFoundError("Dataset must contain data and episode metadata parquet files.")
    data_tables, episode_tables = [], []
    joint_chunks, episode_chunks = [], []
    row_groups = {}
    for path in data_paths + episode_paths:
        parquet = pq.ParquetFile(path)
        row_groups[path] = tuple(
            parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)
        )
        table = parquet.read()
        if path in data_paths:
            if not overwrite and pose_key in table.column_names:
                raise FileExistsError(f"Feature {pose_key!r} already exists in {path}. Use --overwrite.")
            data_tables.append((path, table))
            joint_chunks.append(vector_column_to_numpy(table, joint_key))
            episode_chunks.append(scalar_column_to_numpy(table, "episode_index"))
        else:
            episode_tables.append((path, table))
    joints = np.concatenate(joint_chunks)
    episode_indexes = np.concatenate(episode_chunks)
    if len(joints) != info["total_frames"]:
        raise ValueError("Data row count does not match meta/info.json total_frames.")
    poses = compute_eeposes(joints, urdf_path, base_frame=base_frame, frame_name=frame_name)
    episode_stats = {
        int(index): feature_stats(poses[episode_indexes == index])
        for index in np.unique(episode_indexes)
    }
    seen_episodes = set()
    updated_tables = []
    offset = 0
    for path, table in data_tables:
        updated_tables.append((path, append_pose_column(table, pose_key, poses[offset:offset + table.num_rows])))
        offset += table.num_rows
    for path, table in episode_tables:
        records = table.select(["episode_index", "length"]).to_pylist()
        for episode in records:
            index = int(episode["episode_index"])
            if index in seen_episodes or index not in episode_stats:
                raise ValueError(f"Duplicate or missing data for episode {index}.")
            if episode_stats[index]["count"] != [int(episode["length"])]:
                raise ValueError(f"Frame count does not match metadata for episode {index}.")
            seen_episodes.add(index)
        for name in feature_stats(poses):
            key = f"stats/{pose_key}/{name}"
            value_type = pa.int64() if name == "count" else pa.float64()
            array = pa.array(
                [episode_stats[int(episode["episode_index"])][name] for episode in records],
                type=pa.list_(value_type),
            )
            if key in table.column_names:
                table = table.set_column(table.column_names.index(key), key, array)
            else:
                table = table.append_column(key, array)
        updated_tables.append((path, table))
    if seen_episodes != set(episode_stats) or len(seen_episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata does not match dataset episodes.")
    info["features"][pose_key] = {
        "dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]
    }
    stats[pose_key] = feature_stats(poses)
    summary = {"frames": len(poses), "episodes": len(seen_episodes)}
    if dry_run:
        return summary

    staged = []
    try:
        for path, table in updated_tables:
            staged.append((write_parquet_temp(table, target=path, row_group_sizes=row_groups[path]), path))
        staged.append((write_json_temp(info_path, info), info_path))
        staged.append((write_json_temp(stats_path, stats), stats_path))
        commit_staged_files(staged)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--joint-key", default="observation.joint")
    parser.add_argument("--pose-key", default="action.eepose")
    parser.add_argument("--base-frame", default="link_base")
    parser.add_argument("--frame-name", default="link_eef")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = add_fk_eepose(
        args.input_root, args.urdf, joint_key=args.joint_key, pose_key=args.pose_key,
        base_frame=args.base_frame, frame_name=args.frame_name,
        overwrite=args.overwrite, dry_run=args.dry_run,
    )
    print(f"{'Computed' if args.dry_run else 'Updated'} {args.pose_key}: {summary}")
    print(f"Frame: {args.base_frame} -> {args.frame_name}; format: [x,y,z,qx,qy,qz,qw]; units: m")


if __name__ == "__main__":
    main()
