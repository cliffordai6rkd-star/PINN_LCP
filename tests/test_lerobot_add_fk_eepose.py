import json

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from data_process.tool.lerobot_add_fk_eepose import add_fk_eepose, compute_eeposes


@pytest.fixture
def urdf(tmp_path):
    parts = ['<robot name="test_xarm"><link name="link_base"/>']
    parent = "link_base"
    for i in range(1, 8):
        child = f"link{i}"
        parts.append(
            f'<link name="{child}"/><joint name="joint{i}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0.1 0 0.01" rpy="0 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-6.3" upper="6.3" effort="10" velocity="10"/></joint>'
        )
        parent = child
    parts.append(
        '<link name="link_eef"/><joint name="eef" type="fixed">'
        '<parent link="link7"/><child link="link_eef"/>'
        '<origin xyz="0.2 0 0.3" rpy="0 0 0"/></joint></robot>'
    )
    path = tmp_path / "xarm7.urdf"
    path.write_text("".join(parts))
    return path


def expected(angles):
    angles = np.asarray(angles)
    poses = np.zeros((len(angles), 7))
    poses[:, :3] = np.column_stack(
        (0.1 + 0.8 * np.cos(angles), 0.8 * np.sin(angles), np.full(len(angles), 0.37))
    )
    poses[:, 5] = np.sin(angles / 2)
    poses[:, 6] = np.cos(angles / 2)
    poses[poses[:, 6] < 0, 3:] *= -1
    return poses


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    features = {"observation.joint": {"dtype": "float32", "shape": [7]}}
    features.update({k: {"dtype": "int64", "shape": [1]} for k in ("index", "episode_index")})
    info = {"codebase_version": "v3.0", "total_frames": 6, "total_episodes": 2, "features": features}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps({"observation.joint": {"count": [6]}}))
    angles = [0, np.pi / 2, 3 * np.pi / 2, -np.pi / 2, 0.3, -0.3]
    hf_features = {
        "observation.joint": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "List"},
        "index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
    }
    metadata = {b"huggingface": json.dumps({"info": {"features": hf_features}, "fingerprint": "old"}).encode()}
    for episode in range(2):
        joints = np.zeros((3, 7), dtype=np.float32)
        joints[:, 0] = angles[episode * 3 : episode * 3 + 3]
        table = pa.table({"observation.joint": pa.array(joints.tolist(), type=pa.list_(pa.float32(), 7)),
                          "index": list(range(episode * 3, episode * 3 + 3)), "episode_index": [episode] * 3})
        pq.write_table(table.replace_schema_metadata(metadata), root / f"data/chunk-000/file-{episode:03d}.parquet")
    episodes = [{"episode_index": e, "length": 3, "dataset_from_index": e * 3,
                 "dataset_to_index": e * 3 + 3} for e in range(2)]
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    return root


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() if p.is_file() else None for p in root.rglob("*")}


def test_fk_matches_analytic_xyz_xyzw(urdf):
    joints = np.zeros((3, 7))
    joints[:, 0] = [0, np.pi / 2, 3 * np.pi / 2]
    poses = compute_eeposes(joints, urdf)
    assert poses.dtype == np.float32 and poses.shape == (3, 7)
    np.testing.assert_allclose(poses, expected(joints[:, 0]), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-6)
    assert np.all(poses[:, 6] >= 0)


def test_add_feature_updates_data_and_all_statistics(dataset, urdf):
    files = sorted((dataset / "data").rglob("*.parquet"))
    originals = [pq.read_table(path) for path in files]
    episode_path = dataset / "meta/episodes/chunk-000/file-000.parquet"
    old_episodes = pq.read_table(episode_path)
    summary = add_fk_eepose(dataset, urdf, overwrite=True)
    assert summary["frames"] == 6 and summary["episodes"] == 2
    assert not (dataset / ".backups").exists()
    poses = []
    for path, original in zip(files, originals):
        table = pq.read_table(path)
        assert table.select(original.column_names).equals(original, check_metadata=False)
        assert table.schema.field("action.eepose").type == pa.list_(pa.float32(), 7)
        hf = json.loads(table.schema.metadata[b"huggingface"])
        assert hf["info"]["features"]["action.eepose"]["length"] == 7
        assert hf.get("fingerprint") != "old"
        values = np.asarray(table["action.eepose"].to_pylist())
        np.testing.assert_allclose(values, expected(np.asarray(original["observation.joint"].to_pylist())[:, 0]), atol=1e-6)
        poses.extend(values)
    spec = json.loads((dataset / "meta/info.json").read_text())["features"]["action.eepose"]
    assert spec["dtype"] == "float32" and spec["shape"] == [7]
    stats = json.loads((dataset / "meta/stats.json").read_text())
    assert stats["observation.joint"] == {"count": [6]}
    assert stats["action.eepose"]["count"] == [6]
    for name, operation in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
        np.testing.assert_allclose(stats["action.eepose"][name], operation(poses, axis=0), atol=1e-6)
    episodes = pq.read_table(episode_path)
    assert episodes.select(old_episodes.column_names).equals(old_episodes)
    for episode, row in enumerate(episodes.to_pylist()):
        assert row["stats/action.eepose/count"] == [3]
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            assert len(row[f"stats/action.eepose/{name}"]) == 7
        np.testing.assert_allclose(row["stats/action.eepose/mean"], np.mean(poses[episode * 3 : episode * 3 + 3], axis=0), atol=1e-6)


def test_existing_pose_requires_overwrite(dataset, urdf):
    add_fk_eepose(dataset, urdf)
    before = snapshot(dataset)
    with pytest.raises(FileExistsError):
        add_fk_eepose(dataset, urdf)
    assert snapshot(dataset) == before


@pytest.mark.parametrize("invalid", ["nonfinite", "missing", "dimension"])
def test_invalid_joints_leave_dataset_unchanged(dataset, urdf, invalid):
    path = dataset / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path)
    if invalid == "missing":
        table = table.drop(["observation.joint"])
    else:
        joints = np.asarray(table["observation.joint"].to_pylist())
        if invalid == "nonfinite":
            joints[0, 0] = np.nan
        else:
            joints = joints[:, :6]
        table = table.set_column(0, "observation.joint", pa.array(joints.tolist(), type=pa.list_(pa.float32(), joints.shape[1])))
    pq.write_table(table, path)
    before = snapshot(dataset)
    with pytest.raises((ValueError, KeyError)):
        add_fk_eepose(dataset, urdf)
    assert snapshot(dataset) == before


def test_dry_run_does_not_mutate_dataset(dataset, urdf):
    before = snapshot(dataset)
    summary = add_fk_eepose(dataset, urdf, dry_run=True)
    assert summary["frames"] == 6 and summary["episodes"] == 2
    assert snapshot(dataset) == before
