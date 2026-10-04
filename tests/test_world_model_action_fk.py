"""FK action ingestion and independent 30 Hz action / 100 Hz state windows."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from data_process import contact_world_model_dataset as wm_data
from data_process.action_fk import normalize_action_fk, poses_from_joint_actions
from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from data_process.relative_time_grid import RelativeTimeGrid, GridMetadataError
from data_process.tool.h5_2_lerobotev3 import H5Dataset
from data_process.tool.h5_v3_wm import build_wm_conversion_spec, build_wm_episode_cache


@pytest.fixture
def fk_config(tmp_path):
    pytest.importorskip("pinocchio")
    parts = ['<robot name="arm"><link name="link_base"/>']
    parent = "link_base"
    for i in range(1, 8):
        child = f"link{i}"
        parts.append(
            f'<link name="{child}"/><joint name="joint{i}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<origin xyz="0.1 0 0"/><axis xyz="0 0 1"/>'
            '<limit lower="-6" upper="6" effort="10" velocity="10"/></joint>'
        )
        parent = child
    parts.append('<link name="link_eef"/><joint name="tool" type="fixed">'
                 '<parent link="link7"/><child link="link_eef"/>'
                 '<origin xyz="0.2 0 0"/></joint></robot>')
    path = tmp_path / "arm.urdf"
    path.write_text("".join(parts))
    return normalize_action_fk({"urdf_path": str(path), "joint_names": [f"joint{i}" for i in range(1, 8)]})


def analytic_pose(angle):
    angle = np.asarray(angle)
    result = np.zeros((*angle.shape, 7), dtype=np.float32)
    result[..., 0] = 0.1 + 0.8 * np.cos(angle)
    result[..., 1] = 0.8 * np.sin(angle)
    result[..., 5] = np.sin(angle / 2)
    result[..., 6] = np.cos(angle / 2)
    result[result[..., 6] < 0, 3:] *= -1
    return result


def test_fk_preserves_packed_holds_and_joint_order(fk_config):
    joints = np.zeros((2, 3, 7))
    joints[..., 0] = [[0, 0, np.pi / 2], [np.pi / 2, np.pi, 3 * np.pi / 2]]
    poses = poses_from_joint_actions(joints, fk_config)
    assert poses.dtype == np.float32
    np.testing.assert_allclose(poses, analytic_pose(joints[..., 0]), atol=1e-6)
    # Reordering the input and explicit joint names preserves the physical pose.
    reordered = {**fk_config, "joint_names": list(reversed(fk_config["joint_names"]))}
    np.testing.assert_allclose(poses_from_joint_actions(joints[..., ::-1], reordered), poses, atol=1e-6)


@pytest.fixture
def joint_source(monkeypatch, tmp_path, fk_config):
    n = 180
    # Two episodes reset their clocks. Epoch-scale clocks must retain ns precision.
    origin = 1_800_000_000_000_000_001
    times = torch.arange(n, dtype=torch.int64) * 10_000_000 + origin
    anchor_times = torch.arange(54, dtype=torch.int64) * 100_000_000 // 3 + origin
    indexes = torch.searchsorted(anchor_times, times, right=True) - 1
    joints = torch.zeros(n, 7)
    joints[:, 0] = indexes.float() * 0.025
    columns = {field: torch.ones(2 * n, 7) for field in wm_data.V3_STATE_FIELDS.values() if not field.startswith("action.")}
    columns.update({
        "action.joint": joints.repeat(2, 1),
        "timing.state_timestamp_ns": times.repeat(2),
        "timing.action_index": indexes.repeat(2),
        "timing.action_anchor_timestamp_ns": anchor_times[indexes].repeat(2),
    })
    # Real LeRobot tables expose a shape-[1] signal as a squeezed scalar column.
    columns["observation.tau_ext"] = torch.ones(2 * n)

    class Table:
        @property
        def column_names(self):
            return list(columns)

        def with_format(self, *args, **kwargs):
            selected = kwargs.get("columns")
            if selected is not None and set(selected) - set(columns):
                raise ValueError("Columns not in the dataset")
            return self

        def __getitem__(self, item):
            return columns

    source = SimpleNamespace(hf_dataset=Table(), meta=SimpleNamespace(episodes=[
        {"dataset_from_index": 0, "dataset_to_index": n},
        {"dataset_from_index": n, "dataset_to_index": 2 * n},
    ]))
    monkeypatch.setattr(wm_data, "_load_lerobot_dataset_class", lambda: lambda **kwargs: source)
    cfg = {"wm_v3_only": True, "dataloader": {
        "root": str(tmp_path), "repo_id": "test", "action_key": "action.ee_pose",
        "action_fk": fk_config, "high_fps": 100, "expert_fps": 30,
        "state_history_horizon": 50, "prediction_horizon": 40,
        "action_condition_horizon": 10, "action_start_offset": 1,
        "high_timestamp_key": "timing.state_timestamp_ns",
        "anchor_timestamp_key": "timing.action_anchor_timestamp_ns",
        "normalize_mode": "gaussian",
    }, "model": {"joint_dim": 7, "action_dim": 7}, "train": {"downsample": False}}
    return cfg, columns


@pytest.mark.parametrize("dataset_class", [wm_data.ContactWorldModelDataset, LatentContactWorldModelDataset])
def test_both_world_models_use_fk_actions_and_native_10_by_40_windows(joint_source, dataset_class):
    cfg, columns = joint_source
    dataset = dataset_class(cfg)
    assert dataset.high_tensors["tau_ext"].shape == (360, 1)
    expected = torch.from_numpy(analytic_pose(columns["action.joint"][:, 0].numpy()))
    torch.testing.assert_close(dataset.high_tensors["action"], expected)
    for episode in dataset.episodes:
        raw = int(episode["dataset_from_index"]) + 55
        sample = dataset[dataset.valid_indices.index(raw)]
        assert sample["action"].shape == (10, 7)
        assert sample["q_future"].shape == (40, 7)
        assert sample["q"].shape == (50, 7)
        assert (torch.diff(sample["action_chunk_index"]) == 1).all()
        torch.testing.assert_close(sample["action"], torch.from_numpy(
            analytic_pose(sample["action_chunk_index"].numpy() * 0.025)
        ))
        if isinstance(dataset, LatentContactWorldModelDataset):
            positions = sample["action_grid_positions"]
            assert positions.dtype == torch.int64
            assert set(torch.diff(positions).tolist()) == {3, 4}
            assert sample["future_grid_positions"].tolist() == list(range(1, 41))
    dataset.fit_normalizer(range(len(dataset)))
    assert torch.isfinite(dataset[0]["action"]).all()


def test_existing_pose_wins_and_no_urdf_is_needed(joint_source):
    cfg, columns = joint_source
    columns["action.ee_pose"] = torch.from_numpy(analytic_pose(np.full(360, 0.5)))
    cfg["dataloader"]["action_fk"]["urdf_path"] = "/missing/robot.urdf"
    dataset = wm_data.ContactWorldModelDataset(cfg)
    torch.testing.assert_close(dataset.high_tensors["action"], columns["action.ee_pose"])


@pytest.mark.parametrize("invalid", ["missing", "nan", "dimension", "frame"])
def test_invalid_fk_source_is_rejected(joint_source, invalid):
    cfg, columns = joint_source
    if invalid == "missing":
        columns.pop("action.joint")
    elif invalid == "nan":
        columns["action.joint"][10, 0] = float("nan")
    elif invalid == "dimension":
        columns["action.joint"] = columns["action.joint"][:, :6]
    else:
        cfg["dataloader"]["action_fk"]["frame_name"] = "missing_tool"
    with pytest.raises((ValueError, KeyError)):
        wm_data.ContactWorldModelDataset(cfg)


def test_missing_pose_without_fk_keeps_explicit_error(joint_source):
    cfg, _ = joint_source
    cfg["dataloader"].pop("action_fk")
    with pytest.raises(KeyError, match="action.ee_pose"):
        wm_data.ContactWorldModelDataset(cfg)


def test_delta_q_fallback_uses_joint_action_instead_of_pose(joint_source):
    cfg, columns = joint_source
    columns.pop("observation.delta_q")
    dataset = wm_data.ContactWorldModelDataset(cfg)
    torch.testing.assert_close(dataset.high_tensors["delta_q"], columns["action.joint"] - columns["observation.joint"])


@pytest.fixture
def conversion_case(fk_config):
    from test_h5_v3_wm import _config, _DatasetHandle, _H5Py

    cfg = _config()
    for feature in cfg["features"].values():
        feature["shape"] = [7]
    cfg["action_fk"] = fk_config
    state_ts = np.arange(13) * 10_000
    joints = np.zeros((13, 7), dtype=np.float32)
    joints[:, 0] = np.arange(13) / 10
    h5_file = {
        "state/timestamp_us": _DatasetHandle(state_ts),
        "camera/timestamp_us": _DatasetHandle([5_000, 45_000, 85_000]),
        "action/timestamp_us": _DatasetHandle(state_ts),
        "state/q": _DatasetHandle(np.ones_like(joints)),
        "action/q": _DatasetHandle(joints),
    }
    return cfg, H5Dataset(".", h5py=_H5Py, np=np), h5_file


def test_conversion_generates_pose_after_previous_action_selection(conversion_case):
    cfg, dataset, h5_file = conversion_case
    spec = build_wm_conversion_spec(cfg)
    assert spec["lerobot_features"]["action.ee_pose"]["shape"] == (7,)
    cache = build_wm_episode_cache(dataset, h5_file, spec, Path("test.h5"))
    held = cache["resampled"]["action.joint"]
    assert held[:, 0].tolist() == pytest.approx([0] * 4 + [0.4] * 4 + [0.8] * 4)
    np.testing.assert_allclose(cache["resampled"]["action.ee_pose"], analytic_pose(held[:, 0]), atol=1e-6)


def test_conversion_drops_nonfinite_episode_and_records_reason(conversion_case, monkeypatch, tmp_path):
    from data_process.tool import h5_v3_wm as converter
    from test_h5_v3_wm import _DatasetHandle, _H5Py

    cfg, dataset, good = conversion_case
    cfg.update(io={"input": ".", "output": str(tmp_path), "repo_id": "test"}, nonfinite_episode_policy="drop")
    bad = dict(good)
    invalid = np.ones((13, 7))
    invalid[5, 0] = np.nan
    bad["state/q"] = _DatasetHandle(invalid)
    monkeypatch.setattr(converter, "load_shape_meta", lambda path: cfg)
    monkeypatch.setattr(converter, "load_conversion_deps", lambda: (_H5Py, np, object))
    monkeypatch.setattr(converter, "H5Dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(dataset, "files", lambda: [Path("bad.h5"), Path("good.h5")])
    monkeypatch.setattr(dataset, "open_episode", lambda path: nullcontext(bad if path.name == "bad.h5" else good))
    saved, frames = [], []

    class Writer:
        def __init__(self, *args, **kwargs):
            assert kwargs["features"]["action.ee_pose"]["shape"] == (7,)

        def add_frame(self, frame, **kwargs):
            assert all(np.isfinite(value).all() for value in frame.values())
            frames.append(frame)

        def save_episode(self, **kwargs):
            saved.append(True)

        def finalize(self):
            pass

    monkeypatch.setattr(converter, "LeRobotV3Dataset", Writer)
    args = SimpleNamespace(config=tmp_path / "config.yaml", input=None, output=None)
    converter.run_conversion(args)
    assert len(saved) == 1 and len(frames) == 12
    import json
    manifest = json.loads((tmp_path / "meta/world_model_timeline.json").read_text())
    assert len(manifest["excluded_episodes"]) == 1
    assert manifest["excluded_episodes"][0]["path"] == "bad.h5"
    assert "non-finite" in manifest["excluded_episodes"][0]["reason"]
    cfg["nonfinite_episode_policy"] = "error"
    with pytest.raises(converter.NonFiniteFeatureError, match="non-finite"):
        converter.run_conversion(args)


def test_30hz_grid_checks_true_cadence_and_accumulated_drift():
    grid = RelativeTimeGrid(action_rate_hz=30)
    history = torch.arange(50, dtype=torch.int64)[None] * 10_000_000
    action = history[:, -1:] + torch.arange(1, 11, dtype=torch.int64)[None] * 100_000_000 // 3
    positions = grid.positions(history_ns=history, action_ns=action, future_horizon=40)
    assert positions["action_grid_positions"].tolist() == [[3, 6, 10, 13, 16, 20, 23, 26, 30, 33]]
    # Later timestamp jitter affects the audit, not the relative-step encoding.
    jittered = action.clone()
    jittered[:, 1::2] += 1_000_000
    torch.testing.assert_close(
        grid.positions(history_ns=history, action_ns=jittered, future_horizon=40)["action_grid_positions"],
        positions["action_grid_positions"],
    )
    # Slowly drifting intervals are individually legal but fail cumulative drift.
    action += torch.arange(10)[None] * 1_000_000
    with pytest.raises(GridMetadataError, match="cadence"):
        grid.positions(history_ns=history, action_ns=action, future_horizon=40)


def test_xarm_conversion_and_training_configs_agree():
    root = Path(__file__).resolve().parents[1]
    conversion = yaml.safe_load((root / "config/shape_meta/swm/xarm/erase_board.yaml").read_text())
    assert conversion["features"]["action.joint"]["h5_path"] == "teleop/right_q_xarm"
    assert conversion["features"]["observation.tau_ext"]["shape"] == [1]
    assert conversion["timeline"]["action_anchor_mode"] == "state_frames"
    assert conversion["timeline"]["action_fps"] == 25
    build_wm_conversion_spec(conversion)
    for path in ["pretrain/xarm/cwm_erase_board_100hz_40step.yaml", "latent_cwm_erase_board_100hz_40step.yaml"]:
        cfg = yaml.safe_load((root / "config/train_cfg" / path).read_text())
        assert cfg["dataloader"]["expert_fps"] == 25
        assert cfg["dataloader"]["action_condition_horizon"] == 10
        assert cfg["dataloader"]["prediction_horizon"] == 40
        assert cfg["dataloader"]["action_fk"] == conversion["action_fk"]
        assert cfg["train_data"]["sources"][0]["root"] == conversion["io"]["output"]
