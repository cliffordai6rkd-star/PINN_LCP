from pathlib import Path
from types import SimpleNamespace

import torch

from data_process.h5_direct_dataset import V3H5Collection
from data_process.world_model_dataset import TorqueWorldModelDataset


def _fake_source_collection(monkeypatch, root):
    class FakeDirect:
        def __init__(self, *, root, fields, **kwargs):
            del kwargs
            name = Path(root).name
            length = 6 if name == "insert_usb" else 5
            columns = {
                key: torch.full((length, 7), float(index))
                for index, key in enumerate(fields)
            }
            columns["__h5_timestamp_ns"] = torch.arange(length, dtype=torch.int64) * 10_000_000
            self.hf_dataset = SimpleNamespace(columns=columns)
            self.files = [Path(root) / "episode_0000.h5"]
            self.meta = SimpleNamespace(
                episodes=[
                    {
                        "episode_index": 0,
                        "dataset_from_index": 0,
                        "dataset_to_index": length,
                    }
                ]
            )

    def fake_discover(root, patterns, max_episodes=None):
        del patterns, max_episodes
        return [Path(root) / "episode_0000.h5"]

    monkeypatch.setattr("data_process.h5_direct_dataset.DirectH5EpisodeDataset", FakeDirect)
    monkeypatch.setattr("data_process.h5_direct_dataset.discover_h5_files", fake_discover)
    monkeypatch.setattr(
        "data_process.h5_direct_dataset.V3H5Collection._read_anchor_timestamps",
        staticmethod(
            lambda files, episodes, path, unit, timestamps: torch.arange(
                len(timestamps), dtype=torch.int64
            )
            * 40_000_000
        ),
    )


def test_v3_collection_concatenates_sources_and_offsets_episodes(monkeypatch):
    root = Path("/tmp/train-data-v3-test")
    _fake_source_collection(monkeypatch, root)
    collection = V3H5Collection(
        sources=[
            {"name": "insert_usb", "root": root / "insert_usb"},
            {"name": "push_button", "root": root / "push_button"},
        ],
        fields={"q": "observation.joint"},
        timestamp_path="teleop/timestamp_us",
        timestamp_unit="us",
        anchor_timestamp_path="cameras/wrist/timestamp_us",
        anchor_timestamp_unit="us",
        cache_mode="ram",
    )

    assert collection.columns["q"].shape[0] == 11
    assert collection.episodes[1]["dataset_from_index"] == 6
    assert collection.episodes[1]["source_name"] == "push_button"


def test_v3_collection_accepts_source_folder_names(monkeypatch):
    root = Path("/tmp/train-data-v3-test")
    _fake_source_collection(monkeypatch, root)
    collection = V3H5Collection(
        source_base_root=root,
        sources=["insert_usb", "push_button"],
        fields={"q": "observation.joint"},
        timestamp_path="teleop/timestamp_us",
        timestamp_unit="us",
        anchor_timestamp_path="cameras/wrist/timestamp_us",
        anchor_timestamp_unit="us",
        cache_mode="ram",
    )
    assert collection.episodes[0]["source_name"] == "insert_usb"
    assert collection.episodes[1]["source_name"] == "push_button"


def test_world_model_dataset_rejects_non_v3_train_data_contract():
    config = {
        "train_data": {
            "format": "legacy",
            "sources": [{"name": "insert_usb", "root": "/tmp/data"}],
        },
        "dataloader": {"root": "/tmp/data"},
    }
    try:
        TorqueWorldModelDataset(config)
    except ValueError as exc:
        assert "h5_v3" in str(exc)
    else:
        raise AssertionError("legacy train_data format should be rejected")


def test_lerobot_v3_format_implies_v3_only_backend_guard():
    config = {
        "train_data": {"format": "lerobot_v3"},
        "dataloader": {"backend": "h5", "root": "/tmp/data"},
    }
    try:
        TorqueWorldModelDataset(config)
    except ValueError as exc:
        assert "v3-only" in str(exc)
    else:
        raise AssertionError("lerobot_v3 format must reject the H5 backend")
