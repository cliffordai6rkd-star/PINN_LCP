"""Nero teacher scale/history, the one-second prefix and reversible replacement."""
from pathlib import Path

import numpy as np
import pytest
import torch

from test_contact_world_model_dataset import make_dataset
from model.nero_tau_free import NeroTorqueTeacher
from model.tau_other_sequence import build_tau_other_sequence_model
from model.pinn_model.contact_gate import ContactGateConfig, contact_phase_labels_from_signal
from scripts.prepare_nero_wm_data import replace_datasets
from data_process.contact_world_model_dataset import ContactWorldModelDataset


def teacher_checkpoint(path):
    config = {"dataloader": {"horizon": 25},
              "model": {"architecture": "lstm", "inputs": ["q", "dq", "delta_q"],
                        "input_dims": {"q": 7, "dq": 7, "delta_q": 7}, "output_dim": 7,
                        "hidden_dim": 8, "num_layers": 2, "dropout": 0., "head_hidden_dim": 8,
                        "target_key": "tau", "history_mode": "stateless_sliding_window"}}
    model = build_tau_other_sequence_model(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    normalizer = {"normalize_mode": "gaussian", "normalize_lowdim_keys": ["q", "dq", "delta_q", "tau"],
                  "eps": 1e-6, "stats": {key: {"mean": torch.full((7,), 2.), "std": torch.ones(7)}
                                         for key in ("q", "dq", "delta_q", "tau")}}
    torch.save({"config": config, "model": model.state_dict(), "normalizer": normalizer,
                "sample_rate_hz": 50., "dataloader_filters": {key: {"enabled": False, "operations": []}
                                                           for key in ("q", "dq", "delta_q", "tau")}}, path)


def test_teacher_uses_original_stride_complete_histories_and_physical_nm(tmp_path):
    path = tmp_path/"teacher.pt"
    teacher_checkpoint(path)
    teacher = NeroTorqueTeacher(path)
    n = 150
    timestamps = np.arange(n, dtype=np.int64)*10_000_000+1_800_000_000_000_000_001
    q = np.zeros((n, 7), dtype=np.float32)
    tau = np.full_like(q, 5.)
    seen = []
    hook = teacher.model.recurrent.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    prediction = teacher.predict(timestamps, q, q+1, q+3, tau)
    hook.remove()
    assert not prediction["valid_context"][:48].any() and prediction["valid_context"][48:].all()
    np.testing.assert_array_equal(prediction["tau_free"][48:], np.full((n-48, 7), 2., np.float32))
    np.testing.assert_array_equal(prediction["tau_ext"][48:], np.full((n-48, 7), 3., np.float32))
    assert seen[0].shape[1:] == (25, 21)
    torch.testing.assert_close(seen[0][..., 14:], torch.full_like(seen[0][..., 14:], 1/(1+1e-6)))
    assert teacher.contract["snapshot"] == "model"
    assert all(not p.requires_grad and p.grad is None for p in teacher.model.parameters())


def test_teacher_restarts_warmup_at_gaps(tmp_path):
    path = tmp_path/"teacher.pt"
    teacher_checkpoint(path)
    teacher = NeroTorqueTeacher(path)
    timestamps = np.arange(200, dtype=np.int64)*10_000_000
    timestamps[100:] += 100_000_000
    q = np.zeros((200, 7), dtype=np.float32)
    prediction = teacher.predict(timestamps, q, q, q, q)
    assert prediction["valid_context"][48:100].all()
    assert not prediction["valid_context"][100:148].any()
    assert prediction["valid_context"][148:].all()


def test_alignment_is_real_one_second_before_contact_not_after():
    # 10 ms rows, with one 2 ms jitter. Use actual times rather than 100 row counts.
    times = torch.arange(350, dtype=torch.float64)*.01
    times[80:] += .002
    signal = torch.zeros(350)
    signal[220:250] = 4.
    gate = ContactGateConfig(enabled=True, label_mode="three_phase", metric="tau_ext_l1",
                             contact_threshold=3., precontact_duration_s=1.)
    labels = contact_phase_labels_from_signal(signal, [(0, 350)], gate, timestamps_s=times)[:, 0]
    expected_alignment = (times >= times[220]-1.-1e-9) & (times < times[220])
    torch.testing.assert_close(labels == 1, expected_alignment)
    assert (labels[220:250] == 2).all() and (labels[250:] == 0).all()


def test_stored_teacher_validity_excludes_unknown_future_and_free_history(make_dataset):
    _, cfg = make_dataset(offset=1, future=4)
    from data_process import contact_world_model_dataset as dataset_module
    source = dataset_module._load_lerobot_dataset_class()()
    columns = source.hf_dataset[:]
    n = len(columns["observation.joint"])
    columns["observation.tau_ext"] = torch.zeros(n, 2)
    columns["observation.tau_label_valid"] = torch.ones(n, 1)
    columns["observation.tau_label_valid"][:48] = 0
    cfg["dataloader"]["tau_label_valid_key"] = "observation.tau_label_valid"
    cfg["contact_gate"] = {"enabled": True, "label_mode": "three_phase", "metric": "tau_ext_l1", "contact_threshold": 3.}
    dataset = ContactWorldModelDataset(cfg)
    assert (dataset.contact[:48] == -1).all()
    assert min(dataset.valid_indices) == 47  # First window has four fully valid future rows.
    sample = dataset[0]
    assert not sample["history_valid_mask"].all()
    assert "tau_label_valid_future" not in sample


def test_replacement_backups_and_rolls_back_on_failure(tmp_path, monkeypatch):
    stage, output, backup = (tmp_path/name for name in ("stage", "out", "backup"))
    for root, text in ((stage, "new"), (output, "old")):
        for name in ("a", "b"):
            (root/name).mkdir(parents=True)
            (root/name/"content").write_text(text)
    original = Path.rename
    def fail_second(self, destination):
        if self == stage/"b":
            raise OSError("simulated rename failure")
        return original(self, destination)
    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="simulated"):
        replace_datasets(stage, output, backup, ["a", "b"])
    assert all((output/name/"content").read_text() == "old" for name in ("a", "b"))
    assert all((stage/name/"content").read_text() == "new" for name in ("a", "b"))


def test_relabel_contact_threshold_updates_only_phase_and_its_statistics(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.utils import serialize_dict
    from scripts.relabel_nero_contact import relabel_task
    phase_key = "observation.contact_phase"
    features = {phase_key: {"dtype": "float32", "shape": [1]}}
    n = 400
    before = np.zeros((n, 1), np.float32)
    before[:48] = -1
    external = np.zeros((n, 7), np.float32)
    external[200:220, 0] = 1.2
    valid = np.ones(n, np.uint8)
    valid[:48] = 0
    data = tmp_path/"data/chunk-000/file-000.parquet"
    metadata = tmp_path/"meta/episodes/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    metadata.parent.mkdir(parents=True)
    table = pa.table({"episode_index": np.zeros(n, np.int64), "timing.state_timestamp_ns": np.arange(n)*10_000_000,
                      "observation.tau_ext": external.tolist(), "observation.tau_label_valid": valid,
                      phase_key: before[:, 0]})
    pq.write_table(table, data)
    stats = serialize_dict(compute_episode_stats({phase_key: before}, features))
    meta = {"episode_index": [0], **{f"stats/{phase_key}/{key}": [value] for key, value in stats[phase_key].items()}}
    pq.write_table(pa.table(meta), metadata)
    (tmp_path/"meta/info.json").write_text(json.dumps({"features": features}))
    (tmp_path/"meta/stats.json").write_text(json.dumps(stats))
    (tmp_path/"meta/torque_label_report.json").write_text(json.dumps({"teacher": {"checkpoint_sha256": "test"},
        "episodes": [{"episode_index": 0}], "phase_rule": {"contact_threshold": 3.}}))
    (tmp_path/"meta/world_model_timeline.json").write_text("{}")
    (tmp_path/"meta/wm_validation.json").write_text("{}")
    (tmp_path/"meta/wm_conversion_config.yaml").write_text(yaml.safe_dump({"tau_labels": {}, "io": {}}))
    report = relabel_task(tmp_path, 1., tmp_path)
    result = pq.read_table(data)
    labels = np.asarray(result[phase_key])
    assert (labels[:48] == -1).all() and (labels[100:200] == 1).all() and (labels[200:220] == 2).all()
    for key in table.column_names:
        if key != phase_key:
            assert table[key].equals(result[key])
    assert report["phase_counts"]["2"] == 20
    new_stats = json.loads((tmp_path/"meta/stats.json").read_text())[phase_key]
    assert new_stats["max"] == [2.] and new_stats["count"] == [n]
