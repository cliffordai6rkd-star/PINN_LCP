"""Pilot interpolation preserves causal histories and does not fabricate labels."""
import copy

import numpy as np
import pytest
import torch

from data_process.interpolated_contact_dataset import InterpolatedContactDataset
from model.pinn_model.contact_scfm import SCFMContactWorldModel, make_contact_student
from model.pinn_model.latent_pretrained import normalizer_envelope
from model.pinn_model.scfm_distillation import SCFMSettings
from scripts.posttrain_inverse_gaussian_latent import episode_train_indices
from scripts.posttrain_scfm_latent import validate
from test_contact_world_model import config
from train.nomalizer import Normalizer


def setup():
    cfg = config()
    cfg["model"].update(joint_dim=7, action_dim=7)
    cfg["dataloader"].update(high_fps=100, expert_fps=25, state_history_horizon=20,
                             prediction_horizon=40, action_condition_horizon=10, pad_history=True,
                             normalize_mode="gaussian", normalize_lowdim_keys=["q", "dq", "delta_q", "tau", "action"])
    cfg["train"] = {"seed": 42, "val_ratio": .5}
    episodes = []
    for identifier in (3, 9):
        t = np.arange(40)*.04
        q = np.tile(t[:, None], (1, 7)).astype(np.float32)
        actions = np.zeros((len(t), 7), np.float32)
        actions[:, 0] = t
        actions[:, 6] = 1
        episodes.append({"episode_index": np.full(len(t), identifier),
                         "timing.master_timestamp_ns": np.rint(t*1e9).astype(np.int64)+1700000000000000000,
                         "observation.joint": q, "observation.velocity": np.ones_like(q),
                         "observation.torque": 2*q, "action.joint": q.copy(), "action.ee_pose": actions})
    return cfg, episodes


def test_windows_action_alignment_proxy_and_episode_split():
    cfg, episodes = setup()
    dataset = InterpolatedContactDataset.from_episodes(cfg, episodes)
    assert len(dataset) == 60
    values = dataset[10]
    expected_history = torch.arange(21, 41).float()/100
    torch.testing.assert_close(values["q"][:, 0], expected_history)
    torch.testing.assert_close(values["q_future"][:, 0], torch.arange(41, 81).float()/100)
    torch.testing.assert_close(values["action"][:, 0], torch.arange(11, 21).float()*.04)
    assert values["delta_q"].count_nonzero() == 0
    assert "contact_future" not in values and "contact" not in values
    assert dataset[0]["q"].count_nonzero() == 0
    assert dataset[30]["episode_index"] == 9
    assert dataset.provenance["contact_labels_available"] is False
    train, _ = episode_train_indices(dataset, cfg)
    assert len(train) == 30
    assert len({int(dataset[i]["episode_index"]) for i in train}) == 1
    stats = {key: {"mean": torch.ones(7), "std": torch.ones(7)} for key in cfg["dataloader"]["normalize_lowdim_keys"]}
    normalizer = Normalizer(stats)
    dataset.set_normalizer(normalizer)
    torch.testing.assert_close(dataset[10]["q"], normalizer.gaussian_normalize("q", values["q"]))


def test_interpolation_and_causal_filters_never_read_beyond_anchor():
    cfg, episodes = setup()
    cfg["dataloader"]["filters"] = {key: {"enabled": True, "operations": [{"type": "lowpass", "cutoff_hz": 15, "order": 2}]}
                                     for key in ["q", "dq", "tau", "delta_q"]}
    first = InterpolatedContactDataset.from_episodes(cfg, episodes)
    changed = copy.deepcopy(episodes)
    for key in ["observation.joint", "observation.velocity", "observation.torque", "action.joint"]:
        changed[0][key][11:] += 100
    second = InterpolatedContactDataset.from_episodes(cfg, changed)
    for key in cfg["model"]["inputs"]:
        torch.testing.assert_close(first[10][key], second[10][key], rtol=0, atol=0)
    assert not torch.equal(first[10]["q_future"], second[10]["q_future"])


def test_validator_reports_teacher_agreement_without_contact_f1():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg, episodes = setup()
        dataset = InterpolatedContactDataset.from_episodes(cfg, episodes)
        normalizer = Normalizer({key: {"mean": torch.zeros(7), "std": torch.ones(7)} for key in cfg["dataloader"]["normalize_lowdim_keys"]})
        dataset.set_normalizer(normalizer)
        teacher = SCFMContactWorldModel(cfg).eval()
        student, _ = make_contact_student(teacher, {"config": cfg}, 4)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        settings = SCFMSettings(validation_batches=1, validation_samples=2, reference_steps=2)
        result = validate(student, teacher, loader, settings, torch.device("cpu"), normalizer_envelope(normalizer, cfg))
        assert result["contact_labels_available"] is False
        for row in result["variants"].values():
            assert "contact_macro_f1" not in row and "contact_confusion" not in row
            assert row["paired_teacher_contact_kl"] >= -1e-7
        assert result["variants"]["teacher_fine"]["paired_teacher_flow_rmse"] == 0
    finally:
        torch.set_num_threads(old_threads)


def test_reject_nonmonotonic_episode_timestamps():
    cfg, episodes = setup()
    episodes[0]["timing.master_timestamp_ns"][10] = episodes[0]["timing.master_timestamp_ns"][9]
    with pytest.raises(ValueError, match="nonmonotonic"):
        InterpolatedContactDataset.from_episodes(cfg, episodes)
