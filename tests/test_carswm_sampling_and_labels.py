from types import SimpleNamespace

import pytest
import torch

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_gate import (
    ContactGateConfig,
    contact_phase_labels_from_signal,
)
from train.contact_world_model_loss import ContactWorldModelLoss
from train.trainer.contact_world_model_train import ContactWorldModelTrainer


def test_temporal_precontact_labels_only_offline_prefix():
    config = ContactGateConfig(
        enabled=True,
        label_mode="three_phase",
        metric="tau_ext_l1",
        signal_on_threshold=3.0,
        signal_off_threshold=1.0,
        consecutive_frames=2,
        phase_label_mode="temporal_precontact",
        precontact_frames=3,
    )
    signal = torch.tensor([0.0, 0.0, 0.0, 0.0, 3.5, 3.5, 4.0, 0.5, 0.5])
    labels = contact_phase_labels_from_signal(signal, [(0, len(signal))], config)
    assert labels[:, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0]


def test_phase_sampler_importance_correction_recovers_uniform_expectation():
    dataset = ContactWorldModelDataset.__new__(ContactWorldModelDataset)
    dataset.valid_indices = [10, 20, 30]
    dataset.contact = torch.tensor([[0.0], [1.0], [2.0]])
    dataset.importance_weight_by_sample_index = {}
    dataset.future_contact_phase = lambda high_idx, reduction="max": {
        10: 0,
        20: 1,
        30: 2,
    }[high_idx]
    trainer = ContactWorldModelTrainer.__new__(ContactWorldModelTrainer)
    trainer.dataset = dataset
    trainer.train_config = {
        "contact_sampling": {
            "enabled": True,
            "phase_weights": [1.0, 5.0, 5.0],
            "future_phase_reduction": "max",
            "replacement": True,
        }
    }
    trainer.loss_calculator = SimpleNamespace(contact_state_count=3)
    subset = torch.utils.data.Subset(dataset, [0, 1, 2])
    sampler = trainer.build_train_sampler(subset)
    sampling_weight = sampler.weights.float()
    probability = sampling_weight / sampling_weight.sum()
    correction = torch.tensor(
        [
            dataset.importance_weight_by_sample_index[index]
            for index in (10, 20, 30)
        ]
    )
    arbitrary_loss = torch.tensor([2.0, 7.0, 13.0])
    corrected_expectation = (probability * correction * arbitrary_loss).sum()
    assert corrected_expectation == pytest.approx(arbitrary_loss.mean().item())


def test_validation_sample_without_training_weight_defaults_to_one():
    dataset = ContactWorldModelDataset.__new__(ContactWorldModelDataset)
    dataset.importance_weight_by_sample_index = {10: 0.2}
    assert dataset.importance_weight_by_sample_index.get(99, 1.0) == 1.0


def test_weighted_mean_applies_per_sample_correction():
    value = torch.tensor([1.0, 3.0])
    weight = torch.tensor([2.0, 0.5])
    result = ContactWorldModelLoss._weighted_mean(value, weight)
    assert result == pytest.approx(1.75)


def test_action_rollout_times_use_reanchored_v3_table_without_per_row_search():
    dataset = ContactWorldModelDataset.__new__(ContactWorldModelDataset)
    dataset.action_rollout_horizon = 3
    dataset.action_condition_horizon = 2
    dataset.action_start_offset = 1
    dataset.inference_delay_ns = 0
    dataset.high_timestamps = torch.tensor(
        [0, 10_000_000, 20_000_000, 30_000_000], dtype=torch.int64
    )
    dataset.action_indices = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    episode = {"dataset_to_index": 4}
    dataset._action_tables = {
        id(episode): {
            "indices": torch.tensor([0, 1, 2], dtype=torch.long),
            "rows": torch.tensor([0, 1, 2], dtype=torch.long),
            "times": torch.tensor(
                [0, 40_000_000, 80_000_000], dtype=torch.int64
            ),
        }
    }

    times = dataset._action_rollout_times_for_anchor(0, episode)
    torch.testing.assert_close(
        times,
        torch.tensor(
            [
                [0.04, 0.08],
                [0.03, 0.07],
                [0.02, 0.06],
            ]
        ),
    )
