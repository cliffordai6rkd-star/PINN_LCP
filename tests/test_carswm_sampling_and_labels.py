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
