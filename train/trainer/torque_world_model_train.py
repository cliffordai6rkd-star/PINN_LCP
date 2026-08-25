"""Training entry point for the action-conditioned torque world model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from data_process.world_model_dataset import TorqueWorldModelDataset
from model.pinn_model.torque_world_model import TorqueWorldModel
from train.base_trainer import BaseTrainer
from train.torque_world_model_loss import TorqueWorldModelLoss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the q/dq/tau history-to-future torque world model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/torque_world_model.yaml"),
    )
    return parser.parse_args()


class TorqueWorldModelTrainer(BaseTrainer):
    """Episode-safe trainer for direct state/contact supervision."""

    def __init__(self, config):
        super().__init__(config)
        self.loss_calculator = TorqueWorldModelLoss(config)
        self.validation_flow_time = float(
            self.train_config.get("validation_flow_time", 0.5)
        )
        if not 0.0 <= self.validation_flow_time <= 1.0:
            raise ValueError("train.validation_flow_time must be in [0, 1]")

    def build_dataset(self):
        return TorqueWorldModelDataset(self.config, compute_normalizer=False)

    def build_model(self):
        return TorqueWorldModel(self.config)

    @staticmethod
    def _sample_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            return list(dataset.indices)
        return list(range(len(dataset)))

    def fit_dataset_normalizer(self, train_dataset):
        sample_indices = self._sample_indices(train_dataset)
        self.dataset.fit_normalizer(sample_indices)
        self.loss_calculator.set_normalizer(self.dataset.normalizer)
        self._fit_contact_weight(sample_indices)

    def _fit_contact_weight(self, sample_indices):
        if not (self.config.get("contact_gate") or {}).get("enabled", False):
            return
        if self.loss_calculator.contact_class_weights_is_auto:
            frame_indices = self.dataset.covered_raw_indices(sample_indices)
            labels = self.dataset.contact.index_select(
                0, torch.as_tensor(frame_indices, dtype=torch.long)
            ).reshape(-1).round().to(dtype=torch.long)
            class_count = self.loss_calculator.contact_state_count
            counts = torch.bincount(labels, minlength=class_count).to(dtype=torch.float64)
            if torch.any(counts <= 0):
                raise ValueError(
                    "automatic three-phase weighting requires all contact phases "
                    "to be present in the training episodes"
                )
            total = counts.sum()
            weights = total / (float(class_count) * counts)
            max_weight = float(
                (self.config.get("contact_gate") or {}).get(
                    "max_class_weight", 20.0
                )
            )
            weights = weights.clamp(max=max_weight)
            self.loss_calculator.set_contact_class_weights(weights.tolist())
            log.info(
                "contact phases: counts=%s "
                "class_weights=%s",
                [int(value) for value in counts],
                [round(float(value), 4) for value in weights],
            )
            return
        if not self.loss_calculator.contact_positive_class_weight_is_auto:
            return
        frame_indices = self.dataset.covered_raw_indices(sample_indices)
        labels = self.dataset.contact.index_select(
            0, torch.as_tensor(frame_indices, dtype=torch.long)
        )
        positive = labels.sum()
        negative = labels.numel() - positive
        if positive <= 0 or negative <= 0:
            raise ValueError(
                "automatic contact weighting requires contact and non-contact "
                "samples in the training episodes"
            )
        weight = float((negative / positive).item())
        self.loss_calculator.set_contact_positive_class_weight(weight)
        log.info(
            "contact labels: positive=%d negative=%d pos_weight=%.6f",
            int(positive.item()),
            int(negative.item()),
            weight,
        )

    def build_train_sampler(self, train_dataset):
        """Optionally sample windows by the phase reached in the future target.

        Sampling by the anchor's current phase only changes the history
        distribution.  SWM is intended to learn imminent contact, so each
        sample is weighted by the most advanced phase present in its future
        supervision window.
        """
        sampling = self.train_config.get("contact_sampling") or {}
        if not bool(sampling.get("enabled", False)):
            return None
        if not hasattr(self.dataset, "contact"):
            raise ValueError("contact_sampling requires dataset contact labels")
        base_indices = self._sample_indices(train_dataset)
        if not base_indices:
            raise ValueError("contact_sampling requires a non-empty train dataset")
        phase_weights = sampling.get("phase_weights", [1.0, 1.0, 1.0])
        phase_reduction = str(sampling.get("future_phase_reduction", "max")).lower()
        if phase_reduction != "max":
            raise ValueError(
                "train.contact_sampling.future_phase_reduction must be 'max'"
            )

        def future_labels():
            labels = []
            for sample_idx in base_indices:
                high_idx = self.dataset.valid_indices[int(sample_idx)]
                labels.append(
                    self.dataset.future_contact_phase(
                        high_idx, reduction=phase_reduction
                    )
                )
            return torch.as_tensor(labels, dtype=torch.long)

        if isinstance(phase_weights, str):
            if phase_weights.lower() == "auto":
                labels = future_labels()
                counts = torch.bincount(labels, minlength=self.loss_calculator.contact_state_count).float()
                phase_weights = torch.where(counts > 0, counts.sum() / counts.clamp_min(1.0), torch.zeros_like(counts))
                phase_weights = phase_weights.tolist()
            else:
                raise ValueError("train.contact_sampling.phase_weights must be a list or 'auto'")
        phase_weights = [float(value) for value in phase_weights]
        if len(phase_weights) != self.loss_calculator.contact_state_count or any(value < 0 for value in phase_weights):
            raise ValueError("train.contact_sampling.phase_weights has invalid length/values")
        labels = future_labels()
        weights = torch.as_tensor(phase_weights, dtype=torch.double).index_select(0, labels)
        if torch.any(weights <= 0):
            raise ValueError("every observed phase must have a positive sampling weight")
        return torch.utils.data.WeightedRandomSampler(
            weights,
            num_samples=int(sampling.get("num_samples", len(base_indices))),
            replacement=bool(sampling.get("replacement", True)),
        )

    def compute_loss(self, batch):
        self.loss_calculator.set_global_step(self.global_step)
        flow_time = None if self.model.training else self.validation_flow_time
        out = self.model(batch, flow_time=flow_time)
        loss, loss_dict = self.loss_calculator(out, batch)
        out["loss_dict"] = loss_dict
        return loss, out


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    log.info("torque world-model config: %s", config)
    trainer = TorqueWorldModelTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
