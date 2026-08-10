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
from physics.nero_dynamics import PinocchioDynamics
from train.base_trainer import BaseTrainer
from train.torque_world_model_loss import TorqueWorldModelLoss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the q/tau history-to-future torque world model."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/torque_world_model.yaml"),
    )
    return parser.parse_args()


class TorqueWorldModelTrainer(BaseTrainer):
    """Episode-safe trainer with optional cached Nero physics labels."""

    def __init__(self, config):
        super().__init__(config)
        self.loss_calculator = TorqueWorldModelLoss(config)
        self.pinocchio_dynamics = (
            PinocchioDynamics(config)
            if self.loss_calculator.physics_enabled
            else None
        )
        self.validation_flow_time = float(
            self.train_config.get("validation_flow_time", 0.5)
        )
        if not 0.0 <= self.validation_flow_time <= 1.0:
            raise ValueError("train.validation_flow_time must be in [0, 1]")
        self.dynamics_cache = None

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
        if self.loss_calculator.physics_enabled:
            self.loss_calculator.load_tau_f_checkpoint(self.device)
            required_history = self.loss_calculator.tau_f_predictor.history_horizon
            if self.dataset.history_horizon < required_history:
                raise ValueError(
                    "dataloader.state_history_horizon must be at least the "
                    f"frozen tau_f horizon ({required_history}), got "
                    f"{self.dataset.history_horizon}"
                )
            self._build_dynamics_cache()

    def _fit_contact_weight(self, sample_indices):
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

    def _build_dynamics_cache(self):
        physics_config = self.config.get("physics") or {}
        chunk_size = int(physics_config.get("cache_chunk_size", 2048))
        if chunk_size <= 0:
            raise ValueError("physics.cache_chunk_size must be positive")
        q = self.dataset.high_tensors["q"]
        dq = self.dataset.high_tensors["dq"]
        ddq = self.dataset.high_tensors["ddq"]
        buffers = {
            "rnea_tau_id_future": [],
            "rnea_d_tau_d_q_future": [],
            "rnea_d_tau_d_dq_future": [],
            "rnea_d_tau_d_ddq_future": [],
            "frame_jacobian_future": [],
        }
        log.info(
            "precomputing local RNEA/Jacobian cache: frames=%d chunk=%d",
            q.shape[0],
            chunk_size,
        )
        for start in range(0, q.shape[0], chunk_size):
            end = min(start + chunk_size, q.shape[0])
            cache = self.pinocchio_dynamics.build_cache(
                q[start:end],
                dq[start:end],
                ddq[start:end],
                device=self.device,
                dtype=torch.float32,
            )
            buffers["rnea_tau_id_future"].append(
                cache.rnea.tau_id_reference
            )
            buffers["rnea_d_tau_d_q_future"].append(cache.rnea.d_tau_d_q)
            buffers["rnea_d_tau_d_dq_future"].append(
                cache.rnea.d_tau_d_dq
            )
            buffers["rnea_d_tau_d_ddq_future"].append(
                cache.rnea.d_tau_d_ddq
            )
            buffers["frame_jacobian_future"].append(cache.frame_jacobian)
        self.dynamics_cache = {
            key: torch.cat(values, dim=0).contiguous()
            for key, values in buffers.items()
        }
        cache_bytes = sum(
            value.numel() * value.element_size()
            for value in self.dynamics_cache.values()
        )
        log.info("Nero dynamics cache ready: %.2f MiB", cache_bytes / 2**20)

    def _attach_dynamics_cache(self, batch):
        if self.dynamics_cache is None:
            return
        indices = self._batch_future_indices(batch)
        for key, cache in self.dynamics_cache.items():
            batch[key] = cache[indices]

    @staticmethod
    def _batch_future_indices(batch):
        if "future_indices" not in batch:
            raise KeyError("physics-enabled batches require future_indices")
        indices = batch["future_indices"].to(dtype=torch.long)
        if indices.ndim != 2:
            raise ValueError("future_indices must have shape [B, T]")
        return indices

    def compute_loss(self, batch):
        self.loss_calculator.set_global_step(self.global_step)
        self._attach_dynamics_cache(batch)
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
