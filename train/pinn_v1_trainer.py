import argparse
import logging
import torch
import torch.nn.functional as F
import yaml

from pathlib import Path
from dataset.dataloader import PINNDataset
from base_trainer import BaseTrainer
from pinn_model.model.model_v1 import Fhead_transformerv1

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/dataset_test_cfg.yaml"),
    )
    return parser.parse_args()

class PinnV1_trainer(BaseTrainer):
    def __init__(self, config):
        super(),self.__init__(config)
    def build_dataset(self):
        return PINNDataset(self.config)

    def build_model(self):
         return Fhead_transformerv1(self.config)

    def compute_loss(self, batch):  
        out = self.model(batch)
        loss = F.mse_loss(out["wrench_pred"], out["wrench_target"])
        return loss, out