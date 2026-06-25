import argparse
import logging
import torch
import torch.nn.functional as F
import yaml

from pathlib import Path
from data_process.dataloader import PINNHistoryFutureDataset
from train.base_trainer import BaseTrainer
from pinn_model.model.model_v1 import Fhead_transformerv1
from train.pinn_loss import PinnLossCalculator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/pinn_transformer.yaml"),
    )
    return parser.parse_args()

class PinnV1_trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.loss_calculator = PinnLossCalculator(config)

    def build_dataset(self):
        return PINNHistoryFutureDataset(self.config)

    def build_model(self):
         return Fhead_transformerv1(self.config)

    def compute_loss(self, batch):  
        out = self.model(batch)
        loss, loss_dict = self.loss_calculator(out, batch)
        out["loss_dict"] = loss_dict
        return loss, out
    
if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    log.info(f"pinn v1 train config: {config}")
    trainer = PinnV1_trainer(config)
    summary = trainer.train()
    log.info("\n" + trainer.format_summary(summary))
