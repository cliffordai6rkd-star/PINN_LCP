import argparse
import logging
import torch
import torch.nn.functional as F
import yaml

from pathlib import Path
from dataset.dataloader import PINNDataset
from base_trainer import BaseTrainer
from pinn_model.wrench_bg.wrench_background_v2 import Wrench_Background_V2

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

class WrenchBgTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def build_dataset(self):
        return PINNDataset(self.config)

    def build_model(self):
         return Wrench_Background_V2(self.config)

    def compute_loss(self, batch):  
      
        out = self.model(batch)
        loss = F.mseloss(out(batch["wrench_pred"]),out(batch["wrench_target"]))
        return loss, out
    



if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"wrench background train config:{config}")
    trainer = WrenchBgTrainer(config)
    log.info(f"Start Training--------------------------")
    summary = trainer.train()

    log.info("\n" + trainer.format_summary(summary))