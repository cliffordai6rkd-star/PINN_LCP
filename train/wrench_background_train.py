import argparse
import logging
import torch
import torch.nn.functional as F
import yaml

from pathlib import Path
from dataset.dataloader import PINNDataset
from base_trainer import BaseTrainer
from wrench_bg.wrench_background import Wrench_Background

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
         return Wrench_Background(self.config)

    def compute_loss(self, batch):  
        out = self.model(batch)
        loss = F.mse_loss(out["wrench_pred"], out["wrench_target"])
        return loss, out
    
    @torch.no_grad()
    def summarize_best_gate(self):
        if not self.best_checkpoints:
            log.warning("no best checkpoint found, skip gate summary")
            return None
    
        best = self.best_checkpoints[0]
        ckpt_path = best["path"]
    
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
    
        loader = self.val_loader if self.val_loader is not None else self.loader
    
        gate_values = []
    
        for batch in loader:
            batch = self.batch_to_device(batch)
            out = self.model(batch)
    
            gate = out["gate"]  # [B, H, 3]
            gate = gate.reshape(-1, gate.shape[-1])
            gate_values.append(gate.detach().cpu())
    
        gate_values = torch.cat(gate_values, dim=0)
    
        gate_mean = gate_values.mean(dim=0)
        gate_std = gate_values.std(dim=0)
        gate_min = gate_values.min(dim=0).values
        gate_max = gate_values.max(dim=0).values
    
        names = ["q", "v", "ee_pose"]
    
        log.info(f"best checkpoint: {ckpt_path}")
        log.info(f"best {ckpt.get('monitor_key')}: {ckpt.get('monitor_score')}")
    
        for i, name in enumerate(names):
            log.info(
                f"gate/{name}: "
                f"mean={gate_mean[i]:.4f}, "
                f"std={gate_std[i]:.4f}, "
                f"min={gate_min[i]:.4f}, "
                f"max={gate_max[i]:.4f}"
            )
    
        return {
            "ckpt_path": ckpt_path,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "gate_min": gate_min,
            "gate_max": gate_max,
        }



if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"wrench background train config:{config}")
    trainer = WrenchBgTrainer(config)
    log.info(f"Start Training--------------------------")
    summary = trainer.train()
    trainer.summarize_best_gate()
    log.info("\n" + trainer.format_summary(summary))