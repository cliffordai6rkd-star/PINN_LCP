import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import yaml
import logging as log

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from pathlib import Path
from pinn.lowdim_encoder import MLPBlock, ResidualMLPBlock

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert standalone .h5/.hdf5 episode files to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/dataset_test_cfg.yaml"),
        help="Path to the config",
    )
    return parser.parse_args()


#  q, v, u 输入端到端预测 wrench  
# 静态力和加速度导致的力分两个head分开预测

class Wrench_Background_V2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 
        self.train_config = config.get("train") or {}
        self.q_dim = 7
        self.v_dim = 7
        self.ee_pose_dim = 7
        self.ft_window_size = 4
        self.wrench_dim = 6
        self.wrench_shape = (self.ft_window_size, self.wrench_dim)
        self.wrench_out_dim = self.ft_window_size * self.wrench_dim

        self.hidden_dim = int(self.train_config.get("hidden_dim",256))
        self.activation = self.train_config.get("activation", "silu")
        self.use_norm = bool(self.train_config.get("use_norm", True))
        self.dropout = float(self.train_config.get("dropout", 1e-2))
        self.static_num_res_blocks = int(self.train_config.get("static_num_res_blocks", 1))
        self.dynamic_num_res_blocks = int(self.train_config.get("dynamic_num_res_blocks", 1))

        self.input_dim = self.q_dim + self.v_dim + self.ee_pose_dim 
        
        self.q_encoder = MLPBlock(self.q_dim, self.hidden_dim, self.activation, self.use_norm, self.dropout)
        self.v_encoder = MLPBlock(self.v_dim, self.hidden_dim, self.activation,self.use_norm, self.dropout)
        self.ee_pose_encoder = MLPBlock(self.ee_pose_dim, self.hidden_dim, self.activation, self.use_norm, self.dropout)

        self.static_proj = MLPBlock(
            self.hidden_dim * 2,
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )

        self.dynamic_proj = MLPBlock(
            self.hidden_dim * 3,
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )
        self.static_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    in_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    depth=2,
                    activation=self.activation,
                    use_norm=self.use_norm,
                    dropout=self.dropout,
                    final_activation=self.activation,
                )
                for _ in range(self.static_num_res_blocks)
            ]
        )

        self.dynamic_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    in_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    depth=2,
                    activation=self.activation,
                    use_norm=self.use_norm,
                    dropout=self.dropout,
                    final_activation=self.activation,
                )
                for _ in range(self.dynamic_num_res_blocks)
            ]
        )

        # 线性映射
        self.static_head = nn.Linear(self.hidden_dim, self.wrench_out_dim)
        self.dynamic_head = nn.Linear(self.hidden_dim, self.wrench_out_dim)

    def forward(self, batch):
        q = batch["q"]
        v = batch["v"]
        ee_pose = batch["ee_pose"]

        z_q = self.q_encoder(q)
        z_v = self.v_encoder(v)
        z_ee_pose = self.ee_pose_encoder(ee_pose)

        static_x = torch.cat([z_q, z_ee_pose], dim=-1)
        dynamic_x = torch.cat([z_q, z_v, z_ee_pose], dim =-1)

        z_static = self.static_proj(static_x)
        z_static = self.static_blocks(z_static)

        z_dynamic = self.dynamic_proj(dynamic_x)
        z_dynamic = self.dynamic_blocks(z_dynamic)

        wrench_static = self.static_head(z_static)
        wrench_dynamic = self.dynamic_head(z_dynamic)

        wrench_pred = wrench_static + wrench_dynamic

        wrench_pred = wrench_pred.view(
            *wrench_pred.shape[:-1],
            self.ft_window_size,
            self.wrench_dim,
        )

        wrench_static = wrench_static.view(
            *wrench_static.shape[:-1],
            self.ft_window_size,
            self.wrench_dim,
        )

        wrench_dynamic = wrench_dynamic.view(
            *wrench_dynamic.shape[:-1],
            self.ft_window_size,
            self.wrench_dim,
        )

        out = {
            "wrench_pred": wrench_pred,
            "wrench_static": wrench_static,
            "wrench_dynamic": wrench_dynamic,
            "z_static": z_static,
            "z_dynamic": z_dynamic,
        }
        
        if "wrench" in batch:
            out["wrench_target"] = batch["wrench"]

        return out
    
if __name__ == "__main__":
    pass