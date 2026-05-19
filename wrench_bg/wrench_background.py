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
# 在不同运动状态下 末端力受重力等自然力和控制力的影响程度是不同的  故引入门控机制


class Wrench_Background(nn.Module):
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

        self.hidden_dim = int(self.train_config.get("hidden_dim",128))
        self.activation = self.train_config.get("activation", "ReLU")
        self.use_norm = bool(self.train_config.get("use_norm", True))
        self.dropout = float(self.train_config.get("dropout", 1e-2))
        self.num_res_blocks = int(self.train_config.get("num_res_blocks", 1))

        self.input_dim = self.q_dim + self.v_dim + self.ee_pose_dim 
        
        self.q_encoder = MLPBlock(self.q_dim, self.hidden_dim, self.activation, self.use_norm, self.dropout)
        self.v_encoder = MLPBlock(self.v_dim, self.hidden_dim, self.activation,self.use_norm, self.dropout)
        self.ee_pose_encoder = MLPBlock(self.ee_pose_dim, self.hidden_dim, self.activation, self.use_norm, self.dropout)

        self.gate_input_dim =  self.hidden_dim * 3 # 可扩展：后面对不同输入encoder给的输入维度可以不同
        self.gate_net = nn.Sequential(
            MLPBlock(self.gate_input_dim, self.hidden_dim, self.activation, self.use_norm, self.dropout),
            nn.Linear(self.hidden_dim, 3),
        )

        self.fusion_blocks = nn.Sequential(
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
                for _ in range(self.num_res_blocks)
            ]
        )

        # 线性映射
        self.wrench_head = nn.Linear(self.hidden_dim, self.wrench_out_dim)

    def forward(self, batch):
        q = batch["q"]
        v = batch["v"]
        ee_pose = batch["ee_pose"]

        z_q = self.q_encoder(q)
        z_v = self.v_encoder(v)
        z_ee_pose = self.ee_pose_encoder(ee_pose)

        x = torch.cat([z_q, z_v, z_ee_pose], dim=-1)
        # 输出门控
        gate_logits = self.gate_net(x)
        gate = F.softmax(gate_logits, dim=-1)

        z_stack = torch.stack(
            [z_q, z_v, z_ee_pose],
            dim=-2,
        )

        z = (gate.unsqueeze(-1) * z_stack).sum(dim=-2)
        z = self.fusion_blocks(z)

        wrench_pred = self.wrench_head(z)
        wrench_pred = wrench_pred.view(
            *wrench_pred.shape[:-1],
            self.ft_window_size,
            self.wrench_dim,
        )

        out = {
            "wrench_pred": wrench_pred,
            "gate": gate,
            "z": z,
        }

        if "wrench" in batch:
            out["wrench_target"] = batch["wrench"]

        return out
    
if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model = Wrench_Background(config)

    from dataset.dataloader import PINNDataset
    dataset = PINNDataset(config)

    loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=4 ,
                                            shuffle=False,
                                            num_workers=4)
    batch = next(iter(loader))
    for k, v in batch.items():
        log.info(f"batch {k}: shape={v.shape}, dtype={v.dtype}")

    out = model(batch)

    print(out["wrench_pred"].shape)
    print(out["gate"].shape)
    if "wrench_target" in out:
        print(out["wrench_target"].shape)
    print(out["gate"][0, 0])
    print(out["gate"][0, 0].sum())