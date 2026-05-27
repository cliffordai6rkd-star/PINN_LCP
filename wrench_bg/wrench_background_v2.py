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
        self.model_config = config.get("model") or {}
        self.input_dims = {
            "q": 7,
            "v": 7,
            "a": 7,
            "ee_pose": 7,
            **(self.model_config.get("input_dims") or {}),
        }
        self.static_inputs = list(self.model_config.get("static_inputs") or ["q", "ee_pose"])
        self.dynamic_inputs = list(self.model_config.get("dynamic_inputs") or ["q", "v", "a", "ee_pose"])
        self._validate_input_groups()
        self.wrench_dim = 6
        self.wrench_shape = (self.wrench_dim,)
        self.wrench_out_dim = self.wrench_dim

        self.hidden_dim = int(self.train_config.get("hidden_dim",256))
        self.activation = self.train_config.get("activation", "silu")
        self.use_norm = bool(self.train_config.get("use_norm", True))
        self.dropout = float(self.train_config.get("dropout", 1e-2))
        self.static_num_res_blocks = int(self.train_config.get("static_num_res_blocks", 1))
        self.dynamic_num_res_blocks = int(self.train_config.get("dynamic_num_res_blocks", 1))

        self.active_inputs = list(dict.fromkeys(self.static_inputs + self.dynamic_inputs))
        self.encoders = nn.ModuleDict(
            {
                key: MLPBlock(
                    int(self.input_dims[key]),
                    self.hidden_dim,
                    self.activation,
                    self.use_norm,
                    self.dropout,
                )
                for key in self.active_inputs
            }
        )

        self.static_proj = MLPBlock(
            self.hidden_dim * len(self.static_inputs),
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )

        self.dynamic_proj = MLPBlock(
            self.hidden_dim * len(self.dynamic_inputs),
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

    def _validate_input_groups(self):
        if not self.static_inputs:
            raise ValueError("model.static_inputs must not be empty.")
        if not self.dynamic_inputs:
            raise ValueError("model.dynamic_inputs must not be empty.")

        unknown = [
            key
            for key in self.static_inputs + self.dynamic_inputs
            if key not in self.input_dims
        ]
        if unknown:
            raise ValueError(f"Unknown model input keys: {unknown}")

    def _prepare_input(self, key, value):
        expected_dim = int(self.input_dims[key])
        if value.shape[-1] == expected_dim:
            return value

        if value.ndim >= 2:
            value = value.flatten(start_dim=-2)
            if value.shape[-1] == expected_dim:
                return value

        raise ValueError(
            f"Input {key!r} has shape {tuple(value.shape)}, expected last dim {expected_dim}."
        )

    def _encode_inputs(self, batch):
        encoded = {}
        for key in self.active_inputs:
            if key not in batch:
                raise KeyError(f"Missing model input {key!r} in batch.")
            encoded[key] = self.encoders[key](self._prepare_input(key, batch[key]))
        return encoded

    def forward(self, batch):
        z = self._encode_inputs(batch)

        static_x = torch.cat([z[key] for key in self.static_inputs], dim=-1)
        dynamic_x = torch.cat([z[key] for key in self.dynamic_inputs], dim=-1)

        z_static = self.static_proj(static_x)
        z_static = self.static_blocks(z_static)

        z_dynamic = self.dynamic_proj(dynamic_x)
        z_dynamic = self.dynamic_blocks(z_dynamic)

        wrench_static = self.static_head(z_static)
        wrench_dynamic = self.dynamic_head(z_dynamic)

        wrench_pred = wrench_static + wrench_dynamic

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
