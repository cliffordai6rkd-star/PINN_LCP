import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn.lowdim_encoder import MLPBlock, ResidualMLPBlock


# 输入为长度horizon的q v a （eepose？）条件为下一时刻的eepose序列  输出为未来的 q v a

class State_head_v(nn.Module):
    def __init__(self, config):
        super.__init__()
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
        # self.static_inputs = list(self.model_config.get("static_inputs") or ["q", "ee_pose"])
        # self.dynamic_inputs = list(self.model_config.get("dynamic_inputs") or ["q", "v", "a", "ee_pose"])
        self.inertial_term_inputs = list(self.model_config.get("inertial_term_inputs") or ["q", "a"])
        self.coriolis_term_inputs = list(self.model_config.get("coriolis_term_inputs") or ["q", "v" , "a"])

        self.wrench_dim = 6
        self.wrench_shape = (self.wrench_dim,)
        self.wrench_out_dim = self.wrench_dim

        self.hidden_dim = int(self.train_config.get("hidden_dim",256))
        self.activation = self.train_config.get("activation", "silu")
        self.use_norm = bool(self.train_config.get("use_norm", True))
        self.dropout = float(self.train_config.get("dropout", 1e-2))

        # self.static_num_res_blocks = int(self.train_config.get("static_num_res_blocks", 1))
        # self.dynamic_num_res_blocks = int(self.train_config.get("dynamic_num_res_blocks", 1))

        # self.active_inputs = list(dict.fromkeys(self.static_inputs + self.dynamic_inputs))
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

        self.m_term = MLPBlock(
            self.hidden_dim * len(self.static_inputs),
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )

        self.c_term = MLPBlock(
            self.hidden_dim * len(self.dynamic_inputs),
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )

        self.g_term = MLPBlock(
            self.hidden_dim * len(self.dynamic_inputs),
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )

        self.tau_term = MLPBlock(
            self.hidden_dim * len(self.dynamic_inputs),
            self.hidden_dim,
            self.activation,
            self.use_norm,
            self.dropout,
        )
        self.external_wrench_term = MLPBlock(
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

