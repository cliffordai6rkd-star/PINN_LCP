import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn.lowdim_encoder import MLPBlock, ResidualMLPBlock


# 输入为长度horizon的qk, vk, ak, tauk  输出F_k+1

class Fhead_transformerv1(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.history_keys = ["q", "v", "a", "tau"]
        self.cond_keys = ["ee_pose_next"]

        self.frame_dim = 28
        self.cond_dim = 7
        self.hidden_dim = 256
        self.wrench_dim = 6

        self.input_proj = nn.Linear(self.frame_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim * 4,
            dropout=1e-3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        self.cond_encoder = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.wrench_dim),
        )

    def forward(self, batch):
        q = batch["q"]      # [B, H, 7]
        v = batch["v"]      # [B, H, 7]
        a = batch["a"]      # [B, H, 7]
        tau = batch["tau"]  # [B, H, 7]

        history_x = torch.cat([q, v, a, tau], dim=-1)

        z = self.input_proj(history_x)
        z = z + self.pos_embed[:, :z.shape[1], :]
        z = self.transformer(z)

        h_context = z[:, -1, :]

        ee_pose_next = batch["ee_pose_next"]
        h_cond = self.cond_encoder(ee_pose_next)

        h = torch.cat([h_context, h_cond], dim=-1)
        wrench_pred = self.head(h)

        return {
            "wrench_pred": wrench_pred,
            "h_context": h_context,
            "h_cond": h_cond,
        }