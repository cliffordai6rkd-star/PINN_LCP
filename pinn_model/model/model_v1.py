import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入为长度horizon的qk, vk, ak, tauk  输出F_k+1  decoder_only

class Fhead_transformerv1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.data_config = config.get("dataloader") or {}
        self.train_config = config.get("train") or {}
        self.model_config = config.get("model") or {}

        self.activation = self.train_config.get("activation", "gelu")
        # H
        self.history_horizon = int(self.data_config.get("history_horizon",  8))
        # F
        self.future_horizon = int(self.data_config.get("future_horizon", 8))

        self.q_dim = int(self.model_config.get("q_dim", 7))
        self.v_dim = int(self.model_config.get("v_dim", 7))
        self.a_dim = int(self.model_config.get("a_dim", 7))
        self.tau_dim = int(self.model_config.get("tau_dim", 7))
        self.ee_pose_dim = int(self.model_config.get("ee_pose_dim", 7))
        self.wrench_dim = int(self.model_config.get("wrench_dim", 6))

        self.hidden_dim = int(self.train_config.get("hidden_dim", 256))

        # 将历史状态观测投影到隐藏层
        self.fram_dim = self.q_dim + self.v_dim + self.a_dim + self.tau_dim + self.ee_pose_dim
        self.history_proj = nn.Linear(self.frame_dim, self.hidden_dim)

        self.future_query = nn.Parameter(torch.zeros(1, self.future_horizon, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.history_horizon + self.future_horizon, self.hidden_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(self.train_config.get("nhead", 5)),
            dim_feedforward=self.hidden_dim * 5,
            dropout=float(self.train_config.get("dropout", 1e-3)),
            activation=self.activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(self.train_config.get("num_layers", 4)),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.wrench_dim),
        )
        # nn.init.normal_(self.future_query, mean=0.0, std=0.02)
        # nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, batch):
        q = batch["q"]
        v = batch["v"]
        a = batch["a"]
        tau = batch["tau"]

        history_x = torch.cat([q, v, a, tau], dim=-1)
        history_tokens = self.history_proj(history_x)

        B = history_tokens.shape[0]
        future_tokens = self.future_query.expand(B, -1, -1)

        tokens = torch.cat([history_tokens, future_tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :] # 兼容token长度变化：后面可能引入新的feature发生拼接操作

        # causal mask
        T = tokens.shape[1]
        # 生成一个 [T, T] 的上三角 mask True  = 不允许 attention
                                    # False = 允许 attention
        causal_mask = torch.triu(
            torch.ones(T, T, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        z = self.transformer(tokens, mask=causal_mask)

        future_z = z[:, self.history_horizon:, :]
        wrench_pred = self.head(future_z)

        return {
            "wrench_pred": wrench_pred,
        }
    
if __name__ == "__main__":
    B = 4
    H = 8
    F = 8
    config = {
        "dataloader": {
            "history_horizon": 8,
            "future_horizon": 8,
        },
        "model": {
            "q_dim": 7,
            "v_dim": 7,
            "a_dim": 7,
            "tau_dim": 7,
            "wrench_dim": 6,
        },
        "train": {
            "hidden_dim": 256,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 1e-3,
        },
    }
    model = Fhead_transformerv1(config)
    batch = {
        "q": torch.randn(B, H, 7),
        "v": torch.randn(B, H, 7),
        "a": torch.randn(B, H, 7),
        "tau": torch.randn(B, H, 7),
    }

    out = model(batch)

    print("wrench_pred:", out["wrench_pred"].shape)
    print("future_z:", out["future_z"].shape)

    assert out["wrench_pred"].shape == (B, F, 6)
    assert out["future_z"].shape == (B, F, 256)

    print("fake batch forward ok")