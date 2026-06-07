import torch
import torch.nn as nn
import torch.nn.functional as F

from pinn.lowdim_encoder import MLPBlock, ResidualMLPBlock


# 直接将 q, v, u 输入端到端预测 lambda，再用物理/PINN loss 约束。
class PINN_v1(nn.Module):
    def __init__(
        self,
        q_dim: int = 7,
        v_dim: int = 7,
        tau_dim: int = 7,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        block_depth: int = 2,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.q_dim = q_dim
        self.v_dim = v_dim
        self.u_dim = u_dim
        self.lambda_dim = 4
        self.gamma_dim = 1
        self.xyz_dim = 3
        self.input_dim = self.q_dim + self.v_dim + self.u_dim

        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.block_depth = block_depth
        self.activation = activation
        self.use_norm = use_norm
        self.dropout = dropout

        self.register_buffer("dt", torch.tensor(1 / 30, dtype=torch.float32))

        self.input_proj = MLPBlock(
            in_dim=self.input_dim,
            out_dim=self.hidden_dim,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout,
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    in_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    depth=self.block_depth,
                    activation=activation,
                    use_norm=use_norm,
                    dropout=dropout,
                    final_activation=activation,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        self.lambda_head = nn.Linear(self.hidden_dim, self.lambda_dim)

    def forward(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor):
        x = torch.cat([q, v, u], dim=-1)
        z = self.input_proj(x)
        z = self.res_blocks(z)
        lambda_raw = self.lambda_head(z)

        gamma_raw = lambda_raw[..., : self.gamma_dim]
        xyz = lambda_raw[..., self.gamma_dim :]

        gamma_k = F.softplus(gamma_raw)
        lambda_pred = torch.cat([gamma_k, xyz], dim=-1)

        aux = {
            "gamma_k": gamma_k,
            "xyz": xyz,
            "lambda_raw": lambda_raw,
            "gamma_raw": gamma_raw,
            "z": z,
        }
        return lambda_pred, aux


if __name__ == "__main__":
    batch_size = 8

    model = PINN_v1()
    q = torch.randn(batch_size, model.q_dim)
    v = torch.randn(batch_size, model.v_dim)
    u = torch.randn(batch_size, model.u_dim)

    lambda_pred, aux = model(q, v, u)

    # print("lambda:", lambda_pred.shape)
    # print("gamma_k:", aux["gamma_k"].shape)
    # print("xyz:", aux["xyz"].shape)
