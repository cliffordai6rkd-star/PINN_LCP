import torch 
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    """Return a standard torch.nn activation module by name."""

    name = name.lower()
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }

    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        if use_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()

        self.act = build_activation(activation)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block:

        y = shortcut(x) + MLP(x)

    说明：
    - MLP 负责残差分支 F(x)
    - shortcut 负责保留输入信息
    - final_act 负责残差相加后的非线性
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        depth: int = 2,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
        final_activation: str = "silu",
    ):
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth

        # 残差分支 F(x)
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout,
            final_activation="identity",
            final_norm=use_norm,
        )

        # shortcut 分支
        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_dim, out_dim)

        # 残差相加后的激活
        self.final_act = build_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.mlp(x)

        out = out + residual
        out = self.final_act(out)

        return out

class MLP(nn.Module):
    """
    Multi-layer MLP.

    结构:
        depth = 1:
            Linear(in_dim -> out_dim)

        depth >= 2:
            Linear(in_dim -> hidden_dim)
            hidden layers
            Linear(hidden_dim -> out_dim)

    说明:
        - MLPBlock 是单层积木
        - MLP 是多个 MLPBlock 的堆叠
        - final_activation 默认 identity，适合输出 feature / matrix raw value
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        depth: int = 3,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
        final_activation: str = "identity",
        final_norm: bool = False,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers = []

        if depth == 1:
            layers.append(
                MLPBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=final_activation,
                    use_norm=final_norm,
                    dropout=0.0,
                )
            )

        else:
            # 第一层：in_dim -> hidden_dim
            layers.append(
                MLPBlock(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    activation=activation,
                    use_norm=use_norm,
                    dropout=dropout,
                )
            )

            # 中间层：hidden_dim -> hidden_dim
            for _ in range(depth - 2):
                layers.append(
                    MLPBlock(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        activation=activation,
                        use_norm=use_norm,
                        dropout=dropout,
                    )
                )

            # 最后一层：hidden_dim -> out_dim
            layers.append(
                MLPBlock(
                    in_dim=hidden_dim,
                    out_dim=out_dim,
                    activation=final_activation,
                    use_norm=final_norm,
                    dropout=0.0,
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPEncoder(nn.Module):


    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 256,
        input_depth: int = 2,
        num_res_blocks: int = 3,
        block_depth: int = 2,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
        output_norm: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_depth = input_depth
        self.num_res_blocks = num_res_blocks
        self.block_depth = block_depth

        # 1. 输入投影: in_dim -> hidden_dim
        self.input_proj = MLP(
            in_dim=in_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=input_depth,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout,
            final_activation=activation,
            final_norm=use_norm,
        )

        # 2. 多个残差块
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualMLPBlock(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    out_dim=hidden_dim,
                    depth=block_depth,
                    activation=activation,
                    use_norm=use_norm,
                    dropout=dropout,
                    final_activation=activation,
                )
            )

        self.blocks = nn.Sequential(*blocks)

        # 3. 输出投影: hidden_dim -> out_dim
        if hidden_dim == out_dim:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = MLP(
                in_dim=hidden_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                depth=1,
                activation=activation,
                use_norm=use_norm,
                dropout=0.0,
                final_activation="identity",
                final_norm=False,
            )

        # 4. 输出归一化
        if output_norm:
            self.output_norm = nn.LayerNorm(out_dim)
        else:
            self.output_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = self.blocks(z)
        z = self.output_proj(z)
        z = self.output_norm(z)
        return z

class EncodedHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_shape: tuple[int, ...],
        hidden_dim: int,
        latent_dim: int,
        head_depth: int,
        scale: float = 1.0,
        activation: str = "silu",
        use_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.out_shape = out_shape
        self.scale = scale

        out_dim = 1
        for dim in out_shape:
            out_dim *= dim

        self.encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            input_depth=2,
            num_res_blocks=2,
            block_depth=2,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout,
            output_norm=True,
        )

        self.net = MLP(
            in_dim=latent_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            depth=head_depth,
            activation=activation,
            use_norm=use_norm,
            dropout=dropout,
            final_activation="identity",
            final_norm=False,
        )

    def forward(self, *xs):
        x = torch.cat(xs, dim=-1)
        z = self.encoder(x)

        batch_size = x.shape[0]
        y = self.net(z)
        y = y.view(batch_size, *self.out_shape)
        y = self.scale * y

        return y, z


if __name__ == "__main__":
    batch_size = 8

    q = torch.randn(batch_size, 7)
    dq = torch.randn(batch_size, 7)
    u = torch.randn(batch_size, 7)
    ee_pose = torch.randn(batch_size, 7)

    x = torch.cat([q, dq, u, ee_pose], dim=-1)

    encoder = MLPEncoder(
        in_dim=28,
        hidden_dim=256,
        out_dim=256,
        input_depth=2,
        num_res_blocks=3,
        block_depth=2,
        activation="silu",
        use_norm=True,
        dropout=0.0,
        output_norm=True,
    )

    z = encoder(x)

    print("input:", x.shape)
    print("z:", z.shape)
