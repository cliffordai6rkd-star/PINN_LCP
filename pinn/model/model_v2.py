import torch
import torch.nn as nn


from pinn.lowdim_encoder import EncodedHead, MLPBlock
# s_k+1 = A s_k + B u_k + D lambda_k + d
# 输入:s_k = [q_k, v_k]  7,7  u_k = action:join[7](quaternion) or torque..
# 输出: lambda:4 -> layer ->wrench 
# 状态更新方程   预测下一帧的状态  用下一帧的真实数据与预测数据作loss

class LCP_PINN_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_dim = 7
        self.v_dim = 7
        self.u_dim = 7
        self.ee_pose_dim = 7
        self.lambda_dim = 4
        self.s_dim = self.q_dim + self.v_dim

        self.hidden_dim = 256
        self.latent_dim = 256
        self.head_depth = 2
        self.register_buffer("dt", torch.tensor(1/30, dtype=torch.float32)) # 采样freqeancy

        self.matrix_scale = 0.01
        self.vector_scale = 0.01
        self.lambda_scale = 1.0

        self.A_head = self._make_head(
            in_dim=self.q_dim + self.v_dim+self.u_dim,
            out_shape=(self.s_dim, self.s_dim),
            scale=self.matrix_scale,
        )
        self.B_head = self._make_head(
            in_dim=self.q_dim + self.v_dim + self.u_dim,
            out_shape=(self.s_dim, self.u_dim),
            scale=self.matrix_scale,
        )
        self.D_head = self._make_head(
            in_dim=self.q_dim + self.ee_pose_dim,
            out_shape=(self.s_dim, self.lambda_dim),
            scale=self.matrix_scale,
        )
        self.d_head = self._make_head(
            in_dim=self.q_dim + self.v_dim + self.u_dim, 
            out_shape=(self.s_dim,),
            scale=self.vector_scale,
        )
        self.lambda_head = self._make_head(
            in_dim=self.q_dim + self.v_dim + self.u_dim + self.ee_pose_dim,
            out_shape=(self.lambda_dim,),
            scale=self.lambda_scale,
        )

    # 函数名字前面的下划线:这个函数是类内部使用的辅助函数，不建议外部直接调用
    def _make_head(self, 
                in_dim:int,
                out_shape: tuple[int, ...], 
                scale:float,
                hidden_dim: int | None = None, 
                latent_dim: int | None = None, 
                head_depth: int | None = None) -> EncodedHead:
        return EncodedHead(
            in_dim=in_dim,
            out_shape=out_shape,
            hidden_dim=hidden_dim or self.hidden_dim,
            latent_dim=latent_dim or self.latent_dim,
            head_depth=head_depth or self.head_depth,
            scale=scale,
            activation="silu",
            use_norm=True,
            dropout=0.0,
        )

    def build_A(self, q, v, u):
        return self.A_head(q, v, u)
    
    def build_B(self, q, v, u):
        return self.B_head(q, v, u)

    def build_D(self, q, ee_pose):
        return self.D_head(q, ee_pose)

    def build_d(self, q, v, u):
        return self.d_head(q, v, u)
    
    def build_lambda(self, q, v, u, ee_pose):
        return self.lambda_head(q, v, u, ee_pose)
    
    def s_next_h(self, q, v, u, lambda_pred, ee_pose):
        s = torch.cat([q, v], dim=-1)

        A, z_A = self.build_A(q, v, u)
        B, z_B = self.build_B(q, v, u)
        D, z_D = self.build_D(q, ee_pose)
        d, z_d = self.build_d(q, v, u)

        # bmm按batch 做矩阵乘法
        As = torch.bmm(A, s.unsqueeze(-1)).squeeze(-1)  # A*s^T
        Bu = torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        Dlambda = torch.bmm(D, lambda_pred.unsqueeze(-1)).squeeze(-1)

        s_next_pred = As + Bu + Dlambda + d

        return s_next_pred, {
            "A": A,
            "B": B,
            "D": D,
            "d": d,
            "z_A": z_A,
            "z_B": z_B,
            "z_D": z_D,
            "z_d": z_d,
        }
    def forward(self, q, v, u, ee_pose, phi_k=None):
        lambda_pred, z_lambda = self.build_lambda(q, v, u, ee_pose)

        s_next_pred, aux = self.s_next_h(
            q=q,
            v=v,
            u=u,
            lambda_pred=lambda_pred,
            ee_pose=ee_pose,
        )

        aux["lambda_pred"] = lambda_pred
        aux["z_lambda"] = z_lambda

        if phi_k is not None:
            aux["phi_k"] = phi_k

        return s_next_pred, aux



if __name__ == "__main__":
    x = torch.randn(8, 14)

    # f = MLPBlock(in_dim=14, out_dim=256,activation="silu",use_norm=True,dropout=0.0)
    # y = f(x)
    # print(y.shape)
    model = LCP_PINN_v2()
    print(model)
    print("ok")

# import torch
# from PINN.LCP_model_v1 import LCP_PINN_v1

# model = LCP_PINN_v1()

# B = 8
# q = torch.randn(B, 7)
# v = torch.randn(B, 7)
# u = torch.randn(B, 7)
# ee_pose = torch.randn(B, 7)

# lambda_pred, _ = model.build_lambda(q, v, u, ee_pose)
# s_next_pred, aux = model.s_next_h(q, v, u, lambda_pred, ee_pose)

# print("lambda:", lambda_pred.shape)      # [8, 4]
# print("s_next:", s_next_pred.shape)      # [8, 14]
# print("A:", aux["A"].shape)              # [8, 14, 14]
# print("B:", aux["B"].shape)              # [8, 14, 7]
# print("D:", aux["D"].shape)              # [8, 14, 4]
# print("d:", aux["d"].shape)              # [8, 14]
