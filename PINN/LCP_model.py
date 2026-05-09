import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.get("input_dim",) 
        self.output_dim = config.get("output_dim",)
        self.hidden_dim = config.get("hidden_dim",128)
        self.num_layers = config.get("num_layers",3)

        layers = []

        layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)




class LCPContactPINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.image_feature_dim = config["image_feature_dim"]
        self.v_dim = config.get("v_dim", 7) # velocity:hirol data
        self.u_dim = config.get("u_dim", 8) # action: ee_pose[7]+gripper[1]

        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_layers = config.get("num_layers", 3)
 
        # lambda = [gamma, lambda_n, lambda_t]

        self.gamma_dim = config.get("gamma_dim", 1)      # 摩擦锥
        self.normal_dim = config.get("normal_dim", 1)    # 法向(z)
        self.tangent_dim = config.get("tangent_dim", 2)  # 切向(x,y),维度即摩擦锥近似边数 取决于Ft
        self.ft_dim = config.get("ft_dim", 6)

        self.lambda_dim = self.gamma_dim + self.normal_dim + self.tangent_dim

        # img > z_k > phi_k 
        self.phi_head = MLP({
            "input_dim": self.image_feature_dim,
            "output_dim": self.normal_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        })

        # u 摩擦系数 跟图像抓取的目标形成映射关系
        self.mu_head = MLP({
            "input_dim": self.image_feature_dim,
            "output_dim": 1,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        })

        # 编码v
        self.state_encoder = MLP({
            "input_dim": self.v_dim,
            "output_dim": self.hidden_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }) 
        # 编码 u 
        self.action_encoder = MLP({
            "input_dim": self.u_dim,
            "output_dim": self.hidden_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        })

        # 编码接触状态
        self.lambda_head = nn.Linear(self.hidden_dim, self.lambda_dim)

        # 接触力映射回力
        self.ft_head = nn.Linear(self.lambda_dim, self.ft_dim)

        fusion_input_dim = (
            self.image_feature_dim
            + self.normal_dim
            + 1
            + self.hidden_dim
            + self.hidden_dim
        )
