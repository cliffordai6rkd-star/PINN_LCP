# Transformer 原理与 Fhead_transformerv1 设计笔记

这份笔记用于理解 Transformer 的底层数学，以及为什么当前 `Fhead_transformerv1` 适合使用 Encoder-only 结构。

你的当前任务可以写成：

$$
\{q_t, v_t, a_t, \tau_t\}_{t=k-H+1}^{k},\ e_{k+1}
\rightarrow
F_{k+1}
$$

其中：

$$
q_t, v_t, a_t, \tau_t \in \mathbb{R}^{7}
$$

$$
e_{k+1} \in \mathbb{R}^{7}
$$

$$
F_{k+1} \in \mathbb{R}^{6}
$$

也就是说，模型输入是过去长度为 `H` 的低维状态历史，以及下一时刻末端位姿条件 `ee_pose_{k+1}`，输出是下一时刻的 6 维 wrench。

## 1. Token 是什么

Transformer 处理的是 token 序列。

在语言模型里，一个 token 可以是一个词或一个子词。

在你的任务里，一个 token 可以是一帧机器人状态：

$$
x_t = [q_t,\ v_t,\ a_t,\ \tau_t]
$$

因为每个量都是 7 维，所以：

$$
x_t \in \mathbb{R}^{28}
$$

如果历史窗口长度为 `H`，输入序列就是：

$$
X = [x_{k-H+1}, x_{k-H+2}, \dots, x_k]
$$

对应张量形状为：

```text
[B, H, 28]
```

其中：

```text
B: batch size
H: history horizon
28: 每一帧 q/v/a/tau 拼接后的维度
```

## 2. Input Embedding

Transformer 内部通常使用统一的 hidden dimension，例如：

```text
hidden_dim = 256
```

所以需要先把每个 28 维 token 映射到 256 维：

$$
z_t = W_e x_t + b_e
$$

其中：

$$
W_e \in \mathbb{R}^{256 \times 28}
$$

映射后：

$$
z_t \in \mathbb{R}^{256}
$$

张量形状从：

```text
[B, H, 28]
```

变成：

```text
[B, H, hidden_dim]
```

## 3. 位置编码

Transformer 本身不知道顺序。

如果不加位置编码，模型并不知道哪个 token 是第 1 帧，哪个 token 是第 H 帧。

所以要给每个时间位置加一个位置向量：

$$
\tilde{z}_t = z_t + p_t
$$

其中：

$$
p_t \in \mathbb{R}^{hidden\_dim}
$$

在代码里通常是：

```python
self.pos_embed = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
```

forward 里：

```python
z = z + self.pos_embed[:, :H, :]
```

这样模型就能区分不同时间步。

## 4. Self-Attention 的核心数学

Self-Attention 的核心是让每个 token 根据相关性，从其他 token 中取信息。

给定输入：

$$
X \in \mathbb{R}^{H \times d}
$$

首先通过三组线性变换得到：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

其中：

```text
Q: Query
K: Key
V: Value
```

直觉上：

```text
Query: 当前 token 想找什么信息
Key: 每个 token 提供什么索引
Value: 每个 token 真正携带的信息
```

注意力分数为：

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

其中：

$$
S \in \mathbb{R}^{H \times H}
$$

第 `i,j` 个元素表示：

```text
第 i 个 token 对第 j 个 token 的关注程度
```

然后做 softmax：

$$
A = \mathrm{softmax}(S)
$$

最后用注意力权重加权 Value：

$$
Y = AV
$$

完整公式：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

在你的任务里，Self-Attention 可以理解为：

```text
当前帧 token 可以自动关注历史窗口中更重要的帧
例如速度突变、加速度突变、tau 变化、接触发生前的状态
```

## 5. Multi-Head Attention

一个 attention head 只能学习一种关系。

Multi-head attention 会并行使用多个 attention head：

$$
head_i =
\mathrm{Attention}
(Q_i, K_i, V_i)
$$

然后拼接：

$$
H = [head_1,\ head_2,\ \dots,\ head_m]
$$

再做一次线性变换：

$$
Y = HW_O
$$

直觉上：

```text
head 1 可能关注速度变化
head 2 可能关注 tau 和接触关系
head 3 可能关注加速度冲击
head 4 可能关注特定关节状态
```

所以多头注意力允许模型从多个角度理解同一段历史窗口。

## 6. Transformer Block

一个 Transformer block 通常包含两部分：

```text
Self-Attention
Feed Forward Network
```

Feed Forward Network 是逐 token 的 MLP：

$$
\mathrm{FFN}(x)
=
W_2 \sigma(W_1 x + b_1) + b_2
$$

其中：

$$
\sigma
$$

可以是 ReLU、GELU、SiLU 等激活函数。

一个常见的 Pre-LN Transformer block 可以写成：

$$
x' = x + \mathrm{SelfAttention}(\mathrm{LayerNorm}(x))
$$

$$
y = x' + \mathrm{FFN}(\mathrm{LayerNorm}(x'))
$$

其中：

```text
Residual connection: 保留原始信息，稳定训练
LayerNorm: 归一化 hidden feature，稳定训练
FFN: 对每个 token 做非线性变换
```

## 7. Encoder 是什么

Encoder 的作用是：

```text
输入一段 token
让 token 之间通过 self-attention 交换信息
输出一段融合上下文后的 token
```

数学形式：

$$
H = \mathrm{TransformerEncoder}(X)
$$

输入：

```text
[B, H, hidden_dim]
```

输出：

```text
[B, H, hidden_dim]
```

输出里的最后一个 token：

$$
h_k
$$

已经不仅仅是当前帧的信息，而是：

```text
当前帧信息 + 它从历史窗口中吸收的上下文信息
```

所以你的模型可以取：

```python
h_context = z[:, -1, :]
```

作为历史动力学上下文。

## 8. Encoder-only 是什么

Encoder-only 表示模型只有 Encoder，没有 Decoder。

结构是：

```text
输入序列
-> TransformerEncoder
-> 上下文表示
-> 任务 head
```

它适合：

```text
序列分类
序列回归
历史窗口预测一个结果
```

你的任务：

$$
\{q_t, v_t, a_t, \tau_t\}_{t=k-H+1}^{k},\ e_{k+1}
\rightarrow
F_{k+1}
$$

非常适合 Encoder-only。

原因是你不是在逐步生成一串输出 token，而是用历史上下文预测一个 wrench。

推荐结构：

```text
history(q, v, a, tau)
-> TransformerEncoder
-> h_context

ee_pose_{k+1}
-> condition encoder
-> h_cond

concat(h_context, h_cond)
-> MLP head
-> wrench_{k+1}
```

## 9. Decoder 是什么

Decoder 一般用于生成序列。

Decoder 里有两种 attention：

```text
1. Decoder self-attention
2. Cross-attention to encoder memory
```

Self-attention 是 decoder token 之间互相看。

Cross-attention 是 decoder token 去看 encoder 输出的条件信息。

Cross-attention 的数学形式是：

$$
Q = X_{dec} W_Q
$$

$$
K = X_{enc} W_K
$$

$$
V = X_{enc} W_V
$$

然后：

$$
\mathrm{CrossAttention}
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

注意这里：

```text
Q 来自 Decoder
K, V 来自 Encoder memory
```

直觉上：

```text
Decoder 当前要生成一个输出 token
它去 Encoder 的条件信息里查找相关内容
```

## 10. Encoder-Decoder 是什么

Encoder-Decoder 结构是：

```text
条件输入
-> Encoder
-> memory

待生成序列
-> Decoder
-> 输出序列
```

经典例子是翻译：

```text
英文句子 -> Encoder -> memory
中文句子 -> Decoder -> 逐词生成
```

在 Diffusion Policy 里类似：

```text
obs 条件 + diffusion timestep
-> Encoder
-> memory

noisy action trajectory
-> Decoder
-> denoised action trajectory
```

所以 Encoder-Decoder 适合：

```text
输入条件和输出序列是两组不同 token
输出本身也是一段序列
需要每个输出 token 去 cross-attention 条件信息
```

## 11. Diffusion Policy 的 TransformerForDiffusion

`diffusion_policy/model/diffusion/transformer_for_diffusion.py` 不是普通的 `nn.TransformerEncoder`，而是一个为扩散模型轨迹去噪封装的完整模型。

它包含：

```text
input embedding
position embedding
diffusion timestep embedding
condition encoder
optional transformer decoder
causal mask
output head
optimizer 参数分组
```

它的 forward 接口是：

```text
sample:   [B, T, input_dim]
timestep: [B]
cond:     [B, T_cond, cond_dim]
output:   [B, T, output_dim]
```

普通 `nn.TransformerEncoder` 只做：

```text
[B, T, D] -> [B, T, D]
```

而 `TransformerForDiffusion` 做的是：

```text
noisy trajectory + diffusion timestep + condition
-> predicted noise / denoised trajectory
```

所以它比你当前任务需要的结构复杂得多。

## 12. 为什么你的 Fhead 先用 Encoder-only

你的模型目标是：

$$
\{q_t, v_t, a_t, \tau_t\}_{t=k-H+1}^{k}
\quad \text{conditioned on} \quad
e_{k+1}
\rightarrow
F_{k+1}
$$

这不是扩散去噪，也不是自回归生成一整段序列。

因此第一版推荐：

```text
history tokens:
q, v, a, tau

condition:
ee_pose_{k+1}

output:
wrench_{k+1}
```

结构：

```text
[q, v, a, tau] history
-> Linear input projection
-> position embedding
-> TransformerEncoder
-> take last token h_context

ee_pose_{k+1}
-> condition encoder
-> h_cond

concat(h_context, h_cond)
-> MLP head
-> wrench_pred
```

对应张量形状：

```text
q:            [B, H, 7]
v:            [B, H, 7]
a:            [B, H, 7]
tau:          [B, H, 7]
ee_pose_next: [B, 7]

history_x:    [B, H, 28]
z:            [B, H, hidden_dim]
h_context:    [B, hidden_dim]
h_cond:       [B, hidden_dim]
wrench_pred:  [B, 6]
```

## 13. Fhead_transformerv1 的伪代码

```python
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
```

## 14. 什么时候需要 Encoder-Decoder

当前第一版不需要 Encoder-Decoder。

以后如果你要做下面这些任务，再考虑 Encoder-Decoder：

```text
1. 输入历史窗口，输出未来 K 帧 wrench trajectory
2. 对未来 wrench trajectory 做 diffusion denoising
3. 像 Diffusion Policy 一样，把 obs/history 作为 condition memory，把 noisy trajectory 作为 decoder tokens
4. 输出序列中的每个 token 都需要主动查询历史条件
```

例如：

$$
\{q_t, v_t, a_t, \tau_t\}_{t=k-H+1}^{k},\ e_{k+1:k+K}
\rightarrow
F_{k+1:k+K}
$$

这时可以考虑：

```text
Encoder: history tokens
Decoder: future query tokens / noisy wrench tokens
Output: future wrench tokens
```

但当前阶段，先把 Encoder-only 单步预测跑通，是更稳的路径。
