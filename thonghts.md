# Flow Matching 与 LCP-PINN 思考整理

## 1. Flow Matching、DDPM 与 DDIM

在 Flow Matching 中，policy 最后输出的 action 虽然是一个确定的数值或一段确定的 action sequence，但模型整体学习的并不是简单的确定映射：

$$
o \mapsto a
$$

而是条件动作分布：

$$
p(a \mid o)
$$

这里的 $o$ 是观测，例如图像、力、机器人状态等。它决定“哪些 action 是合理的”。初始噪声：

$$
a_T \sim \mathcal{N}(0, I)
$$

则决定“这次从这些合理 action 中生成哪一个”。

因此，网络本身通常不是直接输出概率值，也不是严格意义上的“分布函数”，而是输出去噪方向、score 或 velocity，例如：

$$
\epsilon_\theta(a_t, t, o)
$$

或：

$$
v_\theta(a_t, t, o)
$$

它们用于告诉采样过程：当前 noisy action 应该往哪里修正，才能逐渐变成合理 action。由于初始噪声是随机的，所以即使观测 $o$ 固定，不同的初始噪声也可能生成不同但合理的 action。这些 action 的整体就构成了：

$$
p_\theta(a \mid o)
$$

DDPM 和 DDIM 的区别在于采样过程的随机性。DDPM 反向去噪时通常每一步都会重新加入随机噪声，所以随机性来自初始噪声和每一步的额外噪声。即使初始噪声固定，只要每一步噪声不同，最终 action 也可能不同。

DDIM 不只是“减少采样步数”。当 $\eta = 0$ 时，它可以变成确定性采样。每一步不再额外加入随机噪声，所以在观测 $o$、初始噪声 $a_T$、网络参数和采样器都固定的情况下，最终生成的 action 是确定的。

马尔可夫过程可以理解为链式传播，但关键条件是：下一步只依赖当前状态，不依赖更早历史。DDPM 的正向加噪和常规反向去噪都是一环接一环的马尔可夫形式，而 DDIM 构造的是一种隐式的、可跨步的非马尔可夫采样路径。

Flow Matching 可以理解为：在给定观测条件 $o$ 下，一团初始噪声概率云 $p_0(a \mid o)$ 在 action 空间中沿着学习到的速度场 $v_\theta(a,t,o)$ 连续流动，并逐渐变形为专家动作概率云 $p_{\mathrm{data}}(a \mid o)$。这个过程类似流体力学中的质量守恒，满足连续性方程：

$$
\frac{\partial p_t(a)}{\partial t}
+ \nabla_a \cdot \left(p_t(a) v_t(a)\right)
= 0
$$

其中：

- $p_t(a)$ 是概率密度，类似流体密度。
- $v_t(a)$ 是概率流速度，也就是噪声分布流向动作分布的速度。
- $p_t(a)v_t(a)$ 类似通量。

## 2. 为什么加入力信号

加入力的任务更适合精细操作，而不是那种视觉完全主导、只要力足够大就能执行的任务。因此 action head 对当前观测下的 end-effector pose 判断应该准确。

所以与图像做 cross attention 的不应该只有 force/torque，也应该考虑：

$$
\text{image},\quad \text{ee pose},\quad \text{force/torque}
$$

一种更合理的融合方式是：

$$
z_{\mathrm{img}} = \mathrm{ImageEncoder}(I_k)
$$

$$
z_{\mathrm{ee}} = \mathrm{MLP}_{ee}(T_k)
$$

$$
z_{\mathrm{ft}} = \mathrm{MLP}_{ft}(f_k)
$$

然后让：

$$
(z_{\mathrm{img}}, z_{\mathrm{ee}})
$$

先形成当前空间状态表征，再与 $z_{\mathrm{ft}}$ 做 cross attention 或 gated fusion。

## 3. 整体问题定义

系统输入可以写成：

$$
\left(I_k, q_k, \dot{q}_k, T_k^{ee}, u_k\right)
$$

其中：

- $I_k$：图像观测。
- $q_k$：关节位置。
- $\dot{q}_k$：关节速度。
- $T_k^{ee}$：末端位姿，可以是 $4 \times 4$ 齐次矩阵，也可以是 $[x,y,z,q_x,q_y,q_z,q_w]$。
- $u_k$：控制输入或动作命令。

目标是预测接触相关变量：

$$
\hat{\phi}_k,\quad \hat{\lambda}_k
$$

其中 $\phi_k$ 是 gap function 或法向接触距离，$\lambda_k$ 是 LCP 中的接触变量。一个可能的定义是：

$$
\lambda_k =
\begin{bmatrix}
\gamma_k \\
\lambda_k^n \\
\lambda_k^t
\end{bmatrix}
$$

其中：

- $\gamma_k$：互补或松弛相关变量。
- $\lambda_k^n$：法向接触力或接触冲量。
- $\lambda_k^t$：切向接触力或摩擦相关变量。

## 4. 图像 Encoder 的双 Head 设计

核心网络可以写成：

$$
z_{\mathrm{img}} = \mathrm{Encoder}(I_k)
$$

这里的 encoder 可以是 DINOv3、CNN、ResNet、ViT 或其他视觉 backbone。之后接两个 head。

### 4.1 Head 1：Gap Function Head

Gap head 用于预测：

$$
\hat{\phi}_k
$$

输入可以是：

$$
\left(z_{\mathrm{img}}, q_k, \dot{q}_k, T_k^{ee}\right)
$$

输出为：

$$
\hat{\phi}_k =
g_\theta(I_k, q_k, \dot{q}_k, T_k^{ee})
$$

这个 head 的作用是让图像 encoder 显式学习接触距离函数。

### 4.2 Gap Function 标签构造

以首次接触帧 $k_c$ 作为锚点：

$$
T_c = (p_c, R_c)
$$

每一帧：

$$
T_k = (p_k, R_k)
$$

其中 $R_k$ 可以由 quaternion 转为旋转矩阵。

如果 ee pose 原点就是接触点，则 gap 标签可以构造为：

$$
\phi_k =
s \cdot e_n^\top R_c^\top (p_k - p_c)
$$

如果 ee pose 原点不是实际接触点，需要考虑工具 tip 偏置 $r_{\mathrm{tip}}$：

$$
\phi_k =
s \cdot e_n^\top R_c^\top
\left[
\left(p_k + R_k r_{\mathrm{tip}}\right)
-
\left(p_c + R_c r_{\mathrm{tip}}\right)
\right]
$$

其中：

- $e_n$：工具坐标系下的接触法向轴。
- $R_c$：接触帧的旋转矩阵。
- $s$：符号修正项，用来保证接触前 $\phi_k > 0$，接触帧 $\phi_{k_c}=0$。

训练 loss：

$$
\mathcal{L}_{\phi}
= \left\| \hat{\phi}_k - \phi_k \right\|_2^2
$$

这一步本质上是：用接触锚点 pose 构造 gap function 标签，再 fine-tune image encoder，让图像学会接触距离表征。

### 4.3 Head 2：Contact / Force Head

Contact head 用于预测 LCP 接触变量：

$$
\hat{\lambda}_k =
\begin{bmatrix}
\hat{\gamma}_k \\
\hat{\lambda}_k^n \\
\hat{\lambda}_k^t
\end{bmatrix}
$$

输入可以是：

$$
\left(z_{\mathrm{img}}, q_k, \dot{q}_k, u_k, T_k^{ee}, \hat{\phi}_k\right)
$$

输出为：

$$
\hat{\lambda}_k =
f_\theta(I_k, q_k, \dot{q}_k, u_k, T_k^{ee}, \hat{\phi}_k)
$$

这个 head 的作用是预测接触力或接触冲量，同时保留图像 latent feature，用来隐式表征材料、摩擦、接触状态、变形等无法直接显式获得的信息。

也就是说，图像不只是为了预测 $\phi_k$，还用于隐式表达：

$$
\mu,\quad k,\quad c,\quad \text{contact mode},\quad \text{local geometry}
$$

这些机器人状态无法直接提供的接触对象性质。

## 5. 为什么不能完全抛弃图像

机器人状态：

$$
q_k,\quad \dot{q}_k,\quad T_k^{ee}
$$

只能描述机器人在哪里、怎么动，但不能完整描述：

- 接触对象是什么材料。
- 局部接触面是否变形。
- 摩擦系数 $\mu$。
- 接触刚度 $k$。
- 阻尼 $c$。
- 是否发生滑动。
- 接触区域形状。
- 表面粗糙度。

所以图像 encoder 的作用有两层。

显式作用：

$$
\hat{\phi}_k = g_\theta(\cdot)
$$

也就是预测 gap function。

隐式作用：

$$
z_{\mathrm{img}} \rightarrow \hat{\lambda}_k
$$

也就是让网络通过图像特征间接利用材料、摩擦、变形、接触模式等信息。

## 6. PINN 的核心

PINN 的核心不是某种固定网络结构，而是把物理方程或物理约束写进 loss。

普通端到端模型：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{data}}
$$

PINN 或 physics-informed model：

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{data}}
+ \mathcal{L}_{\mathrm{physics}}
$$

在这个问题里，physics loss 可以来自：

- gap function 监督。
- 法向互补约束。
- 非负约束。
- 摩擦锥约束。
- 末端空间动力学约束。
- 完整 LCP 动力学残差。

## 7. 分层实验计划

不需要一开始就完整展开：

$$
A,\ B,\ D,\ E,\ F,\ H,\ c
$$

因为完整 LCP 需要：

$$
M(q),\quad C(q,\dot{q}),\quad J(q),\quad J_n,\quad J_t,\quad \mu,\quad \phi(q)
$$

如果这些量拿不到或不准确，强行加入反而会让 physics loss 变成错误约束。因此建议做一个由浅入深的消融实验体系。

### Level 0：纯端到端 Baseline

输入：

$$
\left(I_k, q_k, \dot{q}_k, T_k^{ee}, u_k\right)
$$

输出：

$$
\hat{\lambda}_k
$$

Loss：

$$
\mathcal{L}
= \mathcal{L}_{\lambda}
$$

其中：

$$
\mathcal{L}_{\lambda}
=
\left\|
\hat{\lambda}_k - \lambda_k^{gt}
\right\|_2^2
$$

作用：作为不加物理约束的基线。

### Level 1：时序自回归模型

输入一段历史：

$$
\left(
I_{k-m:k},
q_{k-m:k},
\dot{q}_{k-m:k},
T_{k-m:k}^{ee}
\right)
$$

输出：

$$
\hat{\lambda}_{k+1}
$$

或：

$$
\hat{\phi}_{k+1},\quad \hat{\lambda}_{k+1}
$$

可选网络：

- GRU
- TCN
- 1D CNN
- Transformer encoder
- 小型 MLP + history window

Loss：

$$
\mathcal{L}
=
\mathcal{L}_{\lambda}
+
\mathcal{L}_{\mathrm{smooth}}
$$

其中：

$$
\mathcal{L}_{\mathrm{smooth}}
=
\left\|
\hat{\lambda}_{k+1} - \hat{\lambda}_k
\right\|_2^2
$$

作用：验证接触力预测是否需要历史信息，而不是只依赖单帧图像。

### Level 2：加入 Gap Function Head

网络输出：

$$
\hat{\phi}_k,\quad \hat{\lambda}_k
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{\lambda}
+
w_\phi \mathcal{L}_{\phi}
$$

其中：

$$
\mathcal{L}_{\phi}
=
\left\|
\hat{\phi}_k - \phi_k
\right\|_2^2
$$

作用：验证显式学习 gap function 是否能提升接触力预测。

### Level 3：加入法向互补约束

接触法向互补关系：

$$
0 \leq \lambda^n \perp \phi \geq 0
$$

等价于：

$$
\lambda^n \geq 0,\quad \phi \geq 0,\quad \lambda^n \phi = 0
$$

互补 loss：

$$
\mathcal{L}_{\mathrm{comp}}
=
\left\|
\hat{\lambda}_k^n \hat{\phi}_k
\right\|_2^2
$$

非负约束：

$$
\mathcal{L}_{\mathrm{pos}}
=
\left\|
\mathrm{ReLU}(-\hat{\lambda}_k^n)
\right\|_2^2
+
\left\|
\mathrm{ReLU}(-\hat{\phi}_k)
\right\|_2^2
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{\lambda}
+
w_\phi \mathcal{L}_{\phi}
+
w_{\mathrm{comp}} \mathcal{L}_{\mathrm{comp}}
+
w_{\mathrm{pos}} \mathcal{L}_{\mathrm{pos}}
$$

作用：验证 LCP 中最核心的接触互补关系是否能让预测更物理一致。

### Level 4：加入摩擦锥约束

如果网络预测切向力 $\hat{\lambda}_k^t$，则加入摩擦锥约束：

$$
\left\|
\hat{\lambda}_k^t
\right\|_2
\leq
\mu \hat{\lambda}_k^n
$$

对应 loss：

$$
\mathcal{L}_{\mathrm{fric}}
=
\left\|
\mathrm{ReLU}
\left(
\left\| \hat{\lambda}_k^t \right\|_2
-
\mu \hat{\lambda}_k^n
\right)
\right\|_2^2
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{\lambda}
+
w_\phi \mathcal{L}_{\phi}
+
w_{\mathrm{comp}} \mathcal{L}_{\mathrm{comp}}
+
w_{\mathrm{pos}} \mathcal{L}_{\mathrm{pos}}
+
w_{\mathrm{fric}} \mathcal{L}_{\mathrm{fric}}
$$

作用：约束切向力不能超过库仑摩擦上限，提高切向力预测的物理合理性。

其中 $\mu$ 可以有三种处理方式：

固定经验值：

$$
\mu = \mu_0
$$

可学习标量：

$$
\mu = \mathrm{Softplus}(\rho)
$$

图像条件预测：

$$
\hat{\mu}_k =
h_\mu(z_{\mathrm{img}}, \mathrm{state})
$$

### Level 5：末端空间接触模型

在没有完整 Franka Jacobian 或 dynamics 的情况下，可以先不使用完整关节空间 LCP，而是在末端空间建立简化接触模型。

例如：

$$
\dot{\phi}_k
\approx
\frac{\phi_{k+1} - \phi_k}{\Delta t}
$$

软接触模型：

$$
\lambda_k^n
\approx
k[-\phi_k]_+
+
c[-\dot{\phi}_k]_+
$$

对应 loss：

$$
\mathcal{L}_{\mathrm{soft}}
=
\left\|
\hat{\lambda}_k^n
-
\hat{k}[-\hat{\phi}_k]_+
-
\hat{c}[-\dot{\hat{\phi}}_k]_+
\right\|_2^2
$$

作用：在不依赖完整 $J(q), M(q), C(q,\dot{q})$ 的情况下，引入可观测的末端空间物理约束。

这适合软材料、凝胶、血栓、柔性接触等场景。

### Level 6：完整 LCP-PINN

如果后续能获得 Franka 的：

$$
M(q),\quad C(q,\dot{q}),\quad J(q)
$$

例如通过：

- libfranka
- Pinocchio
- URDF
- Drake
- MuJoCo
- 控制器接口

就可以使用完整 LCP 展开。

状态更新：

$$
\hat{s}_{k+1}
=
A s_k
+
B u_k
+
D \hat{\lambda}_k
+
d
$$

约束变量：

$$
\hat{y}_k
=
E s_k
+
F \hat{\lambda}_k
+
H u_k
+
c
$$

动力学残差：

$$
\mathcal{L}_{\mathrm{dyn}}
=
\left\|
s_{k+1}^{gt}
-
\hat{s}_{k+1}
\right\|_2^2
$$

LCP 残差：

$$
\mathcal{L}_{\mathrm{lcp}}
=
\left\|
\hat{\lambda}_k \odot \hat{y}_k
\right\|_2^2
+
\left\|
\mathrm{ReLU}(-\hat{\lambda}_k)
\right\|_2^2
+
\left\|
\mathrm{ReLU}(-\hat{y}_k)
\right\|_2^2
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{\lambda}
+
w_\phi \mathcal{L}_{\phi}
+
w_{\mathrm{dyn}} \mathcal{L}_{\mathrm{dyn}}
+
w_{\mathrm{lcp}} \mathcal{L}_{\mathrm{lcp}}
$$

作用：验证完整 LCP 动力学展开是否进一步提升预测精度和物理一致性。

