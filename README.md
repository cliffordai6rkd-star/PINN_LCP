# Torque World Model

当前仓库只保留两条训练链：单独训练并在 WM 中冻结的 `tau_f` 网络，以及唯一的
`TorqueWorldModel`。本 README 描述的是训练阶段已经实现的真实契约，不把尚未实现
的在线力矩控制、候选轨迹聚合或安全阈值算进来。

世界模型要学习的是固定机器人、传感器和数据采集控制链下的条件闭环响应：

```text
高频历史 q, tau + 低频 action expert 的未来末端目标
                         |
                         v
              未来高频 q, tau, contact
```

训练完成后，预测的 `tau` 才会进入单独的部署控制链，并通过 Nero 的 MIT `t_ff`
接口下发。训练标签中的 `tau` 是电流估计的实测关节力矩；把它近似当作未来
`tau_cmd` 的监督是部署假设，不代表两者在动态过程中严格相等。

## 1. 问题定义

对高频 anchor `t`，定义：

```text
H_t = [q, tau]_(t-H+1:t)                  历史状态，H 步
A_t = [target_relative_pose]_(1:A)        held action 条件，A 个高频 token
Y_t = [q, tau, contact]_(t+1:t+T)         未来目标，T 步
```

模型表示：

```text
p_theta(Y_t | H_t, A_t)
```

三个对外配置的 horizon 彼此独立：

- `state_history_horizon = H`：历史高频 `q/tau` 长度；
- `action_chunk_horizon = K`：action expert 输出的低频 waypoint 数；
- `prediction_horizon = T`：WM 预测的未来高频长度。

注意 `K` 不等于模型看到的 action token 数 `A`。dataloader 会把 K 个低频 waypoint
零阶保持到高频时间线：

```text
A = round((K - 1) * high_fps / expert_fps) + 1
```

默认 `K=8`、`high_fps=80`、`expert_fps=5`，因此 action chunk 跨越 1.4 s，展开后
为 `A=113` 个高频 token。未来预测 `T=40` 只覆盖 0.5 s，二者不要求等长。

### 模型输入

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `q` | `[B,H,7]` | 归一化的历史关节位置 |
| `tau` | `[B,H,7]` | 归一化的历史实测关节力矩，单位量归一化前为 N*m |
| `target_relative_pose` | `[B,A,7]` | held 目标的 `xyz + quaternion(xyzw)` |
| `target_relative_pose_mask` | `[B,A]` | action token 是否处于当前 chunk 的有效时间范围 |

`dq`、`ddq`、wrench 和 contact 不进入条件编码器，只作为监督标签。这样输入端不会
引入数值微分噪声，同时仍可用它们约束预测轨迹。

### 模型输出

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `q_pred` | `[B,T,7]` | 未来关节位置 |
| `tau_pred` | `[B,T,7]` | 未来关节力矩 |
| `contact_logits` | `[B,T,1]` | 未来二值接触 logit |

物理单位下的结果由 loss 在反归一化后写入 `q_pred_physical`、
`tau_pred_physical`、`dq_pred_physical` 和 `ddq_pred_physical`。

## 2. 双频 action 条件

数据转换器以同一个 master clock 生成相位锁定的均匀时间线。默认转换配置中，每个
10 Hz 图像 anchor 行打包此前 8 个 80 Hz 状态样本。WM dataloader 再按
`expert_fps` 从示教的 `reference.ee_pose` 构造离线伪 action-expert chunk：

```text
图像 anchor / expert refresh r
  + inference_delay_s                   -> 第一个未来 waypoint
  + n / expert_fps, n = 0 ... K-1       -> K 个稀疏 waypoint
  -> 在 high_fps 时间线上零阶保持       -> A 个条件 token
```

训练阶段使用示教未来 EE pose 构造这个伪 action 标签；部署阶段应由低频、
force-aware action expert 输出相同坐标语义和长度的 chunk。WM 本身不读取图像。

一个 chunk 在下一次 expert 更新前持续占位。对每个高频 WM anchor，绝对 target
保持不变，但相对位姿会基于该时刻的实际 EE pose 重新计算。因此条件表达的是实时
更新的 target distance，不会通过递推相对增量累积位姿误差。

`inference_delay_s` 只移动第一个 action waypoint 的标签时间，不移动 `q/tau`
历史，也不移动 current EE pose。它应使用 action expert 的在线端到端延迟统计，
而不是凭训练 loss 反推。

`action_condition_mode` 支持：

- `relative_pose`：默认。计算 `T_current^-1 * T_target`，平移也会旋转到当前 EE
  坐标系；
- `absolute_pose`：直接使用基座坐标系下的 target `xyz + quaternion`。

`action_resample: pose` 对位置做线性插值、对四元数做 SLERP；随后才执行 waypoint
的零阶保持。四元数统一为 `xyzw`、单位化，并把等价的 `q/-q` 规范到同一半球。

## 3. 网络结构

```text
[q, tau] history -> state GRU -> state tokens -----------+
                                                         |
held pose targets -> action GRU -> action tokens --+     |
                                                   |     |
state tokens -- query --> cross-attention <--- action K/V|
                                                   |     |
                         condition memory <---------+-----+
                                   |
history trajectory source + Flow time s -> Flow decoder
                                   |
                         future q, tau, contact
```

cross-attention 的方向是有意固定的：历史 state token 是 query，未来 action token
是 key/value。它让当前动力学状态选择与自己相关的未来意图。attention 之后仍保留
完整 action token，未来每个 Flow token 还能再次查询 state/action memory，不会把
整个“下降、接触、滑动、释放”过程压成一个向量。

### History-to-future Flow Matching

当前 Flow source 不是高斯噪声。令 `X_0` 为与 T 对齐的历史轨迹：

- `H >= T`：取最近 T 个历史 `q/tau`；
- `H < T`：在左侧重复最早历史观测，保持已有样本的原始时间间隔；
- source 的 contact 通道置零。

target `X_1` 是未来 `[q, tau, contact_latent]`。训练时采样无量纲 Flow 时间
`s in [0,1]`：

```text
X_s      = (1 - s) * X_0 + s * X_1
v_target = X_1 - X_0
L_flow   = MSE(v_theta(X_s, s, condition), v_target)
```

这里的 `s` 只是 Flow 路径进度，不是物理时间戳，也不是 `inference_delay_s`。
推理时从同一个历史 source 用 Euler 或 Heun 积分到 `s=1`。

由于当前 source 对同一输入是唯一的，`predict()` 也是确定性的单轨迹预测。重复调用
不会产生有意义的多候选分布；候选均值需要后续显式加入随机 latent、随机残差 source
或 ensemble，不能只重复运行当前模型。

## 4. Loss 与动力学约束

总损失为：

```text
L = w_flow * L_flow
  + w_q * L_q + w_tau * L_tau
  + w_dq * L_dq + w_ddq * L_ddq
  + w_contact * L_contact
  + w_wrench * warmup(step) * L_wrench
```

其中：

- `L_q/L_tau`：Flow endpoint 对未来 q/tau 标签的 MSE；
- `L_dq/L_ddq`：只从预测 q 按 `sampling_dt` 可微差分，再与数据中的 dq/ddq
  标签计算 MSE；
- `L_contact`：由 wrench 标签经过迟滞阈值和连续帧规则得到的 BCE；
- `L_wrench`：预测 q/tau 经冻结 `tau_f`、局部 RNEA 和 Jacobian 阻尼求解后，与
  wrench 标签计算 MSE。

wrench 路径使用：

```text
tau_id_pred  = local_RNEA(q_pred, dq_pred, ddq_pred)
tau_ext_pred = tau_pred - tau_id_pred - tau_f_pred
tau_ext_pred = J(q_reference)^T * wrench_pred
```

`local_RNEA` 是在记录的未来状态附近缓存的一阶线性化，Jacobian 也来自该参考状态。
因此当前实现是可微的局部动力学约束，不是完整非线性 RNEA rollout。启用
`soft_contact_gate` 时，输出 wrench 还会乘预测接触概率。

## 5. 配置与调参

主配置位于
[`config/train_cfg/torque_world_model.yaml`](config/train_cfg/torque_world_model.yaml)。

### 时间与 horizon

| 参数 | 当前值 | 调整原则 |
| --- | ---: | --- |
| `high_fps` | 80 | 必须等于转换数据的真实高频采样率 |
| `state_estimator.sampling_dt` | 0.0125 | 必须满足 `sampling_dt = 1 / high_fps` |
| `expert_fps` | 5 | 设为实际 action expert 的稳定更新频率，计划范围为 3-5 Hz |
| `state_history_horizon` | 50 | 至少覆盖冻结 `tau_f` 的历史长度，并覆盖接触/摩擦迟滞 |
| `action_chunk_horizon` | 8 | 决定低频意图跨度，不要为匹配 T 而修改 |
| `prediction_horizon` | 40 | 由力矩控制所需预见时间和闭环误差共同决定 |
| `inference_delay_s` | 0.05 | 用在线 action expert 端到端延迟的统计值替换 |

先按物理时长选 horizon：

```text
history span    = (H - 1) / high_fps
prediction span = T / high_fps
action span     = (K - 1) / expert_fps
```

建议先固定 `80 Hz, H=50, K=8, T=40`，只比较 `expert_fps` 和实测
`inference_delay_s`；随后再做 H/K/T 消融。时间契约未对齐时，增大网络没有意义。

### 模型容量与 Flow 积分

| 参数 | 调整原则 |
| --- | --- |
| `hidden_dim` | 首要容量旋钮；必须同时被两组 attention head 数整除 |
| `state_layers/action_layers` | GRU 欠拟合时从 2 增至 3；小数据优先保持 1-2 |
| `attention_heads` | 一般保持 4；不要在 hidden 很小时堆 head |
| `flow_layers` | train/val 都高时增加；train 低而 val 高时减少或加 dropout |
| `flow_ffn_multiplier` | 显存敏感，通常保持 4 |
| `flow_inference_steps` | 只影响 `predict()` 的积分精度和延迟，不改变训练 loss |
| `flow_solver` | `heun` 更准但每步两次网络调用；`euler` 更快 |

部署前应在目标 GPU 上联合测试 `solver x steps`，例如 `euler: 4/8/12` 与
`heun: 4/8`，同时记录 tau RMSE、接触 onset 误差和单次推理时延。

### Loss 的推荐调节顺序

当前 YAML 是完整约束配置，`wrench_weight=0.5`。为了定位问题，建议复制配置做以下
分阶段实验，而不是一次同时调所有权重：

1. 数据基线：设 `wrench_weight=0`，先确认 `flow/q/tau/contact` 能稳定下降。
2. 导数约束：从较小的 `dq_weight/ddq_weight` 开始，二阶项应比一阶项更小；观察
   q 精度是否被过度平滑。
3. 物理约束：确认 `tau_f` checkpoint、URDF frame 和 Jacobian 方向正确后，从
   `wrench_weight=0.05` 或 `0.1` 开始，并保留 warmup。
4. 完整实验：只有在各残差已经标准化、梯度量级接近时，再尝试当前的 `0.5`。

具体判断：

- `q_loss` 降但 `dq/ddq_loss` 发散：先检查 `sampling_dt` 和标签单位，再降低
  `ddq_weight`；
- `tau_loss` 降但 `wrench_loss` 不降：检查 tau 符号、frame、`tau_f` normalizer
  和接触标签，不要先加大 `wrench_weight`；
- `contact_loss` 主导：检查正负样本比例、迟滞阈值和 `positive_class_weight`；
- train 降而 val 不降：先保持 episode split，缩小模型或增加 dropout；
- Flow loss 降但积分预测差：增加 validation/inference 积分检查，再比较 solver
  和 steps。

优化器建议从当前 `lr=1e-4`、`weight_decay=1e-4`、gradient clip 1.0 和 EMA
开始。显存不足先减 `batch_size`；只有在时间契约和 loss 比例稳定后再扩大模型。

## 6. 离线 residual 标签构建

`tau_f` 统一采用以下符号，不再沿用旧数据中的反号定义：

```text
tau_id = RNEA(q_measured, dq_measured, ddq_RTS)
tau_f  = tau_measured - tau_id
```

这个等式只有在无外部接触力矩的数据段上才是摩擦/未建模动力学标签；若把接触段直接
送入该流程，`tau_f` 会同时包含 `J^T F_ext`。因此 residual 时序网络的训练集必须使用
free-space episode，或在构建前按可靠的接触标注切除接触区间。

正式标签入口为
[`data_process/tool/build_offline_tau_labels.py`](data_process/tool/build_offline_tau_labels.py)，
配置位于
[`config/data_process/offline_tau_labels.yaml`](config/data_process/offline_tau_labels.yaml)。
它对每个关节使用状态 `x=[q,dq,ddq]` 的常加速度模型，以实测 `q,dq` 为观测：

```text
variable-dt Kalman forward filter -> RTS backward smoother -> ddq_RTS
q,dq measured + ddq_RTS           -> Pinocchio RNEA       -> tau_id
tau measured - tau_id              -> tau_f training label
```

状态转移和过程噪声都使用相邻实测时间戳的 `dt`，因此时间戳抖动不会被假定为固定
采样周期；NaN 观测由滤波器跳过，超过 `max_gap_s` 的采集断点会分段，RTS 不跨断点
传播未来信息。默认 `rnea_state_source: measured` 会保留实测 `q,dq`，只在 NaN 缺测
位置使用平滑状态补齐；离线标签中的加速度始终使用 `ddq_RTS`。

运行前先修改 YAML 中的输入/输出目录、URDF 和 `dq_sign`。当前示例输入已经做过
Nero 速度符号修正，因此 `dq_sign` 全为 `1`；读取原始固件速度时才设置为
`[-1,-1,-1,-1,-1,1,-1]`，且只能应用一次。先验证文件选择：

```bash
python -m data_process.tool.build_offline_tau_labels \
  --config config/data_process/offline_tau_labels.yaml \
  --dry-run
```

再用单个 episode 检查参数和单位：

```bash
python -m data_process.tool.build_offline_tau_labels \
  --config config/data_process/offline_tau_labels.yaml \
  --limit 1
```

确认 manifest 中 `dt_ms_*`、`ddq_abs_*`、`ddq_std_p99` 和 `tau_f_abs_*` 合理后，
删除试验输出或换一个空输出目录，再运行全量构建。工具从不修改输入文件；它先复制
episode，在临时文件中写完并 flush，最后原子替换到输出目录。已存在的输出默认拒绝
覆盖，只有显式传入 `--overwrite` 才会替换。

主要输出字段为：

| HDF5 字段 | 用途 |
| --- | --- |
| `teleop/ddq_follower` | RTS 平滑加速度，供 RNEA/监督使用 |
| `teleop/tau_id_rts` | 使用 `ddq_RTS` 的逆动力学力矩 |
| `teleop/tau_f_cal` | `tau - tau_id`，供 residual 时序网络训练 |
| `teleop/q_rts`, `teleop/dq_rts` | 平滑状态诊断，不作为 residual 网络输入 |
| `teleop/ddq_kf_causal` | forward KF 对照结果，不含 RTS 未来信息 |
| `teleop/ddq_rts_std` | 后验加速度标准差，用于筛查低置信区间 |

`position_std/velocity_std` 越大，滤波器越不信任观测；`jerk_std` 越大，越允许加速度
快速变化、平滑越弱。调参应在 episode 级留出集上同时比较轨迹重构、`ddq` 尖峰、
RTS 后验不确定度和 `tau_f` 分布，不能只按训练 loss 选择。RTS 使用未来帧，只允许
离线构建标签；部署若需要加速度，只能独立运行因果 KF。

residual 网络采用 NEXT 风格的独立滑动窗口：两层 LSTM/GRU（hidden 128）后接
`128 -> 256 -> 7` 的两层 MLP。每个 50 帧窗口都从零 recurrent state 开始，窗口或
batch 之间不传递 hidden state；`model.architecture` 可设为 `lstm` 或 `gru`，默认
使用 LSTM。输入为 50 步历史 `q,dq,delta_q`，其中
`delta_q=q_cmd-q` 是因果可得的低层控制误差；网络不输入 `ddq` 或瞬时 `tau`。
若要严格执行只用 `q,dq` 的消融，需同时从 `model.inputs` 和
`normalize_lowdim_keys` 删除 `delta_q`。由旧的 `tau_id-tau` 标签训练出的 checkpoint
与新符号不兼容，必须用新标签重训。旧脚本 `repair_nero_dynamics_h5.py` 仍保留用于
数据修复，但其因果差分加低通结果不应作为正式离线训练标签。

训练仍以归一化空间的 MSE 为优化目标，同时在 target 反归一化后报告以下 torque
指标：

```text
mae_nm     = mean_{sample,joint} |tau_pred - tau_target|
mae_nm_j1  = mean_sample |tau_pred[:, 0] - tau_target[:, 0]|
...
mae_nm_j7  = mean_sample |tau_pred[:, 6] - tau_target[:, 6]|
```

这些指标按样本数汇总，单位均为 Nm，并写入终端 epoch 日志、W&B、`loss_history`
和 checkpoint 的 `metrics`。验证指标对应 `val_mae_nm`、`val_mae_nm_j1...j7`；默认
`monitor_key: val_loss` 保持原有训练行为，也可以显式改成 `val_mae_nm`，让 scheduler、
early stopping 和 top-k checkpoint 按物理误差选择。判断是否进入标签噪声地板时应
同步调整 `early_stopping.min_delta` 到 Nm 尺度。判断是否进入标签噪声地板时应优先
查看 validation 的逐关节 MAE；整体 MAE 小于 0.1 Nm 仍可能掩盖某个关节明显偏差。

`val_loss` 不能唯一换算为 MAE。它至多给出归一化
`RMSE_norm=sqrt(val_loss)`；若额外假设误差是零均值高斯分布，可粗略估计
`MAE_norm ~= 0.798 * sqrt(val_loss)`。换回关节 `j` 的 Nm 还要乘该关节归一化尺度：
Gaussian 为 `std_j+eps`，quantile/limit 为对应区间宽度的一半。由于一个
`val_loss` 已混合所有关节及其不同尺度，正式判断应直接使用上述反归一化 MAE，不能
依赖这个近似。

`train.train_eval.enabled: true` 会在每个训练 epoch 后，用与 validation 相同的
`model.eval()` 和 EMA 选择重新遍历 train split，并额外记录：

```text
train_loss_online  # model.train()、参数在 batch 间持续更新
train_eval_loss    # epoch 末固定参数、model.eval()、无梯度
val_loss           # 与 train_eval_loss 相同模型模式
```

若 `train_eval_loss` 接近 `val_loss`，但明显低于 `train_loss_online`，差异通常来自
dropout 和在线平均；若 `train_eval_loss` 仍明显高于 `val_loss`，再检查 validation
episode 的运动难度、数据域比例和切分边界。开启该诊断会增加一次完整 train split
前向计算，但不会执行反向传播或优化器更新。

标签构建完成后，再把新 HDF5 转为 residual 时序网络使用的 LeRobot v3 数据。转换配置
已指向 `tau_refinement_rts_labels`：

```bash
python -m data_process.tool.h5_2_lerobotev3 \
  --config config/shape_meta/shape_mata_tau_f.yaml
```

该配置将 `teleop/tau_f_cal` 映射为 `observation.tau_f`，并写入独立的
`data/train_episode/tau_refinement_ped_lerobotv3`。现有 `tau_f_sequence`、
`tau_background_sequence` 和 `tau_free_sequence_v2` 配置仍指向旧的
`data/train_episode/tau_refinement_lerobotv3`，因此不会被这次转换隐式替换；完成数据
检查后，需要显式修改所需训练配置的 `dataloader.root` 才会使用新数据。

`observation.delta_q` 不会先在 H5 原始索引上相减。转换器分别把离散控制命令
`q_cmd` 用 `previous`（物理意义为 ZOH）保持到统一时间轴，把连续状态
`q_follower` 用 PCHIP 插值到同一时间轴，最后计算：

```text
delta_q(t) = q_cmd_previous(t) - q_follower_pchip(t)
```

`previous` 只选择满足 `source_timestamp <= target_timestamp` 的最近命令；目标时刻
早于第一条命令时转换会报错，不会用未来命令向前填充。

已经生成的 LeRobot v3 数据不会因转换器代码变化而自动重写。要让现有数据获得修正后
的 `delta_q`，必须重新执行上述转换；不要覆盖正在被训练或评估进程读取的数据目录。

## 7. 数据转换

双频 LeRobot 配置位于
[`config/shape_meta/data/insert_usb_dual_rate.yaml`](config/shape_meta/data/insert_usb_dual_rate.yaml)。
先确认其中的 `io.input`、`io.output`、URDF 路径和实际数据位置一致，再运行：

```bash
python data_process/tool/h5_2_lerobotev3.py \
  --config config/shape_meta/data/insert_usb_dual_rate.yaml
```

转换后，训练配置中的 `dataloader.root/repo_id` 必须指向同一份数据。WM 至少需要：

```text
observation.joint        [N, 8, 7]
observation.velocity     [N, 8, 7]
observation.acceleration [N, 8, 7]
observation.torque       [N, 8, 7]
observation.wrench_ext   [N, 8, 6]
reference.ee_pose        [N, 8, 7]
timing.high_timestamp_ns [N, 8]
timing.anchor_timestamp_ns [N]
```

这里的 `8` 是转换配置中每个 10 Hz LeRobot 行打包的 80 Hz 样本数，不是
`action_chunk_horizon`。dataloader 会先展平为严格均匀的高频 episode，再构造
H/A/T 窗口，并按 episode 切分 train/validation，避免相邻窗口泄漏。

## 8. 环境与训练命令

首次配置 GPU 环境：

```bash
bash setup.sh
conda activate pinn
```

`setup.sh` 会安装 CUDA 12.4 对应的 PyTorch、项目 editable package、Pinocchio
和测试依赖。安装后先运行：

```bash
python -m pytest -q tests
```

先训练 `tau_f`，或在 WM 配置中填写已有 checkpoint。直接使用 Python 模块入口，
不依赖 editable install 是否已经注册 console script：

```bash
python -m train.trainer.tau_f_sequence_train \
  --config config/train_cfg/tau_f_sequence.yaml
```

训练唯一的世界模型：

```bash
python -m train.trainer.torque_world_model_train \
  --config config/train_cfg/torque_world_model.yaml
```

安装项目后也可以使用等价的 console script：

```bash
pinn-train-tau-f --config config/train_cfg/tau_f_sequence.yaml
pinn-train-wm --config config/train_cfg/torque_world_model.yaml
```

正式启动前至少检查：

- `dataloader.root/repo_id` 指向已转换的数据；
- `physics.tau_f_checkpoint_path` 存在，且其历史 horizon 不大于 WM 的 H；
- `physics.pinocchio.frame_name` 与 URDF 一致；
- `train.device` 指向可用 GPU；
- `sampling_dt == 1/high_fps`。

checkpoint 默认写入 `outputs/torque_world_model/checkpoints/`，包含模型、优化器、
完整配置、EMA 元数据和只在训练 episode 上拟合的 normalizer。W&B 默认关闭，可在
`train.wandb.enabled` 打开。

没有 GPU 时，可以把 YAML 中 `train.device` 改为 `cpu`，并减小 batch/horizon 做
smoke test；完整的 Pinocchio cache 和正式训练仍建议使用 GPU。

## 9. 代码入口与当前边界

- [`data_process/world_model_dataset.py`](data_process/world_model_dataset.py)：双频
  dataloader、action chunk 锁存、延迟、SLERP 和 relative pose；
- [`data_process/offline_tau_labels.py`](data_process/offline_tau_labels.py)：变步长
  Kalman filter、RTS smoother 和 residual 标签符号契约；
- [`model/pinn_model/torque_world_model.py`](model/pinn_model/torque_world_model.py)：
  GRU、state-query/action-KV 和 history-source Conditional Flow Matching；
- [`train/torque_world_model_loss.py`](train/torque_world_model_loss.py)：Flow、轨迹、
  导数、contact 和 wrench loss；
- [`physics/nero_dynamics.py`](physics/nero_dynamics.py)：Pinocchio cache、冻结
  `tau_f`、局部 RNEA 和阻尼 wrench 求解；
- [`train/trainer/torque_world_model_train.py`](train/trainer/torque_world_model_train.py)：
  episode split、normalizer、physics cache 和训练循环。

当前未实现：Nero 在线 runtime、`tau` 下发安全限幅/变化率限制、`first/mean` 执行
策略、随机多候选轨迹和 action expert 网络本身。这些属于训练闭环验证完成后的部署
阶段，不能从当前训练代码的存在推断为已经可上机。
