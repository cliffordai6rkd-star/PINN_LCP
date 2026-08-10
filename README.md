# Torque World Model

当前仓库有三类训练任务：FACTR2 风格的自由空间总力矩拟合 `tau_free`、RNEA 后的
背景 residual 拟合 `tau_f`，以及使用冻结 `tau_f` 的 `TorqueWorldModel`。本 README
描述训练阶段已经实现的真实契约，不把尚未实现的在线力矩控制、候选轨迹聚合或
安全阈值算进来。

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

## 训练命令速查

所有训练脚本都支持从仓库根目录直接运行，统一使用 `-c/--config`。当前 FACTR2
自由空间消融的监督合同是：

```text
[q, dq, delta_q]_(t-49:t) -> tau_t
sample split + normalized MSE + lr=1e-3 + raw-model early stopping + no EMA
```

新增的七轴物理 MSE 消融保持输入和网络输出归一化，但在反向传播前将 torque
恢复为 Nm，分别计算七个关节的 MSE，再等权平均：

```text
MSE_j = mean_sample((tau_pred_nm[:, j] - tau_target_nm[:, j])^2)
loss  = mean_j(MSE_j)
```

LSTM 七轴等权物理 MSE（当前推荐先跑的基线）：

```bash
python train/trainer/tau_free_sequence_train_v2.py \
  -c config/train_cfg/tau_free_sequence_v2_sample_lstm_physical_mse_noema.yaml
```

LSTM（本轮实际训练）：

```bash
python train/trainer/tau_free_sequence_train_v2.py \
  -c config/train_cfg/tau_free_sequence_v2_sample_lstm_noema.yaml
```

GRU（配置已准备，本轮不启动）：

```bash
python train/trainer/tau_free_sequence_train_v2.py \
  -c config/train_cfg/tau_free_sequence_v2_sample_gru_noema.yaml
```

TCN（本轮实际训练）：

```bash
python train/trainer/tau_free_sequence_train_v2.py \
  -c config/train_cfg/tau_free_sequence_v2_sample_tcn_noema.yaml
```

RNEA residual `tau_f`：

```bash
python train/trainer/tau_f_sequence_train.py \
  -c config/train_cfg/tau_f_sequence.yaml
```

Torque world model：

```bash
python train/trainer/torque_world_model_train.py \
  -c config/train_cfg/torque_world_model.yaml
```

需要在终端观察并同时保存完整输出时，在命令末尾增加：

```bash
2>&1 | tee outputs/<run_name>/train.log
```

随机 `sample` 划分会让相邻的 50 帧窗口在训练集和验证集间共享原始帧。它适合当前
“所有运动分布都参与拟合，并用原始 validation loss 找过拟合拐点”的插值消融，但
不能作为跨 episode 泛化指标。严格泛化评估仍应使用 `episode` 或 purged split。

### 2026-08-10 sample/no-EMA 消融结果

两次训练都使用相同数据、seed、batch size、纯 MSE、`lr=1e-3`、plateau scheduler
和 `min_delta=1e-5 / patience=20` 的原始模型 early stopping：

| 分支 | 自动停止 | 最佳 epoch | `val_loss` | `val_mae_nm` | torque error P95 | 最佳 checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LSTM | 231 epochs | 212 | 0.008373 | 0.039641 Nm | 0.161367 Nm | `factr2_lstm_h50_sample_mse_lr1e3_noema/checkpoints/epoch_212_val_loss_0.008373.pt` |
| TCN | 154 epochs | 143 | 0.010779 | 0.045303 Nm | 0.178643 Nm | `factr2_tcn_h50_sample_mse_lr1e3_noema/checkpoints/epoch_143_val_loss_0.010779.pt` |

输出目录均位于 `outputs/tau_free_sequence/`，完整终端输出保存在各 run 的
`train.log`。在这个同分布插值合同下，LSTM 的整体 MAE 和每个关节 MAE 都优于当前
TCN；该结论不能外推为跨 episode 泛化结论。GRU 配置已准备，但本轮没有训练。

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
tau_id_raw      = RNEA(q_measured, dq_measured, ddq_RTS)
tau_id_filtered = causal_lowpass(tau_id_raw)
tau_f           = tau_measured_filtered - tau_id_filtered
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
q,dq measured + ddq_RTS           -> Pinocchio RNEA       -> tau_id_raw
tau_id_raw                         -> matched causal LPF   -> tau_id_filtered
tau measured filtered - tau_id filtered                  -> tau_f label
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
| `teleop/tau_id_rts` | 使用 `ddq_RTS` 的原始逆动力学力矩 |
| `teleop/tau_id_rts_filtered` | 与实测 torque 相同的 10 Hz 因果滤波结果 |
| `teleop/tau_f_cal` | `tau_filtered - tau_id_filtered`，供 residual 网络训练 |
| `teleop/q_rts`, `teleop/dq_rts` | 平滑状态诊断，不作为 residual 网络输入 |
| `teleop/ddq_kf_causal` | forward KF 对照结果，不含 RTS 未来信息 |
| `teleop/ddq_rts_std` | 后验加速度标准差，用于筛查低置信区间 |

`position_std/velocity_std` 越大，滤波器越不信任观测；`jerk_std` 越大，越允许加速度
快速变化、平滑越弱。调参应在 episode 级留出集上同时比较轨迹重构、`ddq` 尖峰、
RTS 后验不确定度和 `tau_f` 分布，不能只按训练 loss 选择。RTS 使用未来帧，只允许
离线构建标签；部署若需要加速度，只能独立运行因果 KF。

输入 HDF5 的 `tau_follower` 已经是 10 Hz 因果一阶低通结果。标签构建器会检查其
`causal/lowpass/zero_phase/cutoff/median_window` 属性，并用同一时间戳和参数独立过滤
`tau_id_rts`；属性不匹配时直接拒绝构建，防止重新引入滤波相位差。

要量化在线 causal KF 与离线 RTS 的 estimator mismatch，可在已构建的标签目录上运行：

```bash
python -m data_process.tool.analyze_causal_rts_rnea_gap \
  --config config/data_process/offline_tau_labels.yaml \
  --output outputs/diagnostics/causal_rts_rnea_gap.json
```

脚本同时计算 raw RNEA gap 和实际在线公式对应的 matched-filter gap；后者为
`LPF(RNEA_causal)-LPF(RNEA_RTS)`，并作为主报告输出逐关节和整体的 bias、MAE、
RMSE、绝对误差 P95/P99/max，以及超过 `0.05/0.1/0.2 Nm` 的样本比例。
默认跳过每个 Kalman segment 的前 49 帧，与 50 帧 torque 网络开始有效的时刻一致；
可用 `--warmup-frames 0` 单独检查 Kalman 启动瞬态。

torque 网络采用 NEXT 风格的独立滑动窗口，并支持 `lstm`、`gru` 和严格左因果
`tcn`。默认 TCN 使用 kernel 2 和 dilation `[1,2,4,8,16,18]`，感受野精确覆盖
50 帧，并用 current-state skip 保留快速通道；LSTM/GRU checkpoint 仍保持兼容。
输入为 50 步历史 `q,dq,delta_q`，其中
`delta_q=q_cmd-q` 是因果可得的低层控制误差；网络不输入 `ddq` 或瞬时 `tau`。
若要严格执行只用 `q,dq` 的消融，需同时从 `model.inputs` 和
`normalize_lowdim_keys` 删除 `delta_q`。旧 checkpoint 学习的是
`tau_filtered - tau_id_raw`，与新的 `matched_causal_torque_filter_v1` target contract
不兼容，必须用新标签重训；nero_ws 会在加载时拒绝缺少该 contract 的 checkpoint。
旧脚本 `repair_nero_dynamics_h5.py` 仍保留用于
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
已指向 `tau_refinement_rts_matched_labels`：

```bash
python -m data_process.tool.h5_2_lerobotev3 \
  --config config/shape_meta/shape_mata_tau_f.yaml
```

该配置将 `teleop/tau_f_cal` 映射为 `observation.tau_f`，并写入独立的
`data/train_episode/tau_refinement_matched_lerobotv3`。`tau_f_sequence` 已显式指向这份
新数据；`tau_free_sequence_v2` 的总力矩标签语义没有变化，仍可使用原数据目录。

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

先训练 `tau_f`，或在 WM 配置中填写已有 checkpoint。训练脚本支持直接入口，不依赖
editable install 是否已经注册 console script：

```bash
python train/trainer/tau_f_sequence_train.py \
  -c config/train_cfg/tau_f_sequence.yaml
```

FACTR2 自由空间总力矩拟合与 residual 训练不是同一个 target：

```text
tau_free_sequence_train_v2.py: target=tau，直接拟合无接触实测总电机力矩
tau_f_sequence_train.py:       target=tau_f，拟合 RNEA 后背景 residual
```

当前 `sample_lstm/gru/tcn_noema` 三份配置使用随机 `sample` 划分、纯 MSE、`1e-3`
学习率和原始模型 early stopping，并显式关闭 EMA。`tau_free_sequence_v2.yaml` 则保留
完整 episode + Jacobian-aware wrench 辅助项的跨轨迹基线；两类结果不能直接混为同一
评估合同。

`tau_free_sequence_v2_sample_lstm_physical_mse_noema.yaml` 是单独的物理单位消融，
只改变 `TauFreeSequenceTrainerV2` 的 torque objective，不修改 `TauFTrainer` 或
`BaseTrainer`。模型仍预测归一化 torque，但 loss 使用反归一化后的 Nm：

```yaml
loss:
  torque_loss_space: physical_nm
  joint_weight_mode: equal
  joint_weights: null
```

`joint_weight_mode` 支持：

| 模式 | 七轴权重 | 用途 |
| --- | --- | --- |
| `equal` | 全部为 1 | 推荐第一版；相同绝对 Nm 误差受到相同惩罚 |
| `mean_abs` | 正比于训练集 `mean(abs(tau_j))` | 更强调长期力矩较大的轴 |
| `max_abs` | 正比于训练集 `max(abs(tau_j))` | 更强调量程大的轴，但对脏尖峰敏感 |
| `manual` | 显式 `joint_weights: [w1,...,w7]` | 固定权重消融 |

自动模式只使用 train split 统计，并把七个权重归一化到均值 1；解析后的权重会写入
checkpoint 配置的 `loss.resolved_joint_weights`。这里不使用带符号
`mean(tau_j)`，因为正负力矩会抵消。当前完整数据的诊断值约为：

```text
mean_abs: [0.401, 3.835, 0.305, 2.092, 0.100, 0.115, 0.153]
max_abs:  [1.028, 3.195, 0.514, 1.690, 0.163, 0.155, 0.255]
```

因此 `mean_abs` 会强烈偏向 J2/J4，`max_abs` 又可能受边界脏数据影响；建议先以
`equal` 建立物理 MSE 基线，再根据逐轴 `val_tau_rmse_nm_j1...j7` 做第二轮权重消融。

`tau_f` 使用普通七轴归一化 MSE。带物理辅助项的 `tau_free_sequence_v2.yaml` 仍以
该 MSE 为主目标，并加入权重 `0.01` 的 Jacobian-aware wrench MSE；实验中的旧
`0.1` 权重会损害 torque 精度：

```text
tau_ext_pred = tau_measured - tau_free_pred
wrench_ext   = damped_solve(J(q).T * wrench_ext = tau_ext_pred)

loss = MSE(tau_free_pred_norm, tau_measured_norm)
     + 0.01 * MSE(wrench_ext / [1N, 1N, 1N, 0.1Nm, 0.1Nm, 0.1Nm], 0)
```

辅助项没有引入新的标签，只重排关节误差在 Jacobian 敏感方向上的代价。Jacobian
使用闭合指尖中心 `gripper_tcp`、LOCAL 坐标系和阻尼 `0.02`，并在训练
初始化时预计算到现有 tensor cache。训练会记录：

```text
val_tau_rmse_nm
val_wrench_rmse_scaled
val_wrench_force_rmse_n
val_wrench_moment_rmse_nm
val_wrench_force_norm_p95_n
```

带 wrench 辅助项的 `tau_free_sequence_v2.yaml` 使用 `1e-3` 初始学习率，checkpoint
按完整 validation split 上的
`val_wrench_force_norm_p95_n` 排序，避免平均 loss 掩盖假接触尖峰；plateau scheduler
和 early stopping 使用不含辅助项的 `val_tau_mse`。最终结果仍应在新采集的无接触
轨迹上复核。

### 8.1 Torque 序列推理可视化

`tau_f` 和 `tau_free` 分别有独立入口。脚本直接恢复 checkpoint 中的模型配置、
history horizon 和 normalizer，不会用可视化数据重新拟合统计量。只有拥有完整 history
的目标点会参与推理；`H=50` 时 episode 前 49 帧不会以补零方式进入指标。

`tau_f` 从 episode 起点重放与 Nero 一致的因果链路：Kalman `ddq`、RNEA、10 Hz
因果 `tau_id` 低通、`tau_id_filtered + tau_f_pred - tau_measured`，最后通过阻尼
Jacobian 求解 `wrench_ext`。`tau_free` 直接使用
`tau_measured - tau_free_pred` 映射 wrench。即使指定 `--start-frame`，因果状态仍从
episode 起点预热，绘图才从指定帧开始。

可视化 `tau_f`：

```bash
python -m data_process.tool.visualize_tau_f_inference \
  --checkpoint outputs/tau_f_sequence/next_gru_h50_matched_tau_id_filter/checkpoints/epoch_339_train_eval_loss_0.005293.pt \
  --episode-index 0 \
  --device cuda:0
```

可视化 `tau_free`：

```bash
python -m data_process.tool.visualize_tau_free_inference \
  --checkpoint outputs/tau_free_sequence/ep4/checkpoints/epoch_189_val_loss_0.046114.pt \
  --episode-index 0 \
  --device cuda:0
```

未指定 `--episode-index` 时默认处理整份数据；该参数可以重复指定以仅处理若干 episode。
`--all-episodes` 保留为显式处理整份数据的写法。`--start-frame`
和 `--end-frame` 按 episode 内相对帧号裁剪评估区间。绘图默认保留全部推理点；只有
显式传入 `--max-plot-points` 时才下采样。默认输出分别位于
`outputs/inference_visualization/tau_f` 和 `outputs/inference_visualization/tau_free`，
每个 episode 包含：

```text
torque_label_vs_prediction.png
torque_prediction_error.png
torque_error_summary.png
tau_ext_rollout.png
wrench_ext_rollout.png
inference_data.npz
metrics.json
```

`metrics.json` 还记录 `tau_ext` 逐轴误差、末端力/力矩范数的 mean/P95/max。
`inference_data.npz` 保存 `tau_ext_nm` 和 `wrench_ext`；`tau_f` 额外保存因果 `ddq`、
raw/filtered `tau_id` 和 rollout 使用的测量力矩。
`--checkpoint` 也接受目录并选择文件名末尾 score 最低的 checkpoint，但不同 split 或
不同实验不能共用 checkpoint 目录；存在混合文件时应显式传入 `.pt` 文件。

训练唯一的世界模型：

```bash
python train/trainer/torque_world_model_train.py \
  -c config/train_cfg/torque_world_model.yaml
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
- [`model/tau_f_lstm.py`](model/tau_f_lstm.py)、[`model/tau_f_gru.py`](model/tau_f_gru.py)
  和 [`model/tau_f_tcn.py`](model/tau_f_tcn.py)：三个互相独立的 torque 时序编码分支；
- [`model/tau_f_sequence.py`](model/tau_f_sequence.py)：只保留公共输入/输出合同和显式
  模型工厂，不再实现具体时序结构；
- [`train/trainer/torque_world_model_train.py`](train/trainer/torque_world_model_train.py)：
  episode split、normalizer、physics cache 和训练循环。

当前未实现：Nero 在线 runtime、`tau` 下发安全限幅/变化率限制、`first/mean` 执行
策略、随机多候选轨迹和 action expert 网络本身。这些属于训练闭环验证完成后的部署
阶段，不能从当前训练代码的存在推断为已经可上机。
