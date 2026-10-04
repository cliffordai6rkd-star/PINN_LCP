# SCFM 迁移到 latent CARS-WM

这是独立的 **Flow 速度场蒸馏**路线。输入原始 `latent_carswm_lstm_v1` 世界模型 checkpoint，保留标准高斯 source、条件编码器、latent codec、q/tau/contact 解码头，更新全部 Flow 参数。默认导出 **4 步 Euler，NFE=4**。没有新增步长 embedding；不会调用条件高斯反演训练。现有 `posttrain_inverse_gaussian_latent.py --mode distill-only` 的终点/瞬时速度匹配基线仍是另一个实验。

## 来源与迁移范围

已阅读并对照 [SCFM 官方仓库](https://github.com/caitree/scfm)，固定来源 commit `41d435d36cfcdeb945cb9562ddb87ca5e5cd285a`，重点是 `trainer/flux_scfm.py` 和 `src/utils/train_utils.py`。原论文：[arXiv:2510.17858](https://arxiv.org/abs/2510.17858)。算法代码在 `model/pinn_model/scfm_distillation.py`，训练入口在 `scripts/posttrain_scfm_latent.py`。上游许可保存在 `third_party/SCFM_LICENSE`。

保留最终 dual-EMA target 构造和区间平均速度损失；图像侧的 Flux/VAE、文本条件、CFG、LoRA、图像分辨率动态 shift 和图像评估器没有迁入。小型世界模型直接训练 Flow 的全部权重，使用冻结的真实轨迹 codec latent。当前适配的是 latent 分支，旧的 `carswm_v9` GRU checkpoint 不兼容。

| 设置 | 官方 Flux 启动脚本/代码 | 本迁移默认 |
| --- | --- | --- |
| teacher 指导比例 | 0.4 | 0.4 |
| 两段合并 | `t_skip=2` | 固定 2 段 |
| teacher target 网格 | 32 个 Euler 区间 | 32 个 Euler 区间 |
| self target 网格 | 2/4/8/16 区间 | 2/4/8/16 区间 |
| 普通 FM anchor 比例 | 启动脚本为 0；parser 默认 0.6 | 0，可单独消融 |
| fast EMA decay | 代码为 0；论文为 0.99 | 0，可改为论文设置 0.99 |
| slow EMA decay | 0.999 | 0.999 |
| 时间 grid shift | 通常随机 2.5–4.5，另有分辨率分支 | 1，匹配 WM 的均匀部署网格 |
| 学生权重 | Flux LoRA | 完整 Flow 参数；导出 raw student |

这里显式保留了代码与论文的区别，默认配置不应称作逐项复现 Flux 实验。两个 EMA 是独立冻结快照；条件 token 只编码一次，快照使用各自的 Flow 投影。避免上游为省显存进行的参数替换，也修正了启用 FM anchor 后上游 shortcut 子批次索引混用的问题。

## 损失与时间方向

我们的生成时间为 `s=0` 噪声、`s=1` 数据；Flux 原实现的 noise fraction `sigma` 方向相反。先在 sigma 上应用 shift，再换成 `s=1-sigma`，同时将速度方向换为 `data-noise`。这里的时间不是机器人动作轨迹的物理时间。

对真实未来的归一化 latent `z_data` 和高斯噪声 `epsilon`，从

```text
z_s = (1-s)*epsilon + s*z_data
```

开始，取相邻两个区间长度 `h1,h2`：

```text
v1 = teacher(z_s, s, c)       # teacher branch
  or fast_EMA(z_s, s, c)      # self branch
z_mid = z_s + h1*v1
v2 = slow_EMA(z_mid, s+h1, c)
v_target = stop_gradient((h1*v1+h2*v2)/(h1+h2))
loss = mean((student(z_s,s,c)-v_target)^2)
```

学生不接收 `h1/h2`，学习跨不同区间的近似一致速度。非均匀网格不能简单把两个速度等权平均。若启用 anchor，一部分窗口用独立随机时间和普通 FM target `z_data-epsilon`，其余窗口照常构建 shortcut target；所有窗口按上游普通 MSE 等权，不再乘 dataset 的 contact importance weight。每个 optimizer update 后更新 fast/slow EMA；固定老师始终不更新。训练关闭 dropout，保持训练和部署速度场一致。

## 训练

在原有世界模型环境、仓库根目录运行。需要原始 Flow checkpoint（codec 已完成、保存了 normalizer 和 episode split hashes）以及其配置所指的 LeRobot v3 数据；原数据根目录需在当前机器可用。不得输入条件高斯或此前蒸馏产生的 checkpoint。

```bash
python scripts/posttrain_scfm_latent.py \
  --base-checkpoint /path/to/original_latent_flow.pt \
  --output runs/scfm_s4/student.pt \
  --config config/train_cfg/latent_cwm_scfm.yaml \
  --device cuda:0 --num-workers 4 \
  > runs_scfm_s4.log 2>&1
```

默认 CUDA BF16、1000 次更新、4 步 Euler。RTX 5090 使用已支持该 GPU 的现有 PyTorch/CUDA 环境。CPU 流程检查使用 `--device cpu --precision fp32`。`--updates` 和 `--steps` 可以覆盖 YAML；其他消融参数建议复制 YAML 后修改，以独立输出目录保存。

先完成正常数据规模实验；若要验证少样本能力，可以独立运行 `--few-shot-windows 10`。这表示 **10 个训练窗口，可能相互重叠，不是 10 个独立 episode**；仍需预训练老师/codec。窗口只从原训练 episode 中抽取，验证 episode 从不进入蒸馏训练，且不会重算归一化或 latent 统计。每次重新抽噪声、时间区间和 target，故不是仅做 10 次训练。

### 续训

```bash
python scripts/posttrain_scfm_latent.py \
  --base-checkpoint /path/to/original_latent_flow.pt \
  --resume runs/scfm_s4/student.pt \
  --output runs/scfm_s4/student.pt \
  --updates 3000 --device cuda:0 --num-workers 4
```

保存快/慢 EMA、优化器、随机状态、epoch/batch cursor，每 `save_every` 次更新原子替换输出。续训时允许增加总更新次数；其余设置、数据划分、原老师文件 SHA256、训练设备类型必须一致。没有改动数据增强流程：此入口要求原配置关闭 action augmentation，以保证配对验证与续训可重复。同一设备环境下恢复采样流；不承诺 CUDA 内核跨硬件/版本逐 bit 相同。

checkpoint 自带完整部署模型和训练状态，现有 `load_latent_checkpoint` 可直接加载 `model` 字段，不需要 fast/slow EMA 或老师参与推理。记录在 `scfm_posttrain`，不会伪装成条件高斯实验。导出的是最后一次更新的 raw student，没有自动用 validation 选择最优模型；保留不同训练预算的独立输出，防止后续续训覆盖待比较的候选。

## 性能验证

训练前、每 `validate_every` 次更新及训练结束，固定 held-out 窗口和配对噪声比较：

1. 原老师的 `reference_steps` 与它原来的 solver；默认 32 步，Heun 时为 **64 NFE**。
2. 原权重 4 步 Euler，**4 NFE**，分离单纯减少步数的影响。
3. SCFM 权重 4 步 Euler，**4 NFE**。

`teacher_min/max_steps` 是 target 构造的 Euler 网格，`reference_steps` 是验证老师的采样步数，两者不同。默认固定随机抽取最多 `validation_batches * batch_size` 个原验证窗口，保存索引 hash，避免只测验证集的连续开头。各行报告真实数据上的 Energy Score、minADE/FDE、sample spread、90% coverage、contact NLL/Brier/全局 macro-F1、contact onset 帧误差，以及配对老师的 latent RMSE。q/tau 距离默认归一化；独立 physical 字段按保存统计还原，q 为 rad、tau 为 Nm（继承原数据单位）；不混合两种物理单位。截断 quantile 归一化无法可逆还原时省略 physical 字段。有限样本 coverage 和 onset 指标都只是诊断。

原模型性能对照必须加载原始 checkpoint，不能把蒸馏模型自身的 32 步结果当成原老师。可以重复评估已保存候选：

```bash
python scripts/posttrain_scfm_latent.py \
  --base-checkpoint /path/to/original_latent_flow.pt \
  --resume runs/scfm_s4/student.pt \
  --evaluate-only --output runs/scfm_s4/validation.json --device cuda:0
```

测实际 GPU 延迟：

```bash
python scripts/benchmark_latent_cwm.py \
  --checkpoint runs/scfm_s4/student.pt \
  --condition /path/to/normalized_condition_batch1.pt \
  --device cuda:0 --steps 1,2,4,8 --precision bf16 \
  --output runs/scfm_s4/latency.json
```

该 benchmark 的 reference 是 **同一候选** 的细步输出，只能用于延迟/数值诊断；训练入口的验证才使用原老师。部署默认 solver 由 checkpoint 决定为 Euler，配置脚本中其它指定 Heun 的调用需要明确改用 Euler，否则 4 步会变成 8 NFE。推理加载标准接口不变，未自动修改独立 xArm 硬件仓库的模型适配器。

SCFM 目标是减少采样离散误差，并不直接约束相邻重规划的物理模式一致性。还需要按 `docs/trajectory_jitter_evaluation.md` 比较同一绝对未来时刻的预测修订，保持控制频率、execute_steps、噪声策略及样本选择规则一致。条件高斯、SCFM 和旧蒸馏各自作为独立实验。真实任务成功率、多峰覆盖、力矩误差、重规划跳变以及 RTX 5090 上的端到端 p95 延迟均需实测；本次 CPU 合成测试不能证明性能不降或达到 100 Hz。
