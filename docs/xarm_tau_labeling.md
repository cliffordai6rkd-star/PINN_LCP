# xArm 世界模型训练与接触标注

以 `config/train_cfg/latent_cwm_erase_board_100hz_40step.yaml` 为训练时间配置基准。Contact WM 使用相同时间设置。

| 配置 | 状态历史 | 未来状态 | Action condition |
|---|---|---|---|
| Contact WM | 100 Hz，50 帧 | 100 Hz，32 帧 | 25 Hz，8 个动作 |
| Latent WM | 100 Hz，50 帧 | 100 Hz，32 帧 | 25 Hz，8 个动作 |

动作从数据原生的连续 `timing.action_index` 读取，相邻动作间隔约 40 ms；动作窗口从下一动作索引开始（`action_start_offset: 1`）。这 8 个动作覆盖约 0.32 秒，与 32 个 100 Hz 未来状态对应。配置文件保留历史的 `40step` 文件名，实际数值为 32/8。

## 启动训练

在项目根目录分别运行：

```bash
# Contact WM
.conda-env/bin/python -m train.trainer.contact_world_model_train \
  --config config/train_cfg/pretrain/xarm/cwm_erase_board_100hz_40step.yaml

# Latent WM：先 codec，再 flow
.conda-env/bin/python -m train.trainer.latent_contact_world_model_train \
  --config config/train_cfg/latent_cwm_erase_board_100hz_40step.yaml
```

两个训练入口自动加载同一个力矩教师，在归一化和采样之前生成/读取接触标签。

## 保留的模型与数据

- 力矩教师：`outputs/tau_free_sequence/xarm_offline_teacher/model.pt`
- 模型说明、哈希、划分及评估指标：同目录 `manifest.json`
- 对应的 90 段力矩缓存：`outputs/cache/wm_tau_labels/`
- 全量最终标签：`outputs/tau_labels/xarm_erase_board_100hz/labels_all.npz`
- 标签统计与训练输入检查：同目录 `summary.json`、`training_contract.json`

源数据位于：

```text
data/xram_cwm/erase_board_25hzcam_cwm_lerobot_v3-20261004T151905Z-1-001/erase_board_25hzcam_cwm_lerobot_v3
```

源 Parquet 和背景录制未修改。标签包含原始 100 Hz 时间戳、episode 索引、7 维 tau_ext、phase 和 valid_context。

## 标注规则

教师为动力学/摩擦先验 + 两层 BiLSTM + 两层全连接头，网络输入仅为 q、dq、delta_q。采用 10 Hz 零相位滤波和内部 50 Hz 推理，预测插值回原始 100 Hz 时间戳；WM 的状态和动作时间轴保持不变。

标注前按照数据中的 `meta/world_model_timeline.json`，在私有副本上还原 dq/tau 已有的 20 Hz 因果滤波。实测 tau 只用于构造残差，不输入力矩回归网络。

- `tau_ext = 同滤波实测 tau - tau_free`。
- 七轴 `sum(abs(tau_ext)) > 11.705677733654019` N·m：contact (2)。
- 接触前 1 秒：alignment (1)，contact 优先。
- 其余有效帧：free (0)。
- 缺少完整上下文：-1，排除监督。
- 两种 WM 均启用阶段加权采样，权重为 `free:alignment:contact = 1:5:5`。

90 段、152,444 帧中，free 65,329，alignment 10,882，contact 58,132，无效边缘帧 18,101。32/8 配置下两种 WM 均有 131,553 个有效窗口，标签逐帧一致。

独立背景测试中，教师对统一 5 Hz 目标的 RMSE 为 0.821 N·m；对原生 10 Hz 目标为 0.955 N·m，自由空间接触误报帧率约 0.60%。模型/预处理由验证集选择，接触阈值由独立校准块确定。

## 验证与清理

实际检查了两种 WM 各 180 个样本：action 为 `[180, 8, 7]`，未来 q 为 `[180, 32, 7]`；动作索引连续，相邻动作间隔中位数 39.999 ms，状态间隔中位数 9.999 ms。迁移后两种 WM 均命中全部 90 段缓存。

本轮实验用的消融权重、对照输出、曲线、临时 WM 检查点和重复训练配置已清理。保留可复用的训练/标注代码和回归测试，正式训练只依赖上面列出的最终文件。
