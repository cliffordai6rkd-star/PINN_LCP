# CaRS-WM 少步 Flow 蒸馏

独立入口使用原始教师 checkpoint 的 `model`（EMA 部署权重）和 normalizer。教师始终冻结，以 FP32、64 步 Heun 生成监督。学生保留教师 GRU、hidden_dim、Flow blocks、输出模态和所有数据时间契约；每个 Euler 步只调用一次 Flow decoder，预测该区间的平均变化。训练使用同一真实 history/action 和每次访问重新采样的一份共享高斯噪声，不回填预测状态到 history。

先把 YAML 中 `teacher_checkpoint_path` 改为**原始教师** EMA checkpoint 的绝对路径。直接 S4：

```bash
PYTHONPATH=. python train/trainer/contact_world_model_distill_train.py -c config/train_cfg/contact_world_model_distill_s4.yaml
```

可选先训练 S8，再将 S8 的 `checkpoints/best_terminal/step_*_mse_*.pt` 填到 S4 YAML 的 `student_init_checkpoint_path`。S2 必须先将已训练 S4 的对应路径填入 S2 YAML：

```bash
PYTHONPATH=. python train/trainer/contact_world_model_distill_train.py -c config/train_cfg/contact_world_model_distill_s8.yaml
PYTHONPATH=. python train/trainer/contact_world_model_distill_train.py -c config/train_cfg/contact_world_model_distill_s4.yaml
PYTHONPATH=. python train/trainer/contact_world_model_distill_train.py -c config/train_cfg/contact_world_model_distill_s2.yaml
```

所有阶段的 `teacher_checkpoint_path` 都必须是同一份原始教师文件。入口核对其 SHA256、完整 model/data/action/time contract 与 normalizer；跨阶段只允许 Flow 积分步数不同。只加载上一阶段 EMA `model` 权重初始化，监督教师不会换成上一阶段学生。YAML 只覆盖蒸馏与优化配置，episode 划分和随机种子继承教师配置。若教师本身没有独立验证划分，须在 `train` 下设置 `val_episode_indices` 或 `val_ratio`，并保持 `split_mode: episode`。

每个 optimizer step 的损失包含局部平均速度、完整 chunk 的同噪声输出、q 的物理时间相邻差分和可选 contact KL/CE。权重均在 `distillation` 下；`q_d1_weight` 与 `q_d2_weight` 设为 0 可关闭。D1(q)[t] = q[t+1] - q[t]，D2(q)[t] = q[t+2] - 2q[t+1] + q[t]；两者在教师原有 q 归一化尺度及模型输出帧率下计算，不除以物理 dt。学生局部起点概率按 optimizer step 从 0 增至配置上限。训练 dropout 关闭以匹配部署，但完整学生积分链保留梯度。

每次 `checkpoint_every_steps` 在固定验证噪声上运行实际 K 步学生、64 步教师和同样 K 次调用的普通教师 Euler baseline。指标包括按模态同噪声误差、真实标签误差、Energy Score、spread、q 差分误差、三个接触阶段及单样本延迟。按完整输出 `val_terminal_mse` 在 `checkpoints/best_terminal` 留存最佳学生 EMA，`checkpoints/latest.pt` 用于恢复训练；训练噪声不固定。

边缘端必须用 `ContactWorldModelStudent(checkpoint["config"])` 构建模型，先调用 `validate_checkpoint(checkpoint)`，再严格加载 checkpoint 的 `model` 和原 normalizer。不能把学生 checkpoint 交给普通 `ContactWorldModel` 加载。这里的 `predict(..., steps=2/4)` 才是部署路径。
