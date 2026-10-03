# 本轮离线验证记录（2026-10-01）

PINN `master`，Nero `pi0_wm`。GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB；Python 3.10，PyTorch 2.6.0+cu124。网络、action、horizon 和 MTC 基线均保留。没有真机连接、长期训练、git commit/push。

## 测试与实际 smoke

- PINN 本轮关键验证：46 passed（完整 Student 链/Teacher 无梯度、共享参数诊断不修改累计梯度、严格 dispatch、cache、数学/物理/contact/执行窗口、原 EMA/resume/checkpoint 回归）。
- Nero 本轮相关回归：91 passed，包括三个真实小架构的 checkpoint 加载和 scheduler 实际 `[d:d+E]` 索引。Nero 自己的 `.venv`（PyTorch 2.7.1+cu126）另跑新增 Student 部署关键测试：8 passed。
- 扩展 PINN 回归曾得到 93 passed / 2 failed。两个失败是已有测试引用不存在的 `config/train_cfg/cwm_insert_usb_50hz.yaml` 和 `cwm_all_50hz.yaml`，与本轮修改无关，未添加假配置或静默跳过。
- 真实原 Teacher 文件：`outputs/cwm_teacher_policy/insert_usb_50h_40step/checkpoints/step_00250000.pt`，正式 schema 10 / `carswm_v9`，EMA，输入/输出 q、delta_q、tau，历史 50、未来 40、action 10，100/25 Hz。该 checkpoint 的默认采样是 32 Heun；蒸馏监督按明确配置运行 64 Heun。
- 真实 USB dataset：200 episode / 213,796 有效窗口；episode split 190 train / 10 val。已有冻结 normalizer。
- 真实规模的 smoke Student 还通过了 adapter + benchmark 的 CPU/eager/cache 流程与 Teacher4 Euler 的同 NFE/contract-SHA 配对检查；只有1次测量，用于接口检查，不将该 JSON 用作速度/质量实验。
- 只执行一个 optimizer step：batch=2、gradient_every=2（两个 microbatch）、BF16 AMP、EMA、contact KL、完整 S4 链和低频梯度诊断。未运行持续训练。保存位置 `outputs/offline_distill_baseline/one_step_smoke_not_deployable/`，明确不作为训练完成的 S4。
- smoke 保存后通过独立 `--validate-only` FP32 CPU 入口重载严格 checkpoint：1 batch ×2 条件、2 noise、T64 Heun/S4/普通 T4 Euler，同噪声 terminal 和物理/contact/per-task 指标均生成。这里仅两个验证窗口，不能用低误差或接触分类数值宣称任务质量。
- 实际连续真实反馈入口：smoke Student 与原 Teacher，4 noise，指定窗口 100/500 与同一 episode 的 100/108/116 原始 anchor；完成 3 个重条件化 anchor、2 次绝对 timestamp 接合检查。没有把预测值填回历史。

日志与完整 JSON 保存在 `outputs/offline_distill_baseline/`：`pinn_tests.log`、`nero_tests.log`、`smoke.log`、`smoke_validation.json`、`smoke_replay.json`、`teacher_benchmark.json`、`compile_benchmark.json`。这些本地生成产物没有作为生产权重提交。

## 真实 Teacher benchmark

一个真实记录窗口，batch=1、num_samples=1、同物理 history/action 和同 source noise（seed=1234）。每种组合 warmup=10 +一次稳定调用、iterations=100；CUDA event/同步计时；FP32 禁用 TF32。这里的 adapter 不含 robot、相机、pi0 RPC、请求排队或 MTC；不能推断系统闭环频率或成功率。

| 方案 | 优化 | NFE | adapter p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| teacher32 | float32/eager | 64 | 75.348 | 76.468 | 76.755 |
| teacher32 | float32/kv | 64 | 60.043 | 60.503 | 60.700 |
| teacher64 | float32/eager | 128 | 151.868 | 156.773 | 157.779 |
| teacher64 | float32/kv | 128 | 120.133 | 121.595 | 122.339 |
| teacher4_heun | float32/eager | 8 | 10.933 | 11.267 | 11.314 |
| teacher4_heun | float32/kv | 8 | 9.149 | 9.353 | 9.485 |
| teacher4_euler | float32/eager | 4 | 6.543 | 7.326 | 7.493 |
| teacher4_euler | float32/kv | 4 | 5.701 | 6.895 | 15.435 |

所有本次 FP32 cache 组合相对同方案 FP32 eager 的物理 q/tau/contact probability 最大差为 0（只限这个记录输入/noise）。Student 则由小架构 cache 等价性测试验证；训练完成 Student 的真实规模测量仍待权重。

BF16 作为独立实验测量：T32 eager p50=95.590 ms、K/V p50=73.953 ms，在本机这个输入上都比对应 FP32 慢。相对 T32 FP32 的 q RMSE≈0.000176 rad、tau RMSE≈0.003495 N·m、contact probability 最大差≈1.47e-5。完整其他方案/分位数/误差见 JSON；这些数值不是跨任务物理容差或通过标准。

T4 Euler 单独跑实际 `torch.compile`（K/V，FP32，warmup=3+一次，iterations=30）：编译 active、graph breaks=0、unique graphs=1，初次编译/worker warmup 9.562 s。eager p50/p95/p99=5.945/6.068/6.083 ms，K/V+compile=2.846/2.974/2.985 ms。相对 eager q 最大差=1.19e-7 rad、tau 最大差=9.54e-7 N·m、contact probability 最大差=5.12e-9。两次 benchmark 分开运行，不将不同测量轮次的微小延迟差当作优化效果。

复现本次两个 benchmark（Nero cwd）：

```bash
PYTHONPATH=. ../PINN/.conda-env/bin/python scripts/benchmark_pi0_wm.py \
  --config inference/configs/pi0_wm.yaml \
  --checkpoint ../PINN/outputs/cwm_teacher_policy/insert_usb_50h_40step/checkpoints/step_00250000.pt \
  --device cuda:0 --payload ../PINN/outputs/offline_distill_baseline/recorded_window.npz \
  --modes eager kv --precisions float32 bfloat16 --warmup 10 --iterations 100 \
  --output ../PINN/outputs/offline_distill_baseline/teacher_benchmark.json
PYTHONPATH=. ../PINN/.conda-env/bin/python scripts/benchmark_pi0_wm.py \
  --config inference/configs/pi0_wm.yaml \
  --checkpoint ../PINN/outputs/cwm_teacher_policy/insert_usb_50h_40step/checkpoints/step_00250000.pt \
  --device cuda:0 --payload ../PINN/outputs/offline_distill_baseline/recorded_window.npz \
  --scenarios teacher4_euler --modes eager kv_compile --precisions float32 \
  --warmup 3 --iterations 30 --output ../PINN/outputs/offline_distill_baseline/compile_benchmark.json
```

## 未测与限制

没有找到已训练完成的 S4/S2 权重：benchmark JSON 明确跳过 Student4/Student2，所以没有报告其质量或加速比。没有实际 pi0 并发负载、真机执行或 MTC 成功率验证。没有完整多任务验证或用训练完成 Student 跑 32-noise 困难窗口。FP16 未测，实际 Student Inductor 编译未测，T32/T64 编译未测；只验证了真实 T4 Euler 编译和 Student 编译失败的明确回退。

固定窗口 Student-start 指标没有替代真实重条件化。本轮回放入口只是 teacher-forced real-feedback replay，缺少 policy/WM 延迟变化下的真实机器人闭环分布。接管/请求频率和观测年龄已经加入 runtime 日志，但尚未以硬件测量这些新字段。

当前默认 execute_steps=8 在100 Hz下是80 ms，需额外容纳 calibration margin。T32 K/V 的这组 p95约60.5 ms加默认20 ms margin 已接近或超过该预算；实际仍须按校准峰值（不是p50）判断，调度器会拒绝不能持续的组合。没有修改 execute_steps。数据集未来第0帧是 anchor+dt，而现有 prefetch 用 step-anchor 数组索引的一帧 nominal offset 已在主文档明确记录，未静默改动基线。
