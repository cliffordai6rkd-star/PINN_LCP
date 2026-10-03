# CaRS-WM 独立 Student 基线与部署（2026-10-01）

本轮修改直接基于 PINN `master` 和 Nero `pi0_wm` 工作区，未重置参考提交、未提交/推送，未连接真机、未运行长期训练。用户原有的四份配置修改未覆盖。网络主体、q/tau 等表示、horizon、高层 absolute EE-pose action contract 和默认 MTC/execute_steps 保持原有定义。

## 实现与兼容性

| 文件 | 目的 |
|---|---|
| `model/pinn_model/checkpoint.py` | 按正式 `carswm_contract.student` 分发 Student；保留原 Teacher 和 deterministic 的 contract 校验。Student 与 Teacher 可能同为 `carswm_v9`，不能按这个版本号区分。 |
| `model/pinn_model/contact_world_model_student.py` | 将各 block 的 condition K/V 缓存传给 decoder；拒绝改变训练阶段步数及非 Euler solver。 |
| `train/trainer/contact_world_model_distill_train.py` | 冻结 Teacher 每 batch 只编码/缓存一次；可选 endpoint、执行区间、接合 loss；低频诊断；扩展验证/任务摘要；增加只验证入口。 |
| `train/carswm_execution_metrics.py` | 执行窗口、物理差分、边际 contact 校准/分类/事件、相邻 chunk 同绝对时间指标。 |
| `train/nomalizer.py` | 校验部署统计的维度、有限性、eps 和范围。保持原归一化公式。 |
| `config/train_cfg/contact_world_model_distill_s{8,4,2}.yaml` | 显式保留 legacy/default 参数；新增 endpoint+执行窗口对照配置。 |
| `scripts/diagnose_cwm_distill.py` | 指定小规模困难窗口、多 noise 和连续真实反馈回放。每个窗口从数据集重新取历史和时间对齐 action，不回填预测值。 |
| Nero `inference/pi0_wm/wm.py` | 严格模型分发/EMA/raw 选择，Student 默认使用 checkpoint 步数；请求内 cache；精度选项；日志/NFE/contact probability/模型计时/编译状态。 |
| Nero `inference/pi0_wm/config.py`、`inference/configs/pi0_wm_student_s{4,2}.yaml` | FP32 默认和独立 Student 部署示例。示例权重路径是占位符。原 Teacher YAML 保留。 |
| Nero `inference/pi0_wm/runtime.py` | 记录真实请求/接管频率、反馈/历史年龄、接管数组索引、旧参考及已下发参考差异。只增加日志。 |
| Nero `scripts/benchmark_pi0_wm.py` | 复用 adapter/compile，比较 Teacher 与实际 Student、多精度和缓存；CUDA 同步、p50/p95/p99、硬件、物理单位误差、同 NFE/延迟接近的配对。 |
| 两仓库新增/扩展的相关 tests | 验证严格加载、拒绝语义冲突、cache 等价、完整链梯度、loss 缩放、真实调度索引、物理/contact 指标和编译失败回退。 |

Teacher 仍支持 Euler/Heun 和显式步数覆盖。Student 用自身 `delta_s` 条件与 K 次 Euler 更新；S4 只能按 S4 部署，S2 必须经过实际 S4→S2 蒸馏。没有通过调低 Teacher 参数或新增 S1 获得 Student。deterministic 不参与 Flow 积分，部署时配置 `flow_steps: null, solver: null`；与旧 v1 的两种正式 checkpoint envelope 都兼容。权重全部 `strict=True`，normalizer、输入/输出顺序、action/time contract 继续校验。

Checkpoint 中 `model` 是保存的 EMA 部署权重，`model_raw` 是优化权重；选择基于实际 `ema.enabled`，与训练配置不一致或请求的权重缺失均报错。默认 `use_ema: true`。新增参数不改变模型 parameter 名称，所以既有正式 Student checkpoint 可继续严格加载；它的 config 不需要包含本轮新增可选字段。

K/V 只复用本次固定条件请求；public `sample()` 对新 history/action 重新编码。训练 Teacher 固定 eval/无梯度，完整轨迹与 Student-start 局部 Heun 区间共享缓存。Student 编码与完整 K 步保留反传；local 起点仍 detach。

## Loss 与时间语义

Flow 时间 `s` 与机器人时间 `t` 独立。terminal 是完成 K 步 Flow 后的**整个未来 chunk**。同条件、同 source noise 的 Teacher/Student 配对不把所有随机未来回归到同一真实未来。

`local_loss_mode: velocity` 是原默认：`||u - (T_interval(x,s,h)-x)/h||²`。`endpoint` 是 `||x+h*u-T_interval||²`，相差 `h²`；配置相同 local_weight 时，各 K 的 local/terminal 强度仍不相同。没有给固定权重最优性结论。

`execution_weight` 默认为 0，在原 full-terminal 之外增加执行数组区间匹配。非零时必须明确 `execution_window: {delay_steps: d, execute_steps: E, mode: prefetch}`。prefetch 按现有 Nero `step-request.anchor` 使用 `[d:d+E]`，openloop 用 `[0:E]`，边界严格拒绝越界。基线验证及执行对照文件的 d=4/E=8 是 **40 ms 延迟/80 ms 执行的离线假设**，不是实测延迟；原 execute_steps=8 不变。

时间对齐的已知限制：dataset 第 0 帧未来为 `anchor+dt`，现有 Nero prefetch 数组第 0 帧却对应 `step=anchor`。因此按现有索引分析的执行 `[d:d+E]` 在数据集时间上是 `(d+1)..(d+E)` 帧未来，存在一帧 nominal offset。本轮保持调度基线；回放的相邻预测比较使用一致的未来 timestamp，并在不一致时拒绝纯索引比较。

q D1/D2 训练项仍匹配 Teacher 的差分形状，仍在原 normalized q 尺度，不替换成零速度/零加速度惩罚。tau 没有新增平滑项。`join_weight` 默认 0，开启时需要实际 q/dq 历史与 q 输出，目标为当前 `q+dt*dq` 的首预测连续性（rad²），不是强制 q 下一帧相等。对缺 dq 的现有 USB Teacher，保持关闭；接合对照配置 `contact_world_model_distill_s4_join.yaml` 使用占位的、本来输入 dq 的正式 Teacher checkpoint，join_weight=0.01 仅为实验值；基线与接合对照必须使用同一个这样的 Teacher。不能为了此项修改当前网络输入。

`diagnostics_every_steps: 0` 默认关闭。启用后每个 optimizer step 编号最多诊断一次：原 loss、加权贡献、共享 projection/encoder 的未缩放 local/terminal 梯度范数、Flow 中间状态距离与局部 target 平均速度范数。`autograd.grad(retain_graph=True)` 不更改 `.grad`，保留梯度累积/AMP/EMA；分布式已初始化时跳过额外 autograd 诊断，仅记录无梯度数值。梯度范数是抽选的共享参数，不是整个网络范数。

## 验证指标

保留 normalized terminal/label、Energy Score、spread、coverage、同噪声 Teacher 匹配和 phase 指标。额外报告反归一化后的 q(rad/rad²)、tau(N·m/(N·m)²)、q D1(rad/s)、D2(rad/s²) 误差；差分用实际 future_time 的 dt，D2 用相邻速度中点间隔。没有时间戳时明确记录 nominal-dt fallback。state stride 情况先按 public sampler 的 ZOH 展开到部署 horizon，再计算这些执行参考指标。

数据中的 dq 可能是硬件测量和独立因果滤波值；预测 q 的物理差分不会等同该 dq。`join_measured_dq` 只是两者的接合诊断。单窗口首帧位移也不是“应为零”的合格线。

物理误差分别汇总 full、first1/4/8（长度不超过 horizon）、配置的执行窗口、free/transition/contact 及 `task_index` 和任务宏平均。任务来自 dataset 的 source metadata，摘要附 sources；不会写死 USB/黄瓜名。这里的 `establishing` 延续三类 transition-band 标签，可能也包含解除，不宣称纯建立阶段。差分按右端帧的 phase 分组。任务宏平均对验证中实际出现的 source 等权，不是重新采样平衡所有任务。

每条 noise trajectory 都计算 contact probability：分类、NLL、Brier、10-bin ECE、precision/recall/macro-F1 用 **noise 平均后的边际概率**；Teacher/Student 概率 MSE 用 **同 noise 配对**，含义不同。累计全验证 confusion/calibration，而不是平均单窗口 F1。建立/解除取窗口内第一个二值接触事件；使用末历史 label 检测首未来帧事件，分别报告匹配事件的秒误差和未检出/多报计数，不给缺失事件编造零误差。预测校准面向数据集 contact proxy，不是独立力传感器真值。

常规验证的 motor tau tail 使用每 batch 真实总力矩幅值的 95% 分位；这只是可重复的局部诊断，不能解释为外力矩/接触力。困难窗口入口只对明确选定的窗口增大 noise 数；报告 Teacher 各随机未来到 Student 集合的最近轨迹 RMS 和最差值、力矩尾部/峰值与接触事件。有限 noise 未覆盖低概率未来时仍不能声称证明模式消失。

`best_terminal` 保留原排序标准。每次保存附 `step_*_diagnostics.json`，包括近未来/phase/接合/contact/tail/per-task 摘要和实验窗口假设。没有人为设置真机合格阈值。

## 可运行命令

从 PINN 仓库执行；下面变量需要指向**实际训练权重**，占位路径不是本机权重：

```bash
TEACHER=/mnt/code/lcx/PINN/outputs/cwm_teacher_policy/insert_usb_50h_40step/checkpoints/step_00250000.pt
S4=/absolute/path/to/trained_s4_ema.pt
S2=/absolute/path/to/trained_s2_ema.pt

# S4 长期训练命令，仅供后续手动运行
PYTHONPATH=. .conda-env/bin/python train/trainer/contact_world_model_distill_train.py \
  -c config/train_cfg/contact_world_model_distill_s4.yaml --teacher-checkpoint "$TEACHER"

# S2 必须使用同一个原 Teacher 与已训练 S4 初始化
PYTHONPATH=. .conda-env/bin/python train/trainer/contact_world_model_distill_train.py \
  -c config/train_cfg/contact_world_model_distill_s2.yaml \
  --teacher-checkpoint "$TEACHER" --student-init-checkpoint "$S4"

# 独立 FP32 验证；移除 max-batches 后执行完整验证
OMP_NUM_THREADS=1 PYTHONPATH=. .conda-env/bin/python train/trainer/contact_world_model_distill_train.py \
  -c config/train_cfg/contact_world_model_distill_s4.yaml --teacher-checkpoint "$TEACHER" \
  --validate-only --student-checkpoint "$S4" --device cuda:0 --batch-size 2 \
  --max-batches 2 --output outputs/s4_validation.json
OMP_NUM_THREADS=1 PYTHONPATH=. .conda-env/bin/python train/trainer/contact_world_model_distill_train.py \
  -c config/train_cfg/contact_world_model_distill_s2.yaml --teacher-checkpoint "$TEACHER" \
  --student-init-checkpoint "$S4" --validate-only --student-checkpoint "$S2" \
  --device cuda:0 --batch-size 2 --max-batches 2 --output outputs/s2_validation.json

# 小困难集/连续真实反馈入口；indices 是 valid-window dataset 索引
# 回放 anchor 必须在同一 episode、有相同绝对 timestamp 的预测重叠。
PYTHONPATH=. .conda-env/bin/python scripts/diagnose_cwm_distill.py \
  --teacher-checkpoint "$TEACHER" --student-checkpoint "$S4" --device cuda:0 \
  --sample-indices 100 500 --num-samples 32 \
  --replay-start-index 100 --replay-count 3 --replay-stride 8 \
  --delay-steps 4 --execute-steps 8 --output outputs/s4_difficult_replay.json

.conda-env/bin/python -m pytest -q tests/test_contact_world_model_distill.py \
  tests/test_carswm_execution_metrics.py tests/test_feedback_reconditioned_validation.py \
  tests/test_carswm_metrics.py tests/test_normalizer.py \
  tests/test_base_trainer_step_checkpoints.py tests/test_base_trainer_resume.py
```

从 Nero 仓库执行。`PYTHONPATH=.` 或 `-m scripts.benchmark_pi0_wm` 确保项目 import；可用 Nero `.venv/bin/python`，本轮使用兄弟 PINN 环境。

```bash
TEACHER=/mnt/code/lcx/PINN/outputs/cwm_teacher_policy/insert_usb_50h_40step/checkpoints/step_00250000.pt
S4=/absolute/path/to/trained_s4_ema.pt
S2=/absolute/path/to/trained_s2_ema.pt
PYTHONPATH=. ../PINN/.conda-env/bin/python scripts/benchmark_pi0_wm.py \
  --config inference/configs/pi0_wm.yaml --checkpoint "$TEACHER" \
  --student-s4 "$S4" --student-s2 "$S2" --device cuda:0 \
  --modes eager kv compile kv_compile --precisions float32 bfloat16 \
  --warmup 10 --iterations 100 --output ../PINN/outputs/paired_wm_benchmark.json

# 可附加 --payload recorded_window.npz，键为物理 history_q/history_tau/…、action、可选 q_future/tau_future
# 没有 payload 时仅测 synthetic 计算成本与配对数值差异，不称为真实任务质量。
# 在另行启动的实际 pi0 负载下重复时，用 --concurrent-load-label 记录条件；脚本不会启动 pi0。
PYTHONPATH=. ../PINN/.conda-env/bin/python -m pytest -q tests/test_pi0_wm_student.py \
  tests/test_pi0_wm.py tests/test_pi0_sessions.py tests/test_pi0_wm_alignment.py \
  tests/test_pi0_openloop.py tests/test_pi0_mtc.py tests/test_pi0_command_rate.py tests/test_pi0_wm_reset.py
```

Student 真机配置为 Nero `inference/configs/pi0_wm_student_s4.yaml` / `s2.yaml`：更换占位 checkpoint；默认从 contract 取步数，Euler、EMA、FP32、请求内 K/V 打开、compile 关闭。后续编译与低精度要独立比较质量。没有自动运行真机入口。

## 已执行与待执行

执行记录、真实 Teacher benchmark 与 smoke 验证详见 `docs/cwm_distill_offline_results.md`。smoke checkpoint 只经过一个 optimizer step，不属于研究比较所需的训练完成 S4。

未进行训练完成的 S4/S2 质量/加速比较、完整所有任务验证、32-noise 困难集普查、真实 pi0 并发和真机/MTC 成功率比较。缺少实际训练 Student 权重与并发场景；相关命令见上。请求/接管频率需要从实际 runtime 日志测量；command_hz=100 只表示指令频率。校准仍在 `ceil((latency_max+margin)*hz)>execute_steps` 或 `prefetch+execute_steps>horizon` 时拒绝不可持续调度，没有增大 execute_steps 掩盖延迟。

## 后续最小实验表

| 比较 | 控制变量 | 需要的结果 |
|---|---|---|
| T32 Heun / T64 Heun / T4 Heun / T4 Euler / S4 / 实训 S2 | 同 history/action/noise、FP32、同任务划分、batch1 | NFE=64/128/8/4/4/2；延迟分位数、物理全/短期/执行误差、contact、tail |
| T4 Euler vs S4 | 同 NFE=4 | 误差、q D1/D2、contact 同 noise 匹配与边际校准 |
| 延迟接近的 Teacher vs Student | 明确毫秒容差，配对同 noise | 质量差异；若实测没有接近的组合，记录没有，而不冒充同延迟 |
| S4 legacy vs endpoint vs endpoint+执行项 | 同初始化/训练预算/采样计划；明确 h² 和 d/E | 各 loss 加权贡献、近未来/全域退化、per-task |
| 接合项关闭 vs 单独打开 | 只选原模型已有 dq 输入的 Teacher；同条件 | q/dq 接合、同绝对时刻 overlap、反馈后修正与误差；合理 contact 修正保留解释 |
| cache / compile / BF16（可选 FP16） | 逐项对 FP32 eager；同 checkpoint/条件/noise | 物理 q/tau/contact 差异、实际 compile 状态、p50/p95/p99 |
| 单任务 vs 多任务 | 相同架构/预算/采样权重，报告每任务 | USB 下降是否与采样/归一化/任务冲突相关；不预设 MoE 或多模态归因 |
| pi0 并发与默认 MTC | 同 execute_steps/增益与反馈频率 | 请求/接管频率、观测年龄、参考接合与成功率；需要后续明确真机授权 |
