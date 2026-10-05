# CARS-WM

CARS-WM is an action-conditioned probabilistic world model for robot contact
dynamics. It predicts the configured continuous state streams and a future
sequence of the existing discrete contact-phase labels.

The model contract is:

```text
input:  selected state history M + future action chunk
output: selected continuous streams M + future contact-phase sequence
M:      any ordered subset of [q, dq, delta_q, tau]
```

Contact phases are produced by the existing labeling pipeline (for example
`free`, `pre-contact`, and `contact`). CARS-WM does not create, consume,
predict, or optimize time-to-contact variables or bins. FIRST strengthened
sampling uses a fixed `max` aggregation of the contact-phase labels already in
each future window, with importance correction for the original empirical
risk.

The Teacher retains every temporal GRU output from each state modality,
adds modality embeddings and shared learned recency positions (newest = 0),
and concatenates history in modality-major order. Actions use a separate GRU
and learned sequence positions. Each Flow block preserves future self-attention,
then reads history and action through two parallel, independent cross-attention
branches and sums their residuals before the FFN. Future positions are learned
discrete embeddings; FlowTimeEmbedding still conditions on integration time s.

State rows remain at 100 Hz. The eight action tokens come from consecutive
recorded 25 Hz camera/VLA action indices, independently of state downsampling.
Timestamp metadata remains available for alignment and diagnostics.

The baseline enables `loss.free_dynamics_weight: 0.1`. A small shared-encoder
head reads only the current raw GRU outputs of q/dq/delta_q to predict current
measured tau in its existing normalization space. Only complete, valid, all-free
history windows supervise it; padding and skipped contact rows cannot qualify.
The auxiliary MSE is normalized by the sum of free-sample importance weights.
Main-model tau history is never masked. Prediction/sampling do not run this head.

## Environment

The repository uses the `pinn` Conda environment with Python 3.10.20. The
validated GPU baseline is PyTorch 2.6.0 with CUDA 12.4, Pinocchio 3.9.0,
MuJoCo 3.3.7, and LeRobot 0.4.0. There is no separate `requirements.txt`;
the dependency declarations are in `setup.py` and the reproducible install
entry point is `setup.sh`.

From the repository root, check that the NVIDIA driver is visible and run:

```bash
nvidia-smi
bash setup.sh
conda activate pinn
python -m pip check
```

`setup.sh` installs the CUDA 12.4 PyTorch wheels, the core data-processing
packages, the physics extras, and the test tools. It also performs a CUDA
availability check. To use another Conda environment name, set
`PINN_CONDA_ENV_NAME` before running the script.

The base environment uses `opencv-python-headless`, which works on servers
without a display. Install optional features only when needed:

```bash
# SSD-backed dataset cache (`train_data.cache.mode: ssd_zarr`)
python -m pip install -e ".[cache]"

# RGB-D/point-cloud and SAM tooling
python -m pip install -e ".[vision]" \
  git+https://github.com/ultralytics/CLIP.git
```

After installation, verify the main runtime with:

```bash
python - <<'PY'
import mujoco
import pinocchio
import torch

print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"pinocchio={pinocchio.__version__}, mujoco={mujoco.__version__}")
PY
```

## Training

For xArm erase-board training with shared torque-derived contact labels, use
`config/train_cfg/latent_cwm_erase_board_100hz_40step.yaml` or
`config/train_cfg/pretrain/xarm/cwm_erase_board_100hz_40step.yaml`.
Both use eight native 25 Hz action tokens and predict 32 state frames at 100 Hz.
See [the xArm training guide](docs/xarm_tau_labeling.md) for commands and the
retained torque model, label cache, and phase rules.

Train Contact WM and Latent WM sequentially, then evaluate each model's final
weights with **32, 16, and 8 Flow integration steps**:

```bash
# Run from the repository root. Keep this directory to resume the same run.
export WM_RUN_ROOT="outputs/xarm_wm_flow_sweep/erase_board_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WM_RUN_ROOT"
nohup bash scripts/train_xarm_wm_flow_sweep.sh \
  > "$WM_RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo "Launcher PID: $!; results: $WM_RUN_ROOT"

# Monitor training and inspect timing/status records.
tail -f "$WM_RUN_ROOT/contact_wm/train.log"
# After Contact WM finishes, the launcher starts latent_wm/train.log.
.conda-env/bin/python scripts/xarm_wm_flow_sweep.py report "$WM_RUN_ROOT"
```

To train both models concurrently when CPU, RAM, and GPU memory permit, set
`export WM_PARALLEL=1` before launching. The launcher waits for both training
jobs to finish, then evaluates them sequentially so the final sampling-time
comparison does not include competing training kernels. Per-model locks
prevent duplicate training, and timing reports support concurrent writers.
Training durations in this mode include contention for the shared GPU and
should not be compared directly with durations from isolated training.

For the downloaded xArm peel-cucumber dataset, use the matching configs
(copied from the completed erase-board run, with only task paths/names changed):

```bash
export CONTACT_CONFIG=config/train_cfg/pretrain/xarm/cwm_peel_cucumber_100hz_40step.yaml
export LATENT_CONFIG=config/train_cfg/latent_cwm_xarm_peel_cucumber_100hz_40step.yaml
export WM_PARALLEL=1
export WM_RUN_ROOT="outputs/xarm_wm_flow_sweep/peel_cucumber_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$WM_RUN_ROOT"
nohup bash scripts/train_xarm_wm_flow_sweep.sh \
  > "$WM_RUN_ROOT/launcher.log" 2>&1 < /dev/null &
```

Both models train for **250,000 optimizer updates**, saving every **50,000**.
Latent WM additionally trains its codec for 10,000 updates before Flow;
its reported training time includes this stage. The action condition remains
eight native 25 Hz actions, with 32 predicted state frames at 100 Hz.
Inference integration steps do not change these training budgets.

The launcher snapshots the two erase-board YAMLs under the run directory and
trains each model only once. It preserves their configured training-time
sampling defaults (Contact: 32; Latent: 16) and aligns Contact's separate
probabilistic-validation step setting with that default. Post-training
evaluation uses the same final EMA weights, validation windows, and source
noise for all three step counts within each model. By default it scores eight
validation batches with eight generated futures per window, using Heun.

`training_times.csv` records training start/completion timestamps, measured
wall time, budgets, and status. Training time includes setup, codec where
applicable, in-training validation, checkpoint writing, and logging.
`evaluation_results.csv` records the six evaluations, prediction metrics, and
sampling time. Sampling time excludes warmup, data loading, and metric
calculation; `sampling_ms_per_window` includes all eight sampled futures and
is a batched throughput measurement, not single-request latency. Each model's
`timing.json` retains detailed attempts and the final checkpoint hash.

Re-run the same command with the same `WM_RUN_ROOT` to skip completed models
and resume incomplete ones from their saved checkpoints. Original config
snapshots are reused; changing the source YAML requires a new run directory.
A failed or interrupted run stops the sequence. Completed attempt times are
summed across restarts; hard-killed attempts without a recorded duration are
flagged with `timing_complete=False`. Use `WM_PREPARE_ONLY=1` to generate
configs without starting training, or `WANDB_MODE=offline` for local W&B logs.

Teacher and OPD Student training use optimizer updates as the authoritative
budget. `num_epochs` is only a data-pass/logging counter when
`max_optimizer_steps` is set.

```bash
carswm-train-contact-wm \
  --config config/train_cfg/cwm_all_100hz_40step_all.yaml
```

The authoritative budgets and checkpoint cadence are the
`train.max_optimizer_steps` and `train.checkpoint_every_steps` values in each
YAML. Set the baseline `dataloader.prediction_horizon` explicitly before training;
it is currently empty.

Every CARS-WM checkpoint contains `model_version: carswm_v9` and a complete
`carswm_contract` (schema 10). Older ContactWorldModel checkpoints do not satisfy this
contract and must be retrained.

## Checkpoint diagnostics

For each saved checkpoint, fixed validation anchors and fixed Flow source noise
produce comparable artifacts under `checkpoint_viz/`:

```text
step_XXXXXXXX_summary.png
step_XXXXXXXX_metrics.json
latest_summary.png
fixed_samples.json
plot_scales.json
```

The PNG contains q and tau trajectories, contact probabilities,
phase-conditioned endpoint samples, distribution metrics, and contact
calibration metrics. Aggregate
Energy Score/minADE/minFDE are computed in normalized space; plots are
denormalized. Contact-onset error is evaluation-only.

## Feedback-reconditioned validation

Validation records three distinct behaviors. `rollout_*` scores one open-loop
future from the initial recorded history. `free_running_*` recursively inserts
the model's predicted state into later histories. `feedback_u{interval}_*`
instead scores at most `interval` 100 Hz state steps, inserts the corresponding
recorded `q`, `dq`, `delta_q`, and `tau` measurements, re-anchors the future
action chunk, and predicts again.

The configured `train.rollout_validation.measurement_update_intervals` default
is `[1, 4, 8, 32]`. Each update consumes the matching `action_rollout` and
`action_rollout_mask` entry. Fixed source noise and validation EMA weights make
the metrics comparable across checkpoints. With multiple samples, every
`feedback_u*` family includes Energy Score, minADE, minFDE, sample spread, 90%
coverage, contact calibration/classification metrics, and metrics grouped by
the existing free/transition/contact phase labels. A scored segment is assigned
to the maximum existing phase label present in that segment; no time-to-contact
target or bin is derived.

For multi-task sources, validation also emits `*_task_<name>_*` metrics and
equal-weight `*_task_macro_*` averages. The task macro is the reported
cross-task comparison; the sample-count-weighted aggregate remains available
for compatibility. Panel D is intentionally named **Phase-conditioned
endpoint samples**: colors identify different task/phase conditions and do not
claim that those clusters are multiple futures for one identical condition.
Evidence for condition-similar, future-different behavior requires an explicitly
curated matched validation group, for example via `paired_validation_indices`.
Checkpoint visualization JSON/PNG files generated before the NLL and global
confusion-matrix fixes are stale and must be regenerated from their checkpoints.

This is an **offline measurement-updated evaluation** using ground-truth
measurements from the validation dataset. It must not be reported as real
closed-loop robot task performance. This path is maintained for Teacher
validation only and is explicitly disabled for the OPD Student. It does not
change the training loss;
`rollout_validation.replace_val_loss` must be explicitly enabled to use the
configured `replace_val_loss_metric` as the validation monitor.

## Benchmark and tests

```bash
python scripts/benchmark_carswm.py --iterations 5 --warmup 2 \
  --batch-size 2 --rollout-depth 4 --num-samples 4 \
  --output outputs/carswm_benchmark_rtx4060.json
python -m pytest -q
```

The benchmark reports `tau_free_contact_model_ms: null` unless a compatible
checkpoint is explicitly benchmarked; no result is inferred or fabricated.

## Data preparation

```bash
python data_process/tool/filter_h5_butterworth.py \
  --input-dir ../nero_ws/runs/bg_data \
  --output-dir ../nero_ws/runs/bg_data_filted --cutoff-hz 15 \
  --dataset teleop/q_follower --dataset teleop/dq_follower \
  --dataset teleop/tau_follower --dataset teleop/q_cmd
python data_process/tool/downsample_h5_2_lerobotv3.py \
  --config config/shape_meta/data/next_data_dual_phase.yaml
```
