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

## Training

Teacher and OPD Student training use optimizer updates as the authoritative
budget. `num_epochs` is only a data-pass/logging counter when
`max_optimizer_steps` is set.

```bash
carswm-train-contact-wm --config config/train_cfg/cwm_all_50hz.yaml
```

The authoritative budgets and checkpoint cadence are the
`train.max_optimizer_steps` and `train.checkpoint_every_steps` values in each
YAML. Set the baseline `dataloader.prediction_horizon` explicitly before training;
it is currently empty.

Every CARS-WM checkpoint contains `model_version: carswm_v8` and a complete
`carswm_contract` (schema 9). Older ContactWorldModel checkpoints do not satisfy this
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
