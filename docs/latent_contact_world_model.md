# Latent LSTM CARS-WM

Independent family/version `latent_carswm_lstm_v1`, branch
`feat/latent-carswm-lstm-grid`. The original WM model, trainer, configuration
files and checkpoint formats keep their existing behavior. No registry is needed.

## Entry points and budgets

From the repository root:

```bash
PYTHON=.conda-env/bin/python bash scripts/train_latent_cwm_two_tasks.sh
# One task, including recovery:
.conda-env/bin/python -m train.trainer.latent_contact_world_model_train \
  --config config/train_cfg/latent_cwm_insert_usb_100hz_40step.yaml \
  --resume outputs/latent_carswm_lstm_grid/insert_usb
```

The launcher locks the run, runs USB first, and verifies USB completion and
all five checkpoints before starting peel. Each task independently runs 10,000
codec optimizer updates followed by 250,000 Flow updates. Both task configurations
were derived from the actual workspace `config/train_cfg/pretrain/` files;
their source roots, feature mapping, filters, labels, splits, batch size,
optimizer/scheduler, AMP, device and EMA settings are retained. Seed is 42.
`train.stage` is `codec`, `flow`, or `all`; `--codec-checkpoint` skips Stage A
only after checking the codec, normalization, data and split contracts.

The separate `codec_step` and `global_step` count successful optimizer updates,
including when gradient accumulation is enabled. An FP16 scaler's skipped update
does not advance the counter, scheduler or EMA. Early stopping is rejected.
Flow checkpoints are `step_00050000.pt`, `step_00100000.pt`,
`step_00150000.pt`, `step_00200000.pt`, `step_00250000.pt`. BaseTrainer retains
the newest steps up to `top_k=5`. `latest.pt` also records complete recovery
state every 1,000 updates and on SIGINT/SIGTERM. Resume restores raw and EMA
states separately, optimizer, scheduler, scaler, RNG, stage and the actual
sample order/consumed cursor; worker prefetch does not advance that cursor.
SIGKILL/power failure falls back to the last atomic recovery file.

`resolved_config.yaml`, `grid_audit.json`, `metrics.jsonl`, `status.json`,
`codec.pt`, `codec_validation.json`, `final_validation.json` and checkpoint
visualizations provide durable records. `train.log` and `sequence.log` belong
to the sequential launcher. A `complete` status is written only after the
configured Flow update budget and final validation finish.

## Architecture and training

Exactly three unidirectional, `batch_first=True`, two-layer LSTMs condition
Flow. Motion receives concatenated `[q,dq,delta_q]` (21 channels), torque
receives measured `tau` (7), action receives the native action sequence (7).
Hidden/cell state resets on every window. Full motion/torque output sequences
form history memory; action remains a separate masked memory. Default width
is 128, history 50 at 100 Hz, future 40, action 10 at 25 Hz. The EE action is
absolute xyz + xyzw quaternion, in the original configured coordinate frame.

### Xarm erase-board data and missing EE actions

From the repository root, convert using
`python data_process/tool/h5_v3_wm.py -c config/shape_meta/swm/xarm/erase_board.yaml`.
This config keeps every native 100 Hz state row. With
`timeline.action_anchor_mode: state_frames` and `action_fps: 25`, action
snapshots come from state frame numbers **0, 4, 8, 12, ...**, and each is held
until the next selected frame. Frame numbers determine both selection and
holding; timestamp jitter does not change selected rows, and camera clocks
are not required. Recorded timestamps are preserved as timing metadata.
`action.joint` uses measured
`teleop/right_q_xarm`, as explicitly chosen for compatibility with that VA
dataset; `observation.delta_q` still uses commanded minus measured joints.
The single-channel external torque L1 signal remains `observation.tau_ext`.

The optional `action_fk` block materializes `action.ee_pose` when it is not
declared in the conversion features. It specifies the robot URDF, joint order,
base frame and target frame; paths resolve from the working directory. FK
uses raw radian joint actions before normalization and includes fixed tool
offsets. Poses are float32 `[x,y,z,qx,qy,qz,qw]`, in metres with unit
quaternions and `qw >= 0`. Non-finite numeric values fail conversion by
default. This xarm config selects `nonfinite_episode_policy: drop`, excluding
the entire invalid episode while keeping other episodes' clocks intact.
`meta/world_model_timeline.json` records excluded paths and reasons. No invalid
contact signal is replaced with zero. Existing declared EE actions take precedence.

Both WM datasets also accept `dataloader.action_fk` for older converted
datasets missing the configured EE action column. They compute and cache FK
from `action.joint` once at ingestion, preserve recorded action indices and
timestamps, and fit normalizers on the resulting EE conditions. Existing EE
columns are kept. Missing joints, incorrect dimensions, unknown frames or a
missing URDF produce an error. Do not substitute observation joints for a
joint-action column with different semantics.

Ready-to-run configurations are
`config/train_cfg/pretrain/xarm/cwm_erase_board_100hz_40step.yaml` for Contact WM
and `config/train_cfg/latent_cwm_erase_board_100hz_40step.yaml` for Latent WM.
Each uses history 50 at 100 Hz, action 10 at nominal 25 Hz and future 40
at 100 Hz. The current state is zero, the first future action is at `d0`,
and subsequent action tokens are at `d0 + 4*k`. Timestamps are not fed
directly to position encoding. Other configurations can retain the default
`recorded_camera` anchor mode, and fractional rate ratios remain supported
by the relative-step grid. In `state_frames` mode, fractional sampling uses
rounded cumulative frame offsets rather than rounding a single stride.

FutureEncoder is a separate deterministic, per-time MLP receiving normalized
q/tau and three-way contact one-hot. It uses two Linear layers and SiLU, mapping
17 to 128 to `latent_dim=128`. Each decoder uses exactly two Linear layers:
latent -> 128 -> q(7), tau(7), or contact logits(3). All heads decode the same
completed latent trajectory. This dimensionality is not a compression or
speed guarantee. There is no VAE, KL loss, autoregressive training, or required
generated delta_q.

Stage A trains only FutureEncoder and the heads with q/tau MSE and weighted
contact CE. Data normalization fits only training episodes. The chosen snapshot
is **final raw**, never a mixture with EMA. After freezing it, a fresh pass over
all training future windows fits per-channel latent mean/population std using
Welford merges. The std floor is 1e-4; near-zero channels are listed. Codec
weights, heads, statistics, snapshot SHA256 and validation metrics are saved
together in `codec.pt`. The 10k default budget is not a quality guarantee;
Stage B cannot recover information the codec discarded.

Stage B freezes the codec in eval mode. With no target-side gradient:

```text
z1 = (FutureEncoder(Y_future) - latent_mean) / latent_std
z0 ~ N(0,I), s ~ Uniform(0,1)
zs = (1-s) z0 + s z1
v_target = z1 - z0
loss = importance_weighted_MSE(v_theta(zs,s,conditions), v_target)
       + lambda_free * free_auxiliary_loss
```

The shared FlowTimeEmbedding and FlowDecoderBlock provide future self-attention,
parallel history/action cross-attention and FFN. Default depth/heads/multiplier
are 4/4/4, taken from the reference files. Training evaluates the velocity field
once; sampling integrates Gaussian noise across the complete [0,1] interval.
Default 16 Heun grid steps means 32 field evaluations; Euler uses 16.
`source_noise` has `[B,S,H,latent_dim]`, allowing fixed-noise comparison.
Conditions are encoded once and stay fixed throughout each integration.
Decoder input is the completed, denormalized latent. Outputs keep corresponding
sample axes `[B,S,H,D]` for q, tau and contact, rather than averaging modes.
`sample()` needs only conditions and grid positions, without future ground truth.

## NEXT motion pretraining

`model.pretrained_taufree_path: null` trains motion from scratch. A two-Linear
head on the last motion output regresses **current measured tau**, in the WM's
normalized tau space. Default `lambda_free=0.1`. Full-history validity and finite,
all-free contact labels determine the mask **before stride**. Padding, disabled
contact labels and any historical contact cannot confirm free. Select rows before
MSE, ignore non-free NaNs, normalize by selected importance weight sum, and
return differentiable zero for an empty/zero-weight set. Only motion and this
head receive this auxiliary gradient; neither tau nor action is read by the head.

For a non-null file/directory, the selected NEXT file is logged. The checkpoint
must specify LSTM, tau target, ordered `[q,dq,delta_q]`, explicit 7D inputs,
128 hidden units, two layers, stateless windows, matching history/cadence,
saved normalizer and compatible preprocessing. Only `recurrent.*` is imported,
with strict key/shape checks. The NEXT head (possibly width 256) is ignored.
The LSTM has no gradients, no optimizer membership, and stays eval after outer
`train()`. Other conditioners and Flow remain trainable. This mode has no random
free head and reports zero auxiliary contribution.

WM inputs are denormalized then normalized by NEXT statistics before the frozen
LSTM. Clipped quantile WM input is rejected because it cannot be inverted.
Filtering/resampling/source preprocessing differences are rejected; changing
normalizer statistics cannot repair them. In particular, the repository's raw
NEXT configuration with unfiltered motion is not automatically compatible with
the filtered WM sources. No unrelated robot/task checkpoint is selected.
Frozen weights, both sets of statistics and the validated contract are embedded
in new checkpoints. Loading/resume never opens the old NEXT or codec path.
Module parameter groups and `freeze_modules()` allow explicit future adaptation;
this run trains Flow normally and introduces no migration task.

## Relative robot-time grid and recorded data

One deterministic shared sinusoidal PE maps signed positions to hidden width.
There is no additional age, freshness, local-index or absolute timestamp feature.
Flow time s remains separate from robot physical time.

For external grid spacing Delta=10,000,000 ns and R=4:

```text
history = -L+1, ..., 0
future  = 1, ..., H
d0 = round((first_native_action_anchor_ns - request_history_anchor_ns)/Delta)
action  = d0, d0+R, ..., d0+(K-1)*R
```

Both timestamps are int64; subtract before division. Round nearest with ties
away from zero. Grid zero is the last measured observation in the **request**,
not inference completion. Explicit absolute grid metadata can instead supply
each history/action/future token's physical grid coordinate; subtract the
request grid anchor per token, including across scheduling windows.
The model receives only final int64 relative positions on its device.

In the V3 fallback the first action anchor is quantized once and other positions
use nominal R. Check every action period within ±4ms of 40ms and cumulative
drift within ±5ms over the chunk; native action indices must be consecutive.
State intervals must be within ±4ms of 10ms. Padding is exempt from state
cadence checks. Offset>0 rejects an actual first anchor preceding the observation.
Rounded d0=0 or 5 is legal and never clamped to 1..4. Ideal phase=0,1,2,3 gives
first positions 4,3,2,1. Concatenated sample_idx modulo four is never used.
State stride preserves external ticks: history retains current zero, and
future[::2] has positions 1,3,5,...; action stays native. Sampling a strided
model returns that sparse future grid, not invented intervening predictions.

Actual sources contain `timing.state_timestamp_ns`, `timing.action_index` and
`timing.action_anchor_timestamp_ns`. Unique action tables and native window
selection are inherited episode-locally. Median cadence alone is insufficient:
preflight found USB 906 and peel 331 periods outside ±10%, with maximum periods
51.151ms and 52.050ms. The new task configs explicitly select
`grid_invalid_window_policy: drop`; the adapter audits before splitting/fitting
and reports all rejected windows per episode. USB retains 182,078 of 213,796
windows; peel retains 68,847 of 80,808. No timestamps, actions or labels are
modified. Use policy `error` to abort rather than exclude incompatible windows.
Clock drift checks make the rejected-window fraction larger than the fraction
of individual exceptional action periods; results apply to the retained windows.

## Nero integration and offline example

Existing Nero WMAdapter is **not** claimed to load this family. Nero needs a
new family dispatch and `load_latent_checkpoint()` entry, plus the shared grid
helper and normalization contract. Example:

```bash
.conda-env/bin/python -m scripts.latent_cwm_nero_offline \
  --checkpoint outputs/latent_carswm_lstm_grid/insert_usb/checkpoints/latest.pt \
  --condition /path/to/measured_condition.pt --output /tmp/future_samples.pt \
  --device cuda:0 --num-samples 8 --seed 1234
```

The condition file contains physical, already compatibly preprocessed batched
`q,dq,delta_q,tau [B,L,7]`, absolute EE `action [B,K,7]`, optional action mask,
int64 `history_timestamp_ns [B,L]`, `action_chunk_timestamp_ns [B,K]`, and
optional `action_chunk_index [B,K]`. Padding can supply `history_real_mask`.
Alternatively `explicit_grid` contains `anchor_grid_position [B]` and absolute
per-token history/action/future grid positions. The example needs no true future
states and returns physical q/tau plus contact probabilities with matching S axes.

Choose the action window by the **currently held recorded native action index**
plus `action_start_offset=1`, then K consecutive native action tokens. Honor
configured prefetch delay selection on native anchors before constructing PE.
The old global-next-slot window can differ from native-next-action when source
clocks, phase, or recording jitter differ. Adding PE while retaining a different
window selection does not reproduce this trained contract. A multi-plan chunk
needs each token's own grid anchor; do not apply the first plan's origin to all.

Keep the original request anchor and prefetch latency indices so an executor
selects the appropriate prefix of the predicted future after computation.
Do not reset the origin to inference completion. Distinguish the 100 Hz state
interface from actual full sampling update frequency and the executed prefix
length. New observations trigger new sampling calls; each call uses fixed
conditions through its complete integration. `evaluate_feedback()` accepts a
series of recorded measured windows with re-anchored actions. It never writes
predicted tau into measured history. Pure free-running autoregression is not
implemented: future q_cmd/delta_q is unavailable when generating only q/tau.

## Validation scope

Codec reconstruction and physical q/tau MAE/RMSE, contact CE and per-class
precision/recall/F1 are separate from Flow latent FM/auxiliary losses. Integrated
validation uses the model's 16 steps/solver and fixed source seed. Sampled
physical errors average individual sample errors, not a mean trajectory.
Distribution metrics/energy scores use normalized streams for scale balance.
Codec metrics and latent FM cover the full validation split; integrated
distribution/physical/contact metrics cover `probabilistic_validation.max_batches`
(8 in the task configs), or all batches when zero. Monitor replacement honors
the preserved energy-score configuration. Contact confusion matrices accumulate
over all evaluated windows before F1. Different seeds can produce different
completed futures; fixed seeds/noise reproduce comparisons.

Targeted tests use synthetic NEXT files and synthetic episode columns. They
verify three LSTMs, gradient isolation, numerical transfer with differing scales,
strict incompatibility failures, self-contained loading, phases/rounding,
episode boundaries, full Heun interval and shared decoder samples, codec reuse,
and reduced-budget real optimizer runs/resume/top_k. These are CPU interface and
training smoke tests, not evidence of 250k training or real robot performance.
GPU/NPU kernels, hardware feedback latency and Nero deployment require their
own measurement. Consult the durable run status and final metrics for real
training progress; absence of a final metric/checkpoint means it has not completed.
