# Deterministic robot-state world model

The backbone is independent modality GRUs plus a future-query Transformer
decoder. Four state GRUs encode q/dq/delta_q/measured tau; a separate GRU encodes
the recorded future action chunk. All history tokens retain modality and recency
embeddings. Each decoder block uses future self-attention, parallel history and
action cross-attention, and an FFN, with pre-norm residual connections.

A learned query plus a learned future position starts each future token. A
linear state head directly predicts the configured streams. The contact head
fuses decoder features, nominal-rate aligned action features, and the
token-weighted condition summary, then applies a temporal attention layer and a
three-class MLP. There is no source noise, flow time, flow loss, or integration.
Use `model.eval()` for deterministic inference; training retains dropout.

The implementation is `DeterministicRobotStateWorldModel(nn.Module)`, version
`deterministic_wm_v1`. The trainer independently extends `BaseTrainer` and uses
`ContactWorldModelDataset`. It does not inherit either CARS-WM class.

## Configuration and shapes

`config/train_cfg/deterministic_wm_baseline.yaml` copies data sources, filters,
normalization, contact labels/sampling, episode split and training settings from
`cwm_all_100hz_40step_all.yaml`. It has its own output directory and regression
objective.
The model, action, and temporal dimensions are all configurable.

| Setting | Default |
| --- | --- |
| `model.inputs` | `[q, dq, delta_q, tau]` |
| `model.outputs` | `[q, tau]` |
| `joint_dim`, `action_dim` | 7, 7 |
| `hidden_dim` | 128 |
| `state_layers`, `action_layers` | 2, 2 |
| `decoder_layers`, `attention_heads`, `ffn_multiplier` | 4, 4, 4 |
| `dropout` | 0.01 |
| `contact_state_count`, `contact_head_hidden_dim` | 3, 64 |
| `dataloader.state_history_horizon` | 40 |
| `dataloader.prediction_horizon` | 16 |
| `dataloader.action_condition_horizon` | 4 |
| `train.downsample` | false |

The 7-D action is the existing absolute end-effector xyz/quaternion contract,
not a joint-position command. Changing horizons does not change that contract.

`train.downsample` accepts false (stride 1), true (stride 2), or a positive
integer. History and prediction horizons must be divisible by the stride.
History striding starts at `stride - 1`, retaining the current anchor; future
target striding starts at zero. Action tokens and masks are never strided.
`prepare_batch()` accepts either external or already-prepared internal windows
and is idempotent. `use_action_padding_mask` follows the WM configuration: when
enabled, invalid action values are zeroed and excluded from cross-attention and
the condition summary. A completely masked action chunk is rejected.

`forward(batch)` returns internal-grid `q_pred` and `tau_pred` of shape
`[B, prediction_horizon / stride, 7]` and `contact_logits` of shape
`[B, prediction_horizon / stride, 3]`. It also returns decoder features and the
prepared batch used by the loss. Targets may be absent at inference.

`predict(batch)` restores the external prediction horizon by repeating each
internal frame, exactly as in WM. It does not synthesize intermediate frames.
For the default configuration the outputs are `[B,16,7]`, `[B,16,7]`, and
`[B,16,3]`. Enabling downsample makes forward outputs 8 steps while predict
outputs remain 16 steps. Other supported output streams (`dq`, `delta_q`) use
the same joint dimension and are split in the configured order.

The three labels retain the existing meaning: free, pre-contact/transition
(contact alignment), and contact. The labels and phase sampler are unchanged.

## Loss and validation

For each sample, state squared errors are averaged over time and joints;
contact CE is averaged over time. Configured stream weights and contact class
weights are applied. The objective is:

```text
mean_batch(importance_weight * (
    sum_streams(stream_weight * per_sample_MSE)
    + contact_weight * per_sample_CE
))
```

This matches the WM importance-correction convention (divide by batch size,
not by the sum of weights). Default stream weights are 1 and contact weight is
0.1. `loss.use_importance_weight: false` disables importance correction.
Statistics and automatic contact class weights are fitted only on training
episodes. Normalized validation MSE/MAE and unweighted contact CE are reported
alongside the overall objective. Contact accuracy and macro-F1 are computed
from one confusion matrix accumulated over the complete validation set; all
configured classes are included, with zero F1 for an absent class. Validation
uses EMA when configured. There is no sampling or rollout validation.

The default baseline has 2,283,537 parameters. The current
`cwm_all_100hz_40step_all.yaml`
model has 2,354,712 parameters, including its auxiliary free-dynamics head.
The baseline intentionally uses only state MSE + contact CE, as requested;
the WM's `free_dynamics_weight: 0.1` auxiliary objective is not included.
Consequently comparison with that experiment also differs in auxiliary
supervision, beyond the prediction paradigm. State/action encoding, decoder
width/depth/heads/dropout, data and optimizer settings are matched.

## Training

From the repository root:

```bash
# Recorded LeRobot v3 data, using the device and settings in the YAML.
bash scripts/train_deterministic_wm_baseline.sh \
  --config config/train_cfg/deterministic_wm_baseline.yaml

# Select CUDA or NPU explicitly (requires the respective runtime).
bash scripts/train_deterministic_wm_baseline.sh --device cuda:0
bash scripts/train_deterministic_wm_baseline.sh --device npu:0

# Resume raw model, EMA, optimizer, scheduler, RNG, normalizer, and step budget.
bash scripts/train_deterministic_wm_baseline.sh \
  --resume outputs/deterministic_wm_baseline/checkpoints/latest.pt

# Data-independent CPU check: one synthetic update, validation and EMA save.
bash scripts/train_deterministic_wm_baseline.sh --smoke-test

# Run the integration and model tests.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_deterministic_world_model.py
```

The equivalent Python entry point is
`python -m train.trainer.deterministic_world_model_train` with the same flags.
The shell script changes to the repository root so relative data paths resolve
as in the teacher experiment. `--output-dir` and `--max-optimizer-steps` override
their YAML values. `--device cpu` disables AMP and dataset-on-device. Smoke mode
defaults to CPU, disables W&B, and writes to
`outputs/deterministic_wm_baseline_smoke`; it does not read the real dataset.
It cannot resume a recorded-data run.

Baseline checkpoints use the same WM envelope (`config`, `model_version`,
`carswm_contract`, `model`, `model_raw`, `normalizer`, and EMA/training state).
The architecture remains `deterministic_wm_v1`; its contract is also saved under
`deterministic_wm_contract` for compatibility. The Nero WM loader selects the
model class by version and validates that architecture's contract before loading
weights. Existing deterministic checkpoints with only `deterministic_wm_contract`
can be loaded or resumed without conversion or retraining. Resume also checks
loss configuration. `model` contains EMA parameters when enabled, with trainable
parameters in `model_raw`; without EMA, `model` contains the trainable parameters.

Nero's `contact_world_model` inference mode supports both architectures. Point
`contactworldmodel.path` in `nero_ws/inference/configs/nero_contact_wm.yaml` at
the deterministic checkpoint and run the same `visualize_wm_lerobotv3.py` command.
The online contact WM pipeline and PI0 WM adapter use the common `predict` and
`sample` interfaces. PI0 additionally requires its existing preprocessing
metadata (including an explicit `dataloader.dq_source`); older checkpoints
without that declaration remain rejected by PI0's preprocessing check.
`predict(batch, steps=..., solver=..., source_noise=...)`
accepts the flow options for compatibility and ignores them. `sample` decodes
once and repeats the prediction as `[B,K,T,D]`; all K futures are identical in
evaluation mode. The offline visualizer uses `predict_from_conditions` to decode
on the internal grid, then restores the external rate and physical units.

The tests use synthetic LeRobot columns with the real dataset to exercise
episode split, training-only normalization, phase sampling, one optimizer step,
validation, checkpoint save and resume with and without EMA/downsampling. CPU
tests do not establish CUDA/NPU runtime compatibility on hardware.
