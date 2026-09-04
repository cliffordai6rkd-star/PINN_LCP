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

The Teacher and OPD Student use the same simplified architecture. Each selected
state stream has an independent GRU whose final hidden state is one modality
token. An action GRU produces the future action tokens, and the state and action
tokens are concatenated directly into the condition memory. Every Flow block
then applies future-token self-attention, cross-attention to that condition
memory, and an FFN before predicting the conditional velocity field. Teacher
and Student differ only in configured capacity and Flow integration steps.

## Training

Teacher and OPD Student training use optimizer updates as the authoritative
budget. `num_epochs` is only a data-pass/logging counter when
`max_optimizer_steps` is set.

```bash
carswm-train-contact-wm --config config/train_cfg/contact_world_model.yaml
carswm-train-contact-wm-opd --config config/train_cfg/contact_world_model_opd.yaml
```

The authoritative budgets and checkpoint cadence are the
`train.max_optimizer_steps` and `train.checkpoint_every_steps` values in each
YAML. The old `max_train_steps` key is accepted only as a compatibility alias.

Every CARS-WM checkpoint contains `model_version: carswm_v2` and a complete
`carswm_contract`. Older ContactWorldModel checkpoints do not satisfy this
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

The PNG contains q and tau trajectories, contact probabilities, q-tau endpoint
modes, distribution metrics, and contact calibration metrics. Aggregate
Energy Score/minADE/minFDE are computed in normalized space; plots are
denormalized. Contact-onset error is evaluation-only.

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
