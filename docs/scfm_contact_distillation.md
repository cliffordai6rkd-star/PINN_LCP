# SCFM for the original GRU world model

`scripts/posttrain_scfm_contact.py` supports `carswm_v9` schema 10 checkpoints,
including the 200,000- and 250,000-step teachers supplied on 2026-10-04.
It reuses the dual-target SCFM objective described in `scfm_latent_distillation.md`.
No step-size embedding or latent codec is added. Standard Gaussian noise remains
the source. GRU encoders, contact head and their shared future position embedding
stay frozen; only Flow input/time projections, decoder blocks and output learn.
Exports load strictly into the ordinary `ContactWorldModel` deployment class.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/posttrain_scfm_contact.py \
  --base-checkpoint /path/to/step_00250000.pt \
  --data-root /path/to/insert_usb_lerobot_v3 \
  --repo-id insert_usb_lerobot_v3 \
  --output /path/to/scfm_gru_4step.pt
```

The teacher config supplies preprocessing, action alignment, contact thresholds
and the episode split. Its saved normalizer is reused without refitting. The
script reconstructs the split and records its hashes; these old checkpoints
have no original split hashes, so equivalence to their historical split cannot
be verified. `--resume` restores student, both EMAs, optimizer and batch/RNG
cursors. Only total update count may change. Initial and subsequent validation
compare original 32-step Heun, original 4-step Euler, and SCFM 4-step Euler using
paired noise and held-out episodes; q/tau distribution scores and contact F1
must be checked before deployment. Fine Heun has 64 NFE, coarse Euler has 4.

## Dataset supplied in October

`insert_usb_lerobotv3.zip` has 200 episodes, 55,335 frames, and 25 Hz states.
Its parquet schema lacks the original teacher's state/action timing fields,
`observation.delta_q` and `observation.tau_ext`. It is a VLA export, not the
100 Hz world-model export. A 20-frame history / 40-frame future would span
0.8 / 1.6 seconds at 25 Hz instead of the teacher's 0.2 / 0.4 seconds.
Do not silently change the rate or fill missing contact targets with zeros.
Interpolated experiments need separate provenance and cannot establish preserved
contact dynamics or hardware success. The original WM metadata in this repo
describes 221,805 frames, but only metadata is tracked, not its parquet data.

### Authorized interpolation pilot

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python scripts/posttrain_scfm_contact.py \
  --config config/train_cfg/contact_cwm_scfm_interpolation_pilot.yaml \
  --base-checkpoint /path/to/step_00250000.pt \
  --data-root /path/to/insert_usb_lerobotv3 --interpolate-vla \
  --output /path/to/student_latest.pt --save-condition /path/to/heldout_condition.pt
```

This explicit adapter anchors every window at an original observed row, linearly
interpolates 100 Hz history/future queries within each episode, and retains the
original future 25 Hz EE action sequence. History interpolation brackets never
extend beyond the anchor. Full teacher causal filters are applied with one second
of warmup (the VLA source preprocessing is unverified), then the teacher's fixed
normalizer is reused. Recorded `action.joint - observation.joint` is interpolated
as a delta-q proxy; it is zero in every provided source row, and cannot recover
the true tracking error. No tau-ext/contact labels are invented. Validation
reports distribution metrics against approximate interpolated future targets,
paired teacher q/tau errors and contact KL; no contact F1 or robot success claim.
Source content hashes and all these approximations are embedded in the artifact.
The best validated teacher-agreement candidate is saved next to the latest
checkpoint, with `_best` appended. Both remain experimental artifacts.

## Latency only

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_contact_checkpoint.py \
  --checkpoint /path/to/step_00250000.pt --synthetic \
  --precision fp32 --output /path/to/latency.json
```

This measures one sample at batch size one, including GRU/Flow/contact decoding
with GPU synchronization. It excludes acquisition, normalization, transfer and
robot control. Synthetic conditions give runtime costs, not quality or success
rates. The script also measures 4-step Heun and 4-step Euler with/without per-call
condition K/V caching. This cache is rebuilt for every changed condition.

## xArm schema 11 checkpoints

The GRU adapter validates both schema 10 and schema 11 explicitly. Schema 11
uses the upstream strict torque threshold and temporal alignment labels; weights
and architecture are unchanged. The original threshold and contact semantics
are preserved in the exported contract. Source `tau_label_contract` is retained.

The 2026-10-05 erase-board archive contains GRU and latent/LSTM WM teachers.
Both configs use the native `erase_board_25hzcam_cwm_lerobot_v3` dataset (100 Hz
states / 25 Hz FK action snapshots). The 25 Hz three-camera dataset is a VLA
export and is not the training input of either archived WM. Actual state timestamp
median interval is 9.999 ms and recorded delta-q is nonzero.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python scripts/posttrain_scfm_contact.py \
  --config config/train_cfg/contact_cwm_scfm_xarm_native.yaml \
  --base-checkpoint /path/to/contact_wm/checkpoints/step_00250000.pt \
  --data-root /path/to/erase_board_25hzcam_cwm_lerobot_v3 \
  --unlabeled-native --output /path/to/student_latest.pt
```

`--unlabeled-native` disables only dataset contact labeling. It preserves the
model's contact head, label contract, original normalizer, state/action alignment
and preprocessing. Real q/tau futures suffice for direct-state GRU SCFM. Original
offline-label validity exclusions cannot be reproduced without the missing
cache, so their absence is recorded; reconstructed train/validation episodes
are still kept separate. Contact F1 is not reported; contact KL measures teacher
agreement. No free/contact labels are invented. This flag is rejected for latent
SCFM, whose frozen future encoder also takes contact one-hot targets.

To reproduce latent SCFM on the original window set, supply the offline torque
teacher or `wm_tau_labels` cache. The original archive's label report identifies
teacher SHA256 `d71489f15c391da3f669d34130e9d55c4dd030c9aab737ebe3973acab854bb61`
and label contract `91032c76bf9f801a91c06041c99c89d853c76cc7f4e68a5afed2bedad78f885b`.
