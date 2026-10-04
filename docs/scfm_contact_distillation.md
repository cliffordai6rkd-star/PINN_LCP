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
