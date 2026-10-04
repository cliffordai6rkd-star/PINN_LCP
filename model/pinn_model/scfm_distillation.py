"""SCFM dual-target velocity distillation in the WM noise(0)->data(1) convention.

Algorithm adapted from caitree/scfm, commit 41d435d36cfcdeb945cb9562ddb87ca5e5cd285a,
trainer/flux_scfm.py and src/utils/train_utils.py. See third_party/SCFM_LICENSE.
There is no step-size input to the student and no change to its architecture.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict

import torch


@dataclass
class SCFMSettings:
    steps: int = 4
    updates: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-5
    teacher_min_steps: int = 32
    teacher_max_steps: int = 32
    teacher_ratio: float = 0.4
    anchor_ratio: float = 0.0
    fast_ema_decay: float = 0.0  # official Flux code; paper uses .99
    slow_ema_decay: float = 0.999
    shift_min: float = 1.0  # WM uniform grid; Flux launch samples 2.5..4.5
    shift_max: float = 1.0
    precision: str = "fp32"
    seed: int = 42
    few_shot_windows: int = 0
    save_every: int = 100
    validate_every: int = 100
    validation_batches: int = 8
    validation_samples: int = 8
    reference_steps: int = 32

    def validate(self):
        for key in ("steps", "updates", "batch_size", "teacher_min_steps", "teacher_max_steps",
                    "save_every", "validate_every", "validation_batches", "validation_samples", "reference_steps"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.teacher_min_steps < 4 or self.teacher_max_steps < self.teacher_min_steps:
            raise ValueError("teacher grid range must satisfy 4 <= min <= max")
        if self.teacher_min_steps % 2 or self.teacher_max_steps % 2:
            raise ValueError("teacher grid endpoints must be even for two-step shortcuts")
        for key in ("teacher_ratio", "anchor_ratio", "fast_ema_decay", "slow_ema_decay"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must lie in [0,1]")
        if self.fast_ema_decay >= 1 or self.slow_ema_decay >= 1:
            raise ValueError("EMA decay must be less than one")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError("learning_rate must be finite and positive")
        if not (math.isfinite(self.shift_min) and math.isfinite(self.shift_max)
                and 0 < self.shift_min <= self.shift_max):
            raise ValueError("shift range must be finite, positive and ordered")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if (isinstance(self.few_shot_windows, bool) or not isinstance(self.few_shot_windows, int)
                or self.few_shot_windows < 0):
            raise ValueError("few_shot_windows must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return self

    def as_dict(self):
        return asdict(self)


@dataclass
class ShortcutSchedule:
    times: torch.Tensor  # [B,3] start, middle, end, increasing WM time
    teacher: torch.Tensor  # [B], first segment uses frozen teacher vs fast EMA
    anchor: torch.Tensor  # [B], ordinary FM targets replace shortcut targets


def sample_schedule(batch_size, settings, generator, *, device="cpu"):
    """Two adjacent Euler intervals, merged into a step-size-free velocity target.

    Sample the upstream *noise sigma* grid, then convert s=1-sigma. In particular,
    applying Flux's rational shift directly to WM time would reverse its bias.
    Anchors are allocated separately, avoiding upstream's suffix/index mismatch.
    """
    settings.validate()
    if batch_size < 1:
        raise ValueError("empty SCFM batch")
    times = torch.empty(batch_size, 3)
    anchor = torch.arange(batch_size) < math.ceil(batch_size * settings.anchor_ratio)
    teacher = torch.rand(batch_size, generator=generator) < settings.teacher_ratio
    self_grids = [2**power for power in range(1, math.ceil(math.log2(settings.teacher_min_steps)))
                  if 2**power < settings.teacher_min_steps]
    teacher_grids = list(range(settings.teacher_min_steps, settings.teacher_max_steps + 1, 2))
    for index in range(batch_size):
        if anchor[index]:
            times[index] = torch.rand((), generator=generator)
            continue
        grids = teacher_grids if teacher[index] else self_grids
        grid = grids[int(torch.randint(len(grids), (), generator=generator))]
        start = 2 * int(torch.randint(grid // 2, (), generator=generator))
        shift = settings.shift_min + (settings.shift_max-settings.shift_min) * float(
            torch.rand((), generator=generator))
        sigma = 1 - torch.arange(start, start+3, dtype=torch.float64) / grid
        sigma = shift * sigma / (1 + (shift-1) * sigma)
        if start == 0:
            sigma[0] = 1 - 1e-5
        times[index] = (1-sigma).float()
    return ShortcutSchedule(times.to(device), teacher.to(device), anchor.to(device))


def slice_conditions(encoded, indices):
    # K/V projections belong to each separate velocity model. Never reuse the
    # teacher cache for an EMA or a student; frozen encoder tokens can be shared.
    return {key: (None if value is None else value[indices]) for key, value in encoded.items()
            if key in {"history", "action", "action_padding_mask", "future_pe",
                       "state_tokens", "action_tokens", "condition_summary"}}


@torch.no_grad()
def dual_target(state, times, teacher_mask, encoded, teacher_velocity, fast_velocity, slow_velocity):
    """T->slow EMA or fast EMA->slow EMA, using interval-weighted mean velocity."""
    first = torch.empty_like(state)
    for mask, velocity in ((teacher_mask, teacher_velocity), (~teacher_mask, fast_velocity)):
        if mask.any():
            first[mask] = velocity(state[mask], times[mask, 0], slice_conditions(encoded, mask)).to(state.dtype)
    h1 = (times[:, 1]-times[:, 0])[:, None, None]
    h2 = (times[:, 2]-times[:, 1])[:, None, None]
    if not ((h1 > 0).all() and (h2 > 0).all()):
        raise ValueError("shortcut intervals must be strictly increasing")
    middle = state + h1*first
    second = slow_velocity(middle, times[:, 1], encoded)
    return ((h1*first+h2*second)/(h1+h2)).detach()


def scfm_loss(student, teacher, fast, slow, target_latent, noise, encoded, schedule):
    time = schedule.times[:, 0]
    state = (1-time[:, None, None])*noise + time[:, None, None]*target_latent
    target = (target_latent-noise).detach().clone()
    shortcut = ~schedule.anchor
    if shortcut.any():
        target[shortcut] = dual_target(
            state[shortcut], schedule.times[shortcut], schedule.teacher[shortcut],
            slice_conditions(encoded, shortcut), teacher.velocity, fast.velocity, slow.velocity)
    prediction = student.velocity(state, time, encoded)
    per_sample = (prediction.float()-target.float()).square().mean(dim=(1, 2))
    # Matches the upstream unweighted MSE. The dataset's contact oversampling
    # weight is intentionally not an extra loss factor in this independent run.
    loss = per_sample.mean()
    stats = {"loss": float(loss.detach()), "teacher_windows": int((shortcut & schedule.teacher).sum()),
             "self_windows": int((shortcut & ~schedule.teacher).sum()),
             "anchor_windows": int(schedule.anchor.sum())}
    return loss, stats


def frozen_snapshot(student):
    # Separate modules avoid the upstream LoRA param.data swaps during autograd.
    return copy.deepcopy(student).eval().requires_grad_(False)


@torch.no_grad()
def update_flow_ema(snapshot, student, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must lie in [0,1)")
    prefixes = tuple(name+"." for name in student.FLOW_MODULES if name != "source_model")
    source = dict(student.named_parameters())
    for name, parameter in snapshot.named_parameters():
        if name.startswith(prefixes):
            parameter.lerp_(source[name].detach(), 1-decay)
    buffers = dict(student.named_buffers())
    for name, value in snapshot.named_buffers():
        if name.startswith(prefixes):
            value.copy_(buffers[name])
