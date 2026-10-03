"""Inversion-trained conditional Gaussian source for latent CARS-WM."""
from __future__ import annotations

import math

import torch
from torch import nn


class ConditionalGaussianLatentSource(nn.Module):
    """A diagonal Gaussian over complete normalized future latent trajectories.

    The zero-initialized output makes a newly attached source exactly N(0, I).
    """

    def __init__(self, condition_dim, horizon, latent_dim, hidden_dim=256,
                 min_log_std=-4.0, max_log_std=1.5):
        super().__init__()
        if horizon < 1 or latent_dim < 1 or hidden_dim < 1 or min_log_std >= max_log_std:
            raise ValueError("invalid conditional Gaussian dimensions or log-std bounds")
        self.horizon, self.latent_dim = int(horizon), int(latent_dim)
        self.min_log_std, self.max_log_std = float(min_log_std), float(max_log_std)
        self.network = nn.Sequential(
            nn.LayerNorm(condition_dim), nn.Linear(condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * horizon * latent_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def parameters_for(self, condition):
        mean, log_std = self.network(condition).chunk(2, dim=-1)
        shape = (condition.shape[0], self.horizon, self.latent_dim)
        return mean.reshape(shape), log_std.reshape(shape).clamp(self.min_log_std, self.max_log_std)

    def transform(self, condition, noise, *, temperature=1.0):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("source temperature must be finite and nonnegative")
        mean, log_std = self.parameters_for(condition)
        if noise.shape[-2:] != mean.shape[-2:] or noise.shape[0] != mean.shape[0]:
            raise ValueError("source noise shape does not match conditional Gaussian")
        if noise.ndim == 4:
            mean, log_std = mean[:, None], log_std[:, None]
        elif noise.ndim != 3:
            raise ValueError("source noise must be [B,H,D] or [B,S,H,D]")
        return mean + temperature * log_std.exp() * noise

    def nll_per_sample(self, condition, target):
        mean, log_std = self.parameters_for(condition)
        if target.shape != mean.shape:
            raise ValueError("inverse source target shape mismatch")
        standardized = (target - mean) * torch.exp(-log_std)
        return (0.5 * standardized.square() + log_std + 0.5 * math.log(2 * math.pi)).mean((1, 2))

    def kl_per_sample(self, condition):
        mean, log_std = self.parameters_for(condition)
        return (0.5 * (mean.square() + torch.exp(2 * log_std) - 1 - 2 * log_std)).mean((1, 2))


@torch.no_grad()
def invert_heun(velocity, target, encoded, *, steps, iterations=8):
    """Invert the model's discrete Heun sampler with a fixed-point solve.

    Returns the source and a forward cycle error; inversion quality must be
    checked before its outputs are used as Gaussian supervision.
    """
    if steps < 1 or iterations < 1:
        raise ValueError("steps and iterations must be positive")
    dt = 1.0 / steps
    next_state = target.float()
    for step in reversed(range(steps)):
        state = next_state
        for _ in range(iterations):
            first = velocity(state, step * dt, encoded)
            second = velocity(state + dt * first, (step + 1) * dt, encoded)
            state = next_state - 0.5 * dt * (first + second)
        next_state = state
    source = next_state
    reconstructed = source
    for step in range(steps):
        first = velocity(reconstructed, step * dt, encoded)
        second = velocity(reconstructed + dt * first, (step + 1) * dt, encoded)
        reconstructed = reconstructed + 0.5 * dt * (first + second)
    cycle_rmse = (reconstructed - target).float().square().mean((1, 2)).sqrt()
    return source, cycle_rmse
