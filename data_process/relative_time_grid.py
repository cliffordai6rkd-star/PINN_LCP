"""One shared robot-time grid for datasets and online request anchors.

Subtract int64 nanoseconds BEFORE division. Ties round away from zero. With
no explicit grid, quantize only the first action anchor, then use nominal R.
Never clamp d0. Check both individual intervals and accumulated clock drift.
Explicit per-token grids can describe plans crossing scheduling windows.
"""
from dataclasses import asdict, dataclass
import math

import torch


class GridMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class RelativeTimeGrid:
    state_rate_hz: float = 100.0
    action_rate_hz: float = 25.0
    action_interval_tolerance_ns: int = 4_000_000
    state_interval_tolerance_ns: int = 4_000_000
    action_drift_tolerance_ns: int = 5_000_000

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.state_rate_hz, self.action_rate_hz)):
            raise ValueError("grid rates must be finite and positive")
        if abs(self.state_rate_hz / self.action_rate_hz - self.ratio) > 1e-9:
            raise ValueError("action period must be an integer number of external ticks")
        if any(v < 0 for v in (self.action_interval_tolerance_ns,
                              self.state_interval_tolerance_ns, self.action_drift_tolerance_ns)):
            raise ValueError("grid tolerances must be nonnegative")

    @property
    def tick_ns(self):
        return round(1e9 / self.state_rate_hz)

    @property
    def ratio(self):
        return round(self.state_rate_hz / self.action_rate_hz)

    @classmethod
    def from_config(cls, config):
        data = config.get("dataloader") or {}
        return cls(state_rate_hz=float(data.get("high_fps", 100)),
                   action_rate_hz=float(data.get("expert_fps", 25)),
                   **(data.get("relative_time_grid") or {}))

    def contract(self):
        return {"schema": "relative_grid_v1", **asdict(self), "tick_ns": self.tick_ns,
                "action_ticks": self.ratio, "origin": "request_history_last_observation",
                "rounding": "nearest_ties_away_from_zero_int64",
                "fallback": "quantize_first_anchor_then_nominal_cadence",
                "explicit": "per_token_grid_positions_minus_request_anchor_grid",
                "state_stride": "retain_external_ticks", "phase_clamping": False}

    def quantize(self, delta_ns):
        if delta_ns.dtype != torch.int64:
            raise GridMetadataError("timestamps/deltas must be int64 nanoseconds")
        magnitude = (delta_ns.abs() + self.tick_ns // 2) // self.tick_ns
        return delta_ns.sign() * magnitude

    def valid_windows(self, history_ns, action_ns, future_ns=None, *, action_start_offset=1,
                      history_valid=None, action_indices=None):
        """Return per-window validity; masks exempt replicated history padding."""
        for value in (history_ns, action_ns, future_ns):
            if value is not None and (value.dtype != torch.int64 or value.ndim != 2):
                raise GridMetadataError("timing metadata must be int64 [B,T]")
        valid = torch.ones(history_ns.shape[0], dtype=torch.bool, device=history_ns.device)
        intervals = torch.diff(action_ns, dim=1)
        period = self.ratio * self.tick_ns
        valid &= ((intervals > 0) & ((intervals - period).abs() <= self.action_interval_tolerance_ns)).all(1)
        offsets = torch.arange(action_ns.shape[1], device=action_ns.device) * period
        drift = action_ns - action_ns[:, :1] - offsets
        valid &= (drift.abs() <= self.action_drift_tolerance_ns).all(1)
        if action_indices is not None:
            valid &= (torch.diff(action_indices, dim=1) == 1).all(1)
        if action_start_offset > 0:
            # A rounded zero is legal; an actually earlier anchor is not.
            valid &= action_ns[:, 0] >= history_ns[:, -1]
        state_intervals = torch.diff(history_ns, dim=1)
        pairs = torch.ones_like(state_intervals, dtype=torch.bool)
        if history_valid is not None:
            pairs = history_valid[:, :-1] & history_valid[:, 1:]
        good = (state_intervals > 0) & ((state_intervals-self.tick_ns).abs() <= self.state_interval_tolerance_ns)
        valid &= (good | ~pairs).all(1)
        if future_ns is not None:
            intervals = torch.diff(torch.cat((history_ns[:, -1:], future_ns), 1), dim=1)
            valid &= ((intervals > 0) & ((intervals-self.tick_ns).abs() <= self.state_interval_tolerance_ns)).all(1)
        return valid

    def positions(self, *, history_ns=None, action_ns=None, future_horizon,
                  history_horizon=None, action_start_offset=1, history_valid=None,
                  action_indices=None, explicit=None):
        if explicit is not None:
            # Trust supplied physical-grid metadata, not a local token index.
            anchor = explicit["anchor_grid_position"]
            result = {key + "_grid_positions": explicit[key + "_grid_positions"] - anchor[..., None]
                      for key in ("history", "action", "future")}
            for key, value in result.items():
                if value.dtype != torch.int64 or value.ndim != 2:
                    raise GridMetadataError(f"{key} must be int64 [B,T]")
                if not (torch.diff(value, dim=1) > 0).all():
                    raise GridMetadataError(f"{key} must increase per token")
            if not (result["history_grid_positions"][:, -1] == 0).all():
                raise GridMetadataError("history must end at request anchor position zero")
            if not (result["future_grid_positions"] > 0).all():
                raise GridMetadataError("future positions must follow the observation")
            if action_start_offset > 0 and (result["action_grid_positions"][:, 0] < 0).any():
                raise GridMetadataError("next action precedes request observation")
            return result
        if history_ns is None or action_ns is None:
            raise GridMetadataError("explicit grid metadata or history/action anchor timestamps are required")
        if not self.valid_windows(history_ns, action_ns, action_start_offset=action_start_offset,
                                  history_valid=history_valid, action_indices=action_indices).all():
            raise GridMetadataError("irregular cadence, clock drift, index gap or next-action anchor before history; inspect window semantics")
        b, length = history_ns.shape
        if history_horizon is not None and length != history_horizon:
            raise GridMetadataError("history timing length differs from history horizon")
        d0 = self.quantize(action_ns[:, 0] - history_ns[:, -1])
        device = history_ns.device
        return {"history_grid_positions": torch.arange(1-length, 1, device=device).expand(b, -1),
                "action_grid_positions": d0[:, None] + self.ratio * torch.arange(action_ns.shape[1], device=device),
                "future_grid_positions": torch.arange(1, future_horizon+1, device=device).expand(b, -1)}


def grid_sinusoidal(positions, hidden_dim, *, dtype):
    """A single deterministic sinusoidal map, shared by all three memories."""
    if positions.dtype != torch.int64 or positions.ndim != 2:
        raise GridMetadataError("model grid positions must be int64 [B,T]")
    frequency = torch.exp(torch.arange(0, hidden_dim, 2, device=positions.device, dtype=torch.float32)
                          * (-math.log(10000.0) / hidden_dim))
    angle = positions.float()[..., None] * frequency
    result = torch.empty(*positions.shape, hidden_dim, device=positions.device, dtype=torch.float32)
    result[..., 0::2] = angle.sin()
    result[..., 1::2] = angle[..., :hidden_dim//2].cos()
    return result.to(dtype=dtype)
