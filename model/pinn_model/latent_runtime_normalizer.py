"""Validated, device-resident normalization constants for latent WM inference."""
from __future__ import annotations

from collections.abc import Mapping
import copy

import torch

from model.pinn_model.latent_pretrained import validate_normalizer


class PreparedNormalizer:
    """Snapshot and validate statistics once before entering the control loop.

    Normalization casts constants to the input dtype, as ``Normalizer`` does.
    Inverse conversion intentionally retains the statistics' original dtype.
    The object belongs to one device; prepare another after moving the model.
    """

    _FIELDS = {"gaussian": ("mean", "std"), "limit": ("min", "max"),
               "quantile": ("q01", "q99")}
    _DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    def __init__(self, envelope, dimensions: Mapping[str, int], device):
        if not isinstance(envelope, Mapping):
            raise ValueError("input normalizer contract is unavailable")
        self._dimensions = dict(dimensions)
        if any(not isinstance(width, int) or isinstance(width, bool) or width < 1
               for width in self._dimensions.values()):
            raise ValueError("normalizer dimensions must be positive integers")
        self.mode = envelope.get("normalize_mode")
        self._keys = frozenset(key for key in self._dimensions
                               if self.mode is not None and
                               key in (envelope.get("normalize_lowdim_keys") or []))
        if self._keys and self.mode not in self._FIELDS:
            raise ValueError(f"unsupported normalization {self.mode}")
        self.eps = copy.deepcopy(envelope.get("eps", 1e-6))
        # CPU validation prevents value-dependent device synchronization later.
        snapshot = {"normalize_mode": self.mode, "eps": self.eps, "stats": {}}
        for key in self._keys:
            original = (envelope.get("stats") or {}).get(key) or {}
            snapshot["stats"][key] = {
                field: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for field in self._FIELDS[self.mode]
                for value in (original.get(field),)
            }
            validate_normalizer(snapshot, key, self._dimensions[key])
        # Resolve unindexed accelerators (e.g. cuda) to their actual device.
        self.device = torch.empty(0, device=device).device
        self._inverse_constants = {
            key: tuple(stats[field].to(self.device) for field in self._FIELDS[self.mode])
            for key, stats in snapshot["stats"].items()
        }
        self._normalize_constants = {
            dtype: {key: tuple(value.to(dtype=dtype) for value in constants)
                    for key, constants in self._inverse_constants.items()}
            for dtype in self._DTYPES
        }

    def convert(self, key, value, *, inverse=False):
        """Use the original arithmetic without per-call validation or copies."""
        if key not in self._dimensions:
            raise ValueError(f"unknown prepared normalizer stream: {key}")
        if not torch.is_tensor(value) or value.ndim < 1 or value.shape[-1] != self._dimensions[key]:
            raise ValueError(f"{key} must have last dimension {self._dimensions[key]}")
        if value.device != self.device:
            raise ValueError(f"{key} must be on prepared device {self.device}, got {value.device}")
        if key not in self._keys:
            return value
        if inverse and self.mode == "quantile":
            raise ValueError("clipped quantile WM inputs cannot be inverted for a pretrained LSTM")
        if value.dtype not in self._normalize_constants:
            raise ValueError(f"normalized {key} requires a supported floating dtype, got {value.dtype}")
        first, second = (self._inverse_constants[key] if inverse else
                         self._normalize_constants[value.dtype][key])
        if self.mode == "gaussian":
            return value * (second + self.eps) + first if inverse else (value - first) / (second + self.eps)
        if inverse:
            return (value + 1) * (second - first + self.eps) / 2 + first
        result = 2 * (value - first) / (second - first + self.eps) - 1
        return result.clamp(-1.0, 1.0) if self.mode == "quantile" else result
