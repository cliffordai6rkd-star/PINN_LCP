"""Frozen NEXT checkpoint inference on the original Nero recorder timeline."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from data_process.causal_data_filter import filter_episode_values
from model.pinn_model.latent_pretrained import convert_scale, validate_normalizer
from model.tau_other_sequence import build_tau_other_sequence_model


class NeroTorqueTeacher:
    """Keep checkpoint preprocessing, normalization and real 50 Hz histories."""

    def __init__(self, checkpoint, *, device="cpu", batch_size=512):
        self.path = Path(checkpoint).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        config = payload["config"]
        model_config, data_config = config["model"], config["dataloader"]
        expected = {"architecture": "lstm", "inputs": ["q", "dq", "delta_q"],
                    "target_key": "tau", "output_dim": 7,
                    "history_mode": "stateless_sliding_window"}
        for key, value in expected.items():
            if model_config.get(key) != value:
                raise ValueError(f"Nero teacher model.{key} must be {value!r}")
        self.rate = float(payload["sample_rate_hz"])
        self.horizon = int(data_config["horizon"])
        if self.rate != 50 or self.horizon < 1 or batch_size < 1:
            raise ValueError("Nero teacher requires a positive history/batch and a 50 Hz checkpoint")
        self.normalizer = copy.deepcopy(payload["normalizer"])
        if self.normalizer.get("normalize_mode") != "gaussian":
            raise ValueError("Nero teacher requires its embedded Gaussian normalizer")
        for key in ("q", "dq", "delta_q", "tau"):
            if key not in self.normalizer["normalize_lowdim_keys"]:
                raise ValueError(f"Nero teacher normalizer does not cover {key}")
            validate_normalizer(self.normalizer, key, 7)
        self.filters = copy.deepcopy(payload["dataloader_filters"])
        self.model = build_tau_other_sequence_model(config).to(device).eval().requires_grad_(False)
        # 'model' is the selected deployment/EMA snapshot, matching repository inference.
        self.model.load_state_dict(payload["model"], strict=True)
        self.device, self.batch_size = torch.device(device), int(batch_size)
        self.contract = {"checkpoint": str(self.path),
            "checkpoint_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "snapshot": "model", "model": expected, "sample_rate_hz": self.rate,
            "history_horizon": self.horizon, "history_mode": "stateless_sliding_window",
            "normalization": {key: value for key, value in self.normalizer.items() if key != "stats"},
            "filters": self.filters, "input_sampling": "every_second_raw_100hz_frame_episode_local",
            "prediction_alignment": "linear_50hz_to_raw_timestamps_with_last_prediction_held_at_tail",
            "residual": "tau_measured_minus_tau_free", "warmup": "invalid_until_first_complete_history",
            "velocity": "recorded_sign_corrected_velocity_no_additional_sign_flip"}

    @torch.inference_mode()
    def predict(self, timestamps_ns, q, dq, q_cmd, tau):
        timestamps = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
        values = {key: np.asarray(value)
                  for key, value in zip(("q", "dq", "q_cmd", "tau"), (q, dq, q_cmd, tau))}
        n = len(timestamps)
        if n < 2 or any(v.shape != (n, 7) or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nero teacher needs finite aligned [N,7] state/command/torque arrays")
        dt = np.diff(timestamps)*1e-9
        if np.any(dt <= 0) or abs(float(np.median(dt))-.01) > .001:
            raise ValueError("Nero teacher needs increasing original 100 Hz timestamps")
        times = (timestamps-timestamps[0]).astype(np.float64)*1e-9
        segments = np.r_[0, np.flatnonzero(dt > .03)+1, n]
        free = np.zeros((n, 7), dtype=np.float32)
        measured = np.zeros_like(free)
        valid = np.zeros(n, dtype=bool)
        for start, end in zip(segments[:-1], segments[1:]):
            t = times[start:end]
            # The actual recorder q_cmd defines tracking error, independent of held action chunks.
            columns = {"q": values["q"][start:end].astype(np.float32), "dq": values["dq"][start:end].astype(np.float32),
                       "delta_q": (values["q_cmd"][start:end]-values["q"][start:end]).astype(np.float32),
                       "tau": values["tau"][start:end].astype(np.float32)}
            for key, spec in self.filters.items():
                if spec.get("enabled", False):
                    columns[key] = filter_episode_values(t, columns[key], spec.get("operations", []))
            native = np.arange(0, end-start, 2)
            if len(native) < self.horizon:
                continue
            inputs = {key: convert_scale(key, torch.as_tensor(columns[key][native], device=self.device), self.normalizer)
                      for key in ("q", "dq", "delta_q")}
            predictions = []
            for offset in range(self.horizon-1, len(native), self.batch_size):
                last = torch.arange(offset, min(offset+self.batch_size, len(native)), device=self.device)
                indices = last[:, None]+torch.arange(1-self.horizon, 1, device=self.device)
                output = self.model({key: value[indices] for key, value in inputs.items()})["tau_other_pred"]
                predictions.append(convert_scale("tau", output.float(), self.normalizer, inverse=True).cpu().numpy())
            predictions = np.concatenate(predictions)
            prediction_times = t[native[self.horizon-1:]]
            ready = t >= prediction_times[0]
            rows = np.flatnonzero(ready)
            aligned = np.stack([np.interp(t[ready], prediction_times, predictions[:, j]) for j in range(7)], axis=1)
            free[start+rows] = aligned.astype(np.float32)
            measured[start:end] = columns["tau"]
            valid[start+rows] = True
        external = np.zeros_like(free)
        external[valid] = measured[valid]-free[valid]
        if not np.isfinite(free).all() or not np.isfinite(external).all():
            raise FloatingPointError("Nonfinite Nero torque predictions")
        return {"tau_free": free, "tau_ext": external, "tau_measured": measured, "valid_context": valid}
