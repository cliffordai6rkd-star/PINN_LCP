"""Frozen offline xArm torque teacher, exported with normalization and dynamics.

Matches xarm_ws's validated zero5/50 Hz/51-frame residual BiLSTM pipeline.
Only used for episode labeling before WM training; never a WM input encoder.
"""
from __future__ import annotations

import numpy as np


class LearnedTorqueModel:
    """Portable artifact: weights, normalization, 98 coefficients and URDF XML."""

    def __init__(self, checkpoint, device='cpu', threads=None):
        import pinocchio as pin
        import torch
        from torch import nn
        from scipy.signal import butter

        if threads is not None:
            torch.set_num_threads(threads)
        cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
        expected = dict(arch='bilstm', filter='zero5', rate=50, horizon=51)
        if cp.get('format_version') != 1 or any(cp['spec'].get(k) != v for k, v in expected.items()):
            raise ValueError('Expected the exported zero5/50 Hz/51-frame residual BiLSTM artifact')

        class Regressor(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(21, 128, 2, batch_first=True, dropout=.1, bidirectional=True)
                self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 7))

            def forward(self, x):
                x, _ = self.encoder(x)
                return self.head(x[:, x.shape[1]//2])

        self.device = torch.device(device)
        self.network = Regressor().to(self.device).eval()
        self.network.load_state_dict(cp['model'])
        self.norm = {k: v.numpy() for k, v in cp['normalization'].items()}
        for key, size in [('x_mean', 21), ('x_std', 21), ('y_mean', 7), ('y_std', 7)]:
            value = self.norm[key]
            if value.shape != (size,) or not np.isfinite(value).all() or ('std' in key and np.any(value <= 0)):
                raise ValueError(f'Invalid learned torque normalization: {key}')
        self.coefficients = cp['coefficients'].numpy()
        if self.coefficients.shape != (98,) or not np.isfinite(self.coefficients).all():
            raise ValueError('Expected 98 finite dynamics/friction coefficients')
        self.physics = pin.buildModelFromXML(cp['urdf_xml'])
        if self.physics.nq != 7 or self.physics.nv != 7:
            raise ValueError('Learned torque requires the training 7-DOF URDF')
        self.data = self.physics.createData()
        self.sos = butter(4, 5, fs=100, output='sos')

    def predict(self, times, q, dq, q_cmd, tau, *, output_times=None, grid_origin=None):
        """One uninterrupted segment; optional sparse outputs reduce live CPU cost."""
        import pinocchio as pin
        import torch
        from scipy.signal import resample_poly, sosfiltfilt

        times = np.asarray(times, dtype=float)
        columns = [np.asarray(a) for a in (q, dq, q_cmd, tau)]
        if (times.ndim != 1 or len(times) < 32 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .03 + 1e-6)
                or any(a.shape != (len(times), 7) or not np.isfinite(a).all() for a in columns)):
            raise ValueError('Learned torque needs >=32 finite, increasing, contiguous 100 Hz rows')
        q, dq, q_cmd, tau = columns
        origin = times[0] if grid_origin is None else grid_origin
        t = times-origin
        # Keep the 50 Hz lattice anchored to the beginning of the segment,
        # even when the live input buffer discards old rows.
        start = int(np.ceil(t[0]*50-1e-5))*2
        stop = int(np.floor(t[-1]*100+1e-5))+1
        grid = np.arange(start, stop)*.01
        if len(grid) < 32:
            raise ValueError('Insufficient uniform-grid context')
        processed = []
        for a in (q, dq, q_cmd-q, tau):
            uniform = np.stack([np.interp(grid, t, a[:, j]) for j in range(7)], axis=1)
            filtered = sosfiltfilt(self.sos, uniform, axis=0)
            processed.append(resample_poly(filtered, 1, 2, axis=0, padtype='line').astype(np.float32))
        pq, pdq, delta, measured = processed
        grid = grid[::2]
        target = times if output_times is None else np.asarray(output_times, dtype=float)
        if target.ndim != 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0):
            raise ValueError('output_times must be finite and increasing')
        if len(target) and (target[0] < times[0] or target[-1] > times[-1]):
            raise ValueError('output_times must be within the input segment')
        # Only evaluate the 50 Hz points bracketing requested 100 Hz outputs.
        high = np.searchsorted(grid, target-origin).clip(0, len(grid)-1)
        indices = np.unique(np.concatenate((high, (high-1).clip(0))))
        if not len(indices):
            return dict(timestamp_s=target, tau_free=np.empty((0, 7)),
                        tau_measured=np.empty((0, 7)), tau_ext=np.empty((0, 7)), valid_context=np.empty(0, bool))
        features = (np.concatenate((pq, pdq, delta), axis=1)-self.norm['x_mean'])/self.norm['x_std']
        x = torch.as_tensor(features, device=self.device)
        predictions = []
        with torch.inference_mode():
            for first in range(0, len(indices), 512):
                ix = indices[first:first+512, None] + np.arange(-25, 26)[None, :]
                batch = x[torch.as_tensor(ix.clip(0, len(x)-1), device=self.device)]
                predictions.append(self.network(batch).cpu().numpy())
        pred = np.concatenate(predictions)*self.norm['y_std']+self.norm['y_mean']
        q64, dq64 = pq.astype(float), pdq.astype(float)
        ddq = np.gradient(dq64, .02, axis=0)
        prior = []
        for i in indices:
            reg = pin.computeJointTorqueRegressor(self.physics, self.data, q64[i], dq64[i], ddq[i])
            dynamics = reg@self.coefficients[:70] + pin.rnea(self.physics, self.data, q64[i], dq64[i], ddq[i])
            friction = np.stack((dq64[i], np.tanh(dq64[i]/.02), np.tanh(dq64[i]/.002), np.ones(7)), axis=1)
            prior.append(dynamics + (friction*self.coefficients[70:].reshape(7, 4)).sum(axis=1))
        pred += np.asarray(prior, dtype=np.float32)
        free100 = np.stack([np.interp(target-origin, grid[indices], pred[:, j]) for j in range(7)], axis=1)
        tau100 = np.stack([np.interp(target-origin, grid, measured[:, j]) for j in range(7)], axis=1)
        return dict(timestamp_s=target.copy(), tau_free=free100, tau_measured=tau100, tau_ext=tau100-free100,
                    valid_context=(target >= times[0]+1.) & (target <= times[-1]-1.))


def predict_episode(model, episode):
    """Split on missing data/gaps, leaving short segments and filter edges invalid."""
    t = episode['t']
    output = dict(timestamp_s=t.copy(), valid_context=np.zeros(len(t), bool),
                  **{k: np.full((len(t), 7), np.nan) for k in ('tau_free', 'tau_measured', 'tau_ext')})
    breaks = np.r_[True, (np.diff(t) <= 0) | (np.diff(t) > .03+1e-6), True]
    valid = episode['valid']
    start = None
    for i in range(len(t)+1):
        if start is not None and (i == len(t) or breaks[i] or not valid[i]):
            if i-start >= 32 and t[i-1]-t[start] >= 2.:
                result = model.predict(t[start:i], *[episode[k][start:i] for k in ('q', 'dq', 'q_cmd', 'tau')])
                for key in output:
                    output[key][start:i] = result[key]
            start = None
        if i < len(t) and valid[i] and start is None:
            start = i
    return output
