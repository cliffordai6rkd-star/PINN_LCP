"""Portable, offline q/dq/delta_q torque regressors (artifact format 2)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SequenceTorqueRegressor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        dropout = float(spec.get('dropout', .1))
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        input_dim = (14 if spec.get('no_delta') else 21) + (7 if spec.get('ddq') else 0)
        if spec['arch'] == 'mlp':
            self.encoder = None
            self.head = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, 256),
                                      nn.SiLU(), nn.Linear(256, 7))
        else:
            bidirectional = spec['arch'] == 'bilstm'
            self.encoder = nn.LSTM(input_dim, 128, 2, batch_first=True, dropout=dropout,
                                   bidirectional=bidirectional)
            self.head = nn.Sequential(nn.Linear(256 if bidirectional else 128, 256),
                                      nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 7))

    def forward(self, value):
        if self.encoder is None:
            return self.head(value[:, -1])
        value, _ = self.encoder(value)
        return self.head(value[:, value.shape[1]//2 if self.spec['arch'] == 'bilstm' else -1])


class SequenceTorqueTeacher:
    """Full-segment inference; measured tau never enters the network or prior."""
    def __init__(self, checkpoint, device='cpu'):
        self.spec = checkpoint['spec']
        spec = self.spec
        if (checkpoint.get('format_version') != 2 or spec.get('arch') not in ('lstm', 'bilstm')
                or spec.get('filter') not in ('raw', 'zero5', 'zero10') or spec.get('rate') not in (50, 100)
                or spec.get('horizon') != spec['rate'] + (1 if spec['arch'] == 'bilstm' else 0)
                or spec.get('ddq') or spec.get('no_delta')):
            raise ValueError('Unsupported format-2 torque teacher contract')
        self.device = torch.device(device)
        self.network = SequenceTorqueRegressor(spec).to(self.device).eval()
        self.network.load_state_dict(checkpoint['model'])
        self.norm = {k: v.detach().cpu().numpy() for k, v in checkpoint['normalization'].items()}
        for key, size in [('x_mean',21),('x_std',21),('y_mean',7),('y_std',7)]:
            value = self.norm[key]
            if value.shape != (size,) or not np.isfinite(value).all() or ('std' in key and np.any(value <= 0)):
                raise ValueError(f'Invalid normalization: {key}')
        self.physics = None
        if spec.get('physics_prior'):
            import pinocchio as pin
            self.coefficients = checkpoint['coefficients'].detach().cpu().numpy()
            if self.coefficients.shape != (98,) or not np.isfinite(self.coefficients).all():
                raise ValueError('Expected 98 finite dynamics/friction coefficients')
            self.physics = pin.buildModelFromXML(checkpoint['urdf_xml'])
            if self.physics.nq != 7 or self.physics.nv != 7:
                raise ValueError('Expected a seven-joint dynamics model')
        self.calibration = checkpoint.get('calibration', {})
        self.contract = checkpoint.get('contract', {})

    def predict(self, times, q, dq, q_cmd, tau, *, output_times=None, grid_origin=None):
        from scipy.signal import butter, resample_poly, sosfiltfilt
        times = np.asarray(times, dtype=float)
        arrays = [np.asarray(a) for a in (q, dq, q_cmd, tau)]
        if (times.ndim != 1 or len(times) < 32 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > .03+1e-6)
                or any(a.shape != (len(times),7) or not np.isfinite(a).all() for a in arrays)):
            raise ValueError('Torque teacher needs >=32 finite, contiguous, increasing 100 Hz rows')
        if grid_origin is not None and not np.isclose(grid_origin, times[0], rtol=0, atol=1e-9):
            raise ValueError('Format-2 teacher requires the full uninterrupted segment')
        q, dq, q_cmd, tau = arrays
        target = times if output_times is None else np.asarray(output_times, dtype=float)
        if (target.ndim != 1 or not np.isfinite(target).all() or np.any(np.diff(target) <= 0)
                or (len(target) and (target[0] < times[0] or target[-1] > times[-1]))):
            raise ValueError('output_times must increase within the input segment')
        grid = np.arange(times[0], times[-1]+1e-8, .01)
        mode, rate = self.spec['filter'], self.spec['rate']
        sos = None if mode == 'raw' else butter(4, 5 if mode == 'zero5' else 10, fs=100, output='sos')
        processed = []
        for value in (q, dq, q_cmd-q, tau):
            uniform = np.stack([np.interp(grid, times, value[:,j]) for j in range(7)], axis=1)
            filtered = uniform if sos is None else sosfiltfilt(sos, uniform, axis=0)
            if rate == 50:
                filtered = resample_poly(filtered,1,2,axis=0,padtype='line')
            processed.append(filtered.astype(np.float32))
        pq, pdq, delta, measured = processed
        grid = grid[::100//rate]
        if not len(target):
            return dict(timestamp_s=target.copy(), tau_free=np.empty((0,7)),
                        tau_measured=np.empty((0,7)),tau_ext=np.empty((0,7)),valid_context=np.empty(0,bool))
        features = (np.concatenate((pq,pdq,delta),axis=1)-self.norm['x_mean'])/self.norm['x_std']
        x = torch.as_tensor(features,device=self.device)
        horizon = self.spec['horizon']
        offsets = (np.arange(horizon)-horizon//2 if self.spec['arch']=='bilstm'
                   else np.arange(1-horizon,1))
        predictions = []
        with torch.inference_mode():
            for start in range(0,len(x),512):
                ix = np.arange(start,min(start+512,len(x)))[:,None]+offsets[None,:]
                predictions.append(self.network(x[torch.as_tensor(ix.clip(0,len(x)-1),device=self.device)]).cpu().numpy())
        pred = np.concatenate(predictions)*self.norm['y_std']+self.norm['y_mean']
        if self.physics is not None:
            import pinocchio as pin
            data = self.physics.createData()
            q64,dq64 = pq.astype(float),pdq.astype(float)
            ddq = np.gradient(dq64,1/rate,axis=0)
            prior = []
            for i in range(len(q64)):
                reg = pin.computeJointTorqueRegressor(self.physics,data,q64[i],dq64[i],ddq[i])
                dynamic = reg@self.coefficients[:70]+pin.rnea(self.physics,data,q64[i],dq64[i],ddq[i])
                fric = np.stack((dq64[i],np.tanh(dq64[i]/.02),np.tanh(dq64[i]/.002),np.ones(7)),axis=1)
                prior.append(dynamic+(fric*self.coefficients[70:].reshape(7,4)).sum(axis=1))
            pred += np.asarray(prior,dtype=np.float32)
        free = np.stack([np.interp(target,grid,pred[:,j]) for j in range(7)],axis=1)
        measured = np.stack([np.interp(target,grid,measured[:,j]) for j in range(7)],axis=1)
        return dict(timestamp_s=target.copy(),tau_free=free,tau_measured=measured,tau_ext=measured-free,
                    valid_context=(target>=times[0]+1.) & (target<=times[-1]-1.))
