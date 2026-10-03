"""Physical-time diagnostics, independent of flow integration time.

Execution slices deliberately reproduce Nero prefetch's array indices [d:d+E].
The dataset forecast index 0 is anchor+dt; see deployment documentation.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from train.carswm_metrics import contact_confusion_matrix, contact_macro_f1_from_confusion


def execution_slice(horizon, delay_steps, execute_steps, mode='prefetch'):
    for key, value in [('delay_steps', delay_steps), ('execute_steps', execute_steps)]:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f'{key} must be an integer')
    if mode not in ('prefetch', 'openloop') or delay_steps < 0 or execute_steps <= 0:
        raise ValueError('invalid execution window')
    start = delay_steps if mode == 'prefetch' else 0
    if start + execute_steps > horizon:
        raise ValueError(f'execution window [{start}:{start + execute_steps}] exceeds horizon={horizon}')
    return slice(start, start + execute_steps)


def local_interval_loss(prediction, x, teacher_endpoint, h, mode, loss_fn):
    if not math.isfinite(h) or h <= 0:
        raise ValueError('flow interval h must be positive')
    if mode == 'velocity':
        return loss_fn(prediction, ((teacher_endpoint - x) / h).detach())
    if mode == 'endpoint':
        return loss_fn(x + h * prediction, teacher_endpoint.detach())
    raise ValueError('local_loss_mode must be velocity or endpoint')


def physical_derivatives(samples, future_time, order):
    """[B,K,H,D] finite differences, using actual dt and midpoint dt for D2."""
    if order not in (1, 2):
        raise ValueError('derivative order must be 1 or 2')
    intervals = torch.diff(future_time.double(), dim=1).to(samples.dtype)
    if intervals.shape[1] < order or torch.any(intervals <= 0) or not torch.isfinite(intervals).all():
        raise ValueError('physical derivatives require increasing, finite future timestamps')
    velocity = torch.diff(samples, dim=2) / intervals[:, None, :, None]
    if order == 1:
        return velocity
    midpoint_dt = (intervals[:, 1:] + intervals[:, :-1]) / 2
    return torch.diff(velocity, dim=2) / midpoint_dt[:, None, :, None]


def physical_frame_errors(samples, target, future_time, stream):
    errors = {f'{stream}_mse': (samples - target[:, None]).square().mean((1, 3))}
    if stream == 'q':
        for order in (1, 2):
            if samples.shape[2] > order:
                ds = physical_derivatives(samples, future_time, order)
                dt = physical_derivatives(target[:, None], future_time, order)
                errors[f'q_d{order}_mse'] = (ds - dt).square().mean((1, 3))
    return errors


class ContactAccumulator:
    """Global marginal calibration/classification; same-noise pairing is separate."""
    def __init__(self, classes, bins=10):
        self.classes, self.bins = classes, bins
        self.confusion = torch.zeros(classes, classes, dtype=torch.float64)
        self.bin_count = torch.zeros(bins, dtype=torch.float64)
        self.bin_confidence = torch.zeros(bins, dtype=torch.float64)
        self.bin_correct = torch.zeros(bins, dtype=torch.float64)
        self.nll = self.brier = self.count = 0.0
        self.events = {key: [0.0, 0, 0, 0] for key in ('onset', 'release')}

    def update(self, probability_samples, labels, future_time, history_contact=None):
        p = probability_samples.float().mean(1).clamp_min(1e-8)
        y = labels.squeeze(-1).long()
        self.confusion += contact_confusion_matrix(probability_samples, labels).double().cpu()
        self.nll += float((-p.gather(-1, y[..., None]).log()).sum())
        self.brier += float((p - F.one_hot(y, self.classes)).square().sum(-1).sum())
        self.count += y.numel()
        confidence, prediction = p.max(-1)
        correct = prediction.eq(y).double()
        ids = (confidence * self.bins).long().clamp_max(self.bins - 1)
        for i in range(self.bins):
            mask = ids == i
            self.bin_count[i] += int(mask.sum())
            self.bin_confidence[i] += float(confidence[mask].sum())
            self.bin_correct[i] += float(correct[mask].sum())
        # Match the first event per window; report missing/extra separately.
        # Last historical label makes a first-forecast-frame transition visible.
        contact_id = self.classes - 1
        for b in range(y.shape[0]):
            truth, pred = y[b].eq(contact_id), prediction[b].eq(contact_id)
            if history_contact is None:
                t0, p0 = truth[:1], pred[:1]
            else:
                t0 = history_contact[b].reshape(-1)[-1:].eq(contact_id)
                p0 = t0
            truth_prev, pred_prev = torch.cat((t0, truth[:-1])), torch.cat((p0, pred[:-1]))
            for name, truth_mask, pred_mask in (
                ('onset', ~truth_prev & truth, ~pred_prev & pred),
                ('release', truth_prev & ~truth, pred_prev & ~pred),
            ):
                ti, pi = truth_mask.nonzero().flatten(), pred_mask.nonzero().flatten()
                e = self.events[name]
                if len(ti) and len(pi):
                    e[0] += float((future_time[b, ti[0]] - future_time[b, pi[0]]).abs())
                    e[1] += 1
                elif len(ti):
                    e[2] += 1
                elif len(pi):
                    e[3] += 1

    def finalize(self):
        if not self.count:
            return {}
        valid = self.bin_count > 0
        ece = ((self.bin_confidence[valid] - self.bin_correct[valid]).abs().sum() / self.count)
        tp = self.confusion.diag()
        result = {'nll': self.nll / self.count, 'brier': self.brier / self.count,
                  'ece': float(ece), 'accuracy': float(tp.sum() / self.count),
                  'macro_f1': float(contact_macro_f1_from_confusion(self.confusion)), 'frames': self.count}
        for c in range(self.classes):
            result[f'class_{c}_precision'] = float(tp[c] / self.confusion[:, c].sum().clamp_min(1))
            result[f'class_{c}_recall'] = float(tp[c] / self.confusion[c].sum().clamp_min(1))
        for name, (total, matched, missing, extra) in self.events.items():
            if matched:
                result[f'{name}_matched_mae_s'] = total / matched
            result.update({f'{name}_matched': matched, f'{name}_missing': missing, f'{name}_extra': extra})
        return result


def adjacent_prediction_metrics(old, new, anchor_delta, delay_steps):
    """Compare same-noise trajectories at equal absolute future timestamps.

    old/new: [B,K,H,D]. anchor_delta and delay_steps are external frame counts.
    No gradient regularizer is implied by a correction caused by new feedback.
    """
    if anchor_delta <= 0 or anchor_delta >= old.shape[2]:
        raise ValueError('anchor_delta must leave an overlapping prediction interval')
    overlap = min(old.shape[2] - anchor_delta, new.shape[2])
    if delay_steps < 0 or delay_steps >= overlap:
        raise ValueError('takeover delay outside overlapping forecasts')
    delta = old[:, :, anchor_delta:anchor_delta + overlap] - new[:, :, :overlap]
    return {'overlap_q_mse_rad2': delta.square().mean((1, 2, 3)),
            'takeover_q_mse_rad2': delta[:, :, delay_steps].square().mean((1, 2)),
            'takeover_q_max_abs_rad': delta[:, :, delay_steps].abs().flatten(1).amax(1)}
