"""Distributional metrics for CARS-WM conditional future samples."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate(samples, target):
    if samples.ndim != 4 or target.ndim != 3:
        raise ValueError("samples/target must have shapes [B,K,H,D] and [B,H,D]")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("samples and target have incompatible shapes")


def energy_score(samples, target):
    _validate(samples, target)
    flat = samples.flatten(2)
    truth = target.flatten(1)
    observation = torch.linalg.vector_norm(flat - truth[:, None], dim=-1).mean(dim=1)
    pairwise = torch.cdist(flat, flat, p=2).mean(dim=(1, 2))
    return observation - 0.5 * pairwise


def min_ade_fde(samples, target):
    _validate(samples, target)
    distance = torch.linalg.vector_norm(samples - target[:, None], dim=-1)
    ade = distance.mean(dim=-1).min(dim=1).values
    fde = distance[:, :, -1].min(dim=1).values
    return ade, fde


def sample_spread(samples):
    if samples.ndim != 4:
        raise ValueError("samples must have shape [B,K,H,D]")
    flat = samples.flatten(2)
    return torch.cdist(flat, flat, p=2).mean(dim=(1, 2))


def marginal_coverage(samples, target):
    _validate(samples, target)
    lower = torch.quantile(samples.float(), 0.05, dim=1)
    upper = torch.quantile(samples.float(), 0.95, dim=1)
    return ((target >= lower) & (target <= upper)).float().mean(dim=(1, 2))


def contact_metrics(probability_samples, target):
    if probability_samples.ndim != 4 or probability_samples.shape[-1] < 2:
        raise ValueError("contact probabilities must have shape [B,K,H,C], C >= 2")
    class_count = probability_samples.shape[-1]
    labels = target.squeeze(-1).round().long()
    if torch.any(labels < 0) or torch.any(labels >= class_count):
        raise ValueError("contact target contains a phase outside [0, C)")
    probability = probability_samples.mean(dim=1).clamp_min(1.0e-8)
    nll = -probability.gather(-1, labels[..., None]).squeeze(-1).mean(dim=1)
    one_hot = F.one_hot(labels, num_classes=class_count).to(probability.dtype)
    brier = (probability - one_hot).square().sum(dim=-1).mean(dim=1)
    prediction = probability.argmax(dim=-1)
    f1_values = []
    for phase in range(class_count):
        true_positive = ((prediction == phase) & (labels == phase)).sum(dim=1).float()
        predicted = (prediction == phase).sum(dim=1).float()
        support = (labels == phase).sum(dim=1).float()
        precision = true_positive / predicted.clamp_min(1.0)
        recall = true_positive / support.clamp_min(1.0)
        f1_values.append(2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8))
    macro_f1 = torch.stack(f1_values, dim=1).mean(dim=1)

    def onset(value):
        contact = value == class_count - 1
        indices = torch.arange(value.shape[1], device=value.device)[None].expand_as(value)
        sentinel = torch.full_like(indices, value.shape[1])
        return torch.where(contact, indices, sentinel).min(dim=1).values

    onset_error = (onset(prediction) - onset(labels)).abs().float()
    return {
        "contact_nll": nll,
        "contact_brier": brier,
        "contact_macro_f1": macro_f1,
        "contact_onset_error_frames": onset_error,
        "contact_entropy": -(probability * probability.log()).sum(dim=-1).mean(dim=1),
    }


def distribution_metrics(samples_by_stream, targets_by_stream, contact_probability=None, contact_target=None):
    sample_state = torch.cat(
        [samples_by_stream[key] for key in samples_by_stream], dim=-1
    )
    target_state = torch.cat(
        [targets_by_stream[key] for key in samples_by_stream], dim=-1
    )
    ade, fde = min_ade_fde(sample_state, target_state)
    result = {
        "energy_score": energy_score(sample_state, target_state),
        "min_ade": ade,
        "min_fde": fde,
        "sample_spread": sample_spread(sample_state),
        "coverage_90": marginal_coverage(sample_state, target_state),
    }
    for key, samples in samples_by_stream.items():
        result[f"{key}_energy_score"] = energy_score(
            samples, targets_by_stream[key]
        )
        result[f"{key}_coverage_90"] = marginal_coverage(
            samples, targets_by_stream[key]
        )
    if contact_probability is not None and contact_target is not None:
        result.update(contact_metrics(contact_probability, contact_target))
    return result


__all__ = [
    "contact_metrics",
    "distribution_metrics",
    "energy_score",
    "marginal_coverage",
    "min_ade_fde",
    "sample_spread",
]
