#!/usr/bin/env python3
"""Visualize the nominal-torque decomposition of a Contact WM checkpoint.

The checkpoint still predicts the measured/total torque ``tau_total``.  This
script applies the deterministic post-processing proposed in the experiment
notes to both the data and the model output::

    tau_nom = tau_total - Kp * (q_des - q) - Kd * (dq_des - dq)

``q_des`` comes from the checkpoint's direct action chunk.  The dataset has no
separate desired-velocity stream, so ``dq_des`` defaults to the finite
difference of that action chunk at its recorded action-refresh timestamps.
The action target is then held on the 100 Hz future timeline.  Both choices
are explicit in ``summary.json`` and can be changed from the command line.

Example (from the repository root)::

    ./.conda-env/bin/python test/visualize_contact_wm_nominal_torque.py

The default run samples one rollout per episode (up to 12 rollouts) and writes
PNG figures, compressed arrays, and a JSON report below ``test/outputs``.
Use ``--max-total-rollouts 0`` to process every valid window.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# LeRobot/datasets and matplotlib create small lock/cache files during a read.
# Keep those files in a writable temporary location on shared workstations
# where the user's home cache may be mounted read-only.
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/pinn_hf_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinn_mplconfig")
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel
from train.nomalizer import Normalizer


LOG = logging.getLogger("contact_wm_nominal_torque")
JOINT_LABELS = tuple(f"joint {index}" for index in range(1, 8))
KP_DEFAULT = np.asarray([3.0, 3.0, 2.0, 3.5, 2.0, 2.0, 3.0])
KD_DEFAULT = np.asarray([0.1, 0.1, 0.25, 0.25, 0.1, 0.1, 0.1])


def resolve_checkpoint(path: Path) -> Path:
    """Resolve an explicit checkpoint, or the newest epoch in a directory."""

    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    candidates = sorted(path.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"checkpoint directory contains no .pt files: {path}")
    epochs = []
    steps = []
    scored = []
    for candidate in candidates:
        match = re.match(r"epoch_(\d+)\.pt$", candidate.name)
        if match:
            epochs.append((int(match.group(1)), candidate))
        match = re.match(r"step_(\d+)\.pt$", candidate.name)
        if match:
            steps.append((int(match.group(1)), candidate))
        match = re.search(r"_(?:val_loss|loss)_([0-9.eE+-]+)\.pt$", candidate.name)
        if match:
            scored.append((float(match.group(1)), candidate))
    if epochs:
        return max(epochs, key=lambda item: (item[0], item[1].name))[1].resolve()
    if steps:
        return max(steps, key=lambda item: (item[0], item[1].name))[1].resolve()
    if scored:
        return min(scored, key=lambda item: (item[0], item[1].name))[1].resolve()
    latest = path / "latest.pt"
    if latest.is_file():
        return latest.resolve()
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise ValueError(f"cannot choose one checkpoint from {path}; pass a .pt file")


def _checkpoint_normalizer(checkpoint: Mapping) -> Normalizer:
    payload = checkpoint.get("normalizer")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("stats"), Mapping):
        raise KeyError("checkpoint is missing normalizer.stats")
    stats = {
        str(key): {
            str(statistic): torch.as_tensor(value).detach().cpu()
            for statistic, value in values.items()
        }
        for key, values in payload["stats"].items()
    }
    return Normalizer(stats, eps=float(payload.get("eps", 1.0e-6)))


def _denormalize(
    normalizer: Normalizer,
    mode: str | None,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if mode is None or key not in normalizer.stats:
        return value
    function = getattr(normalizer, f"{mode}_denormalize", None)
    if function is None:
        raise ValueError(f"unsupported normalize_mode {mode!r}")
    return function(key, value.float())


def _resolve_relative_data_paths(config: dict) -> dict:
    """Make checkpoint-relative dataset roots independent of the cwd."""

    resolved = copy.deepcopy(config)
    train_data = resolved.get("train_data")
    if isinstance(train_data, Mapping):
        sources = train_data.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                root = source.get("root", source.get("path"))
                if root is not None:
                    root_path = Path(str(root)).expanduser()
                    source["root"] = str(
                        root_path if root_path.is_absolute() else (REPO_ROOT / root_path).resolve()
                    )
    data_config = resolved.get("dataloader")
    if isinstance(data_config, Mapping) and data_config.get("root") is not None:
        root_path = Path(str(data_config["root"])).expanduser()
        data_config["root"] = str(
            root_path if root_path.is_absolute() else (REPO_ROOT / root_path).resolve()
        )
    return resolved


def _parse_vector(value: str, name: str) -> np.ndarray:
    """Parse a scalar or seven comma/space separated gain values."""

    tokens = [token for token in re.split(r"[,\s]+", str(value).strip()) if token]
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numbers") from exc
    if values.size == 1:
        values = np.repeat(values, 7)
    if values.size != 7 or not np.isfinite(values).all() or np.any(values < 0):
        raise argparse.ArgumentTypeError(
            f"{name} must be one non-negative scalar or seven finite values"
        )
    return values


def _episode_indices(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    try:
        result = {int(token) for token in re.split(r"[,\s]+", value) if token}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--episodes must be integer ids") from exc
    return result


def select_rollouts(
    dataset: ContactWorldModelDataset,
    *,
    episodes: set[int] | None,
    stride: int,
    start_offset: int,
    max_per_episode: int | None,
    max_total: int | None,
) -> list[int]:
    """Select sample indices without crossing episode boundaries."""

    if stride <= 0 or start_offset < 0:
        raise ValueError("stride must be positive and start_offset non-negative")
    # ``ContactWorldModelDataset`` already records the owning episode for every
    # valid high-rate row.  Group once instead of rescanning all windows for
    # every episode (the three training sources contain ~600k rows).
    grouped: dict[int, list[int]] = {}
    for dataset_index, raw in enumerate(dataset.valid_indices):
        episode = dataset.raw_idx_to_episode.get(int(raw))
        if episode is None:
            continue
        episode_id = int(episode.get("episode_index", -1))
        grouped.setdefault(episode_id, []).append(dataset_index)
    selected: list[int] = []
    seen_episodes: set[int] = set()
    for fallback, episode in enumerate(dataset.episodes):
        episode_id = int(episode.get("episode_index", fallback))
        if episodes is not None and episode_id not in episodes:
            continue
        seen_episodes.add(episode_id)
        candidates = grouped.get(episode_id, [])[start_offset::stride]
        if max_per_episode is not None:
            candidates = candidates[:max_per_episode]
        selected.extend(candidates)
        if max_total is not None and len(selected) >= max_total:
            return selected[:max_total]
    if episodes is not None:
        missing = sorted(episodes - seen_episodes)
        if missing:
            raise ValueError(f"requested episode ids do not exist: {missing}")
    if not selected:
        raise ValueError("rollout selection produced no valid samples")
    return selected


def _action_to_future(
    action: torch.Tensor,
    action_times_ns: torch.Tensor,
    future_times_ns: torch.Tensor,
    *,
    interpolation: str,
) -> torch.Tensor:
    """Upsample an action chunk to future state timestamps."""

    if action.ndim != 3 or action_times_ns.ndim != 2 or future_times_ns.ndim != 2:
        raise ValueError("action/timestamps have unexpected dimensions")
    if action.shape[:2] != action_times_ns.shape:
        raise ValueError("action and action timestamps disagree")
    if action.shape[0] != future_times_ns.shape[0]:
        raise ValueError("future timestamps disagree with action batch")
    if interpolation not in {"previous", "nearest", "linear"}:
        raise ValueError("interpolation must be previous, nearest, or linear")

    query = future_times_ns.to(dtype=torch.float64)
    anchors = action_times_ns.to(dtype=torch.float64)
    if interpolation == "previous":
        # ``right=True`` makes a query exactly at an action refresh use the
        # newly refreshed command rather than the preceding one.
        right = torch.searchsorted(anchors, query, right=True)
        index = (right - 1).clamp(0, action.shape[1] - 1)
        return torch.gather(action, 1, index[..., None].expand(-1, -1, action.shape[-1]))
    right = torch.searchsorted(anchors, query, right=False)
    right = right.clamp(0, action.shape[1] - 1)
    left = (right - 1).clamp(0, action.shape[1] - 1)
    left_value = torch.gather(action, 1, left[..., None].expand(-1, -1, action.shape[-1]))
    right_value = torch.gather(action, 1, right[..., None].expand(-1, -1, action.shape[-1]))
    left_time = torch.gather(anchors, 1, left)
    right_time = torch.gather(anchors, 1, right)
    if interpolation == "nearest":
        choose_right = (right_time - query).abs() < (query - left_time).abs()
        return torch.where(choose_right[..., None], right_value, left_value)
    denominator = (right_time - left_time).clamp_min(1.0)
    alpha = ((query - left_time) / denominator).clamp(0.0, 1.0).to(dtype=action.dtype)
    return left_value + alpha[..., None] * (right_value - left_value)


def _derive_action_velocity(action: torch.Tensor, action_times_ns: torch.Tensor) -> torch.Tensor:
    """Estimate desired velocity from the action chunk in physical units."""

    if action.shape[1] < 2:
        return torch.zeros_like(action)
    dt = torch.diff(action_times_ns.to(dtype=action.dtype), dim=1) * 1.0e-9
    dt = dt.clamp_min(1.0e-6)
    velocity = torch.zeros_like(action)
    velocity[:, 1:] = torch.diff(action, dim=1) / dt[..., None]
    velocity[:, 0] = velocity[:, 1]
    return velocity


def decompose_nominal_torque(
    tau_total: torch.Tensor,
    q: torch.Tensor,
    dq: torch.Tensor,
    q_des: torch.Tensor,
    dq_des: torch.Tensor,
    kp: np.ndarray,
    kd: np.ndarray,
) -> torch.Tensor:
    """Apply the proposed total-to-nominal torque transformation."""

    kp_tensor = torch.as_tensor(kp, device=tau_total.device, dtype=tau_total.dtype)
    kd_tensor = torch.as_tensor(kd, device=tau_total.device, dtype=tau_total.dtype)
    return tau_total - kp_tensor * (q_des - q) - kd_tensor * (dq_des - dq)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {"mean_abs": math.nan, "rms": math.nan, "p95_abs": math.nan, "max_abs": math.nan}
    return {
        "mean_abs": float(np.mean(np.abs(flat))),
        "rms": float(np.sqrt(np.mean(flat * flat))),
        "p95_abs": float(np.percentile(np.abs(flat), 95.0)),
        "max_abs": float(np.max(np.abs(flat))),
    }


def _smoothness(values: np.ndarray, time_s: np.ndarray) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"mean_abs_d_tau": math.nan, "rms_dd_tau": math.nan}
    first = np.gradient(values, time_s, axis=0, edge_order=1)
    second = np.gradient(first, time_s, axis=0, edge_order=1) if values.shape[0] >= 3 else np.zeros_like(first)
    return {
        "mean_abs_d_tau": float(np.nanmean(np.abs(first))),
        "rms_dd_tau": float(np.sqrt(np.nanmean(second * second))),
    }


def _plot_rollout(record: Mapping, output_path: Path, dpi: int) -> None:
    time_s = record["time_s"]
    phase = record["contact_phase"]
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, squeeze=False)
    for joint, axis in enumerate(axes.flat[:7]):
        axis.plot(time_s, record["tau_total_gt"][:, joint], color="tab:blue", label="total data", linewidth=1.5)
        axis.plot(time_s, record["tau_total_pred"][:, joint], color="#E6A700", label="total WM", linewidth=1.4)
        axis.plot(time_s, record["tau_nom_gt"][:, joint], color="tab:green", label="nominal data", linewidth=1.5)
        axis.plot(time_s, record["tau_nom_pred"][:, joint], color="tab:red", label="nominal WM", linewidth=1.4)
        axis.set_title(JOINT_LABELS[joint])
        axis.set_ylabel("N m")
        axis.grid(True, alpha=0.28)
        if joint == 0:
            axis.legend(loc="best", fontsize=8)
    contact_axis = axes.flat[7]
    contact_axis.step(time_s, phase, where="post", color="#5B4B8A", linewidth=1.8)
    contact_axis.set_yticks([0, 1, 2], ["free", "align", "contact"])
    contact_axis.set_ylabel("phase")
    contact_axis.set_title("ground-truth contact phase")
    contact_axis.grid(True, alpha=0.28)
    for axis in axes[-1]:
        axis.set_xlabel("future time from anchor (s)")
    figure.suptitle(
        f"episode {record['episode_index']} / anchor row {record['sample_idx']} — total vs nominal torque"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_target_alignment(record: Mapping, output_path: Path, dpi: int) -> None:
    time_s = record["time_s"]
    figure, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, squeeze=False)
    for joint, axis in enumerate(axes.flat[:7]):
        axis.plot(time_s, record["q_des"][:, joint], color="#5B4B8A", label="q_des", linewidth=1.5)
        axis.plot(time_s, record["q_gt"][:, joint], color="tab:blue", label="q data", linewidth=1.3)
        axis.plot(time_s, record["q_pred"][:, joint], color="#E6A700", label="q WM", linewidth=1.3)
        axis.set_title(f"{JOINT_LABELS[joint]} position")
        axis.set_ylabel("rad")
        axis.grid(True, alpha=0.28)
        if joint == 0:
            axis.legend(loc="best", fontsize=8)
    phase_axis = axes.flat[7]
    phase_axis2 = phase_axis.twinx()
    phase_axis.plot(time_s, record["dq_des"][:, 0], color="#5B4B8A", label="dq_des j1", linewidth=1.5)
    phase_axis.plot(time_s, record["dq_gt"][:, 0], color="tab:blue", label="dq data j1", linewidth=1.3)
    phase_axis.plot(time_s, record["dq_pred"][:, 0], color="#E6A700", label="dq WM j1", linewidth=1.3)
    phase_axis2.step(time_s, record["contact_phase"], where="post", color="#444444", alpha=0.35, label="phase")
    phase_axis.set_title("velocity target example (joint 1) and contact")
    phase_axis.set_ylabel("rad/s")
    phase_axis2.set_ylabel("phase")
    phase_axis.grid(True, alpha=0.28)
    for axis in axes[-1]:
        axis.set_xlabel("future time from anchor (s)")
    figure.suptitle("Action target alignment used by nominal-torque decomposition")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _plot_aggregate(records: Sequence[Mapping], metrics: Mapping, output_path: Path, dpi: int) -> None:
    if not records:
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)
    time_s = records[0]["time_s"]
    for key, label, color in (
        ("tau_total_gt", "total data", "tab:blue"),
        ("tau_total_pred", "total WM", "#E6A700"),
        ("tau_nom_gt", "nominal data", "tab:green"),
        ("tau_nom_pred", "nominal WM", "tab:red"),
    ):
        values = np.stack([np.mean(np.abs(record[key]), axis=1) for record in records])
        axes[0, 0].plot(time_s, values.mean(axis=0), label=label, color=color, linewidth=1.8)
    axes[0, 0].set_title("mean absolute torque over future time")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].set_ylabel("N m")
    axes[0, 0].grid(True, alpha=0.28)
    axes[0, 0].legend(fontsize=8)

    joints = np.arange(1, 8)
    nominal_mae = metrics["errors"]["nominal"]["mae_per_joint"]
    total_mae = metrics["errors"]["total"]["mae_per_joint"]
    width = 0.38
    axes[0, 1].bar(joints - width / 2, total_mae, width, label="total", color="#E6A700")
    axes[0, 1].bar(joints + width / 2, nominal_mae, width, label="nominal", color="tab:red")
    axes[0, 1].set_xticks(joints, [str(value) for value in joints])
    axes[0, 1].set_title("prediction MAE by joint")
    axes[0, 1].set_xlabel("joint")
    axes[0, 1].set_ylabel("N m")
    axes[0, 1].grid(True, axis="y", alpha=0.28)
    axes[0, 1].legend(fontsize=8)

    for key, label, color in (
        ("nominal_gt", "nominal data", "tab:green"),
        ("nominal_pred", "nominal WM", "tab:red"),
        ("total_gt", "total data", "tab:blue"),
        ("total_pred", "total WM", "#E6A700"),
    ):
        phase_values = metrics["phase_mean_abs"].get(key, [])
        if phase_values:
            axes[1, 0].plot([0, 1, 2], phase_values, marker="o", label=label, color=color)
    axes[1, 0].set_xticks([0, 1, 2], ["free", "align", "contact"])
    axes[1, 0].set_title("mean absolute torque by contact phase")
    axes[1, 0].set_ylabel("N m")
    axes[1, 0].grid(True, alpha=0.28)
    axes[1, 0].legend(fontsize=8)

    smooth_names = ["nominal_gt", "nominal_pred", "total_gt", "total_pred"]
    smooth_first = [metrics["smoothness"][name]["mean_abs_d_tau"] for name in smooth_names]
    smooth_second = [metrics["smoothness"][name]["rms_dd_tau"] for name in smooth_names]
    x = np.arange(len(smooth_names))
    axes[1, 1].bar(x - width / 2, smooth_first, width, label="mean |d tau/dt|", color="#4C78A8")
    axes[1, 1].bar(x + width / 2, smooth_second, width, label="RMS d2 tau/dt2", color="#F58518")
    axes[1, 1].set_xticks(x, [name.replace("_", " ") for name in smooth_names], rotation=20, ha="right")
    axes[1, 1].set_title("smoothness diagnostics")
    axes[1, 1].set_ylabel("scaled torque derivative")
    axes[1, 1].grid(True, axis="y", alpha=0.28)
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("Contact WM total-to-nominal torque validation")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def build_metrics(records: Sequence[Mapping]) -> dict:
    result: dict = {"num_rollouts": len(records), "signals": {}, "errors": {}, "smoothness": {}, "phase_mean_abs": {}}
    for name in ("tau_total_gt", "tau_total_pred", "tau_nom_gt", "tau_nom_pred"):
        values = np.concatenate([record[name] for record in records], axis=0)
        result["signals"][name] = _finite_stats(values)
        smooth_name = {
            "tau_total_gt": "total_gt",
            "tau_total_pred": "total_pred",
            "tau_nom_gt": "nominal_gt",
            "tau_nom_pred": "nominal_pred",
        }[name]
        result["smoothness"][smooth_name] = {
            key: float(np.mean([_smoothness(record[name], record["time_s"])[key] for record in records]))
            for key in ("mean_abs_d_tau", "rms_dd_tau")
        }
    for label, gt_name, pred_name in (("total", "tau_total_gt", "tau_total_pred"), ("nominal", "tau_nom_gt", "tau_nom_pred")):
        errors = np.concatenate([record[pred_name] - record[gt_name] for record in records], axis=0)
        result["errors"][label] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae_per_joint": [float(value) for value in np.mean(np.abs(errors), axis=0)],
            "rmse_per_joint": [float(value) for value in np.sqrt(np.mean(errors * errors, axis=0))],
        }
    for phase in (0, 1, 2):
        mask_values = {"nominal_gt": [], "nominal_pred": [], "total_gt": [], "total_pred": []}
        for record in records:
            mask = record["contact_phase"] == phase
            if not np.any(mask):
                continue
            for key, source in (("nominal_gt", "tau_nom_gt"), ("nominal_pred", "tau_nom_pred"), ("total_gt", "tau_total_gt"), ("total_pred", "tau_total_pred")):
                mask_values[key].append(float(np.mean(np.abs(record[source][mask]))))
        for key, values in mask_values.items():
            result["phase_mean_abs"].setdefault(key, []).append(float(np.mean(values)) if values else math.nan)
    onset_deltas = []
    for record in records:
        onset = np.flatnonzero(record["contact_phase"] >= 2)
        if onset.size == 0:
            continue
        onset = int(onset[0])
        window = max(1, min(10, onset, record["tau_nom_gt"].shape[0] - onset))
        before = np.mean(np.abs(record["tau_nom_gt"][onset - window : onset]))
        after = np.mean(np.abs(record["tau_nom_gt"][onset : onset + window]))
        onset_deltas.append(float(after - before))
    result["contact_onset_nominal_data_delta_mean_abs"] = float(np.mean(onset_deltas)) if onset_deltas else math.nan
    result["contact_onset_nominal_data_count"] = len(onset_deltas)
    return result


def _to_device(batch: Mapping, device: torch.device, keys: Sequence[str]) -> dict:
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in keys
        if key in batch and torch.is_tensor(batch[key])
    }


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise KeyError("checkpoint is missing config")
    config = _resolve_relative_data_paths(dict(checkpoint_config))
    normalizer = _checkpoint_normalizer(checkpoint)
    mode = (config.get("dataloader") or {}).get("normalize_mode")
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = ContactWorldModel(config)
    weight_key = "model" if args.weights == "ema" else "model_raw"
    state_dict = checkpoint.get(weight_key)
    if not isinstance(state_dict, Mapping):
        raise KeyError(f"checkpoint does not contain {weight_key}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    dataset = ContactWorldModelDataset(config, normalizer=normalizer, compute_normalizer=False)
    if (dataset.history_horizon, dataset.future_horizon, dataset.action_condition_horizon) != (
        model.history_horizon,
        model.future_horizon,
        model.action_condition_horizon,
    ):
        raise ValueError("dataset horizons do not match checkpoint model")

    selected = select_rollouts(
        dataset,
        episodes=_episode_indices(args.episodes),
        stride=dataset.future_horizon if args.sample_stride is None else args.sample_stride,
        start_offset=args.start_offset,
        max_per_episode=None if args.max_rollouts_per_episode == 0 else args.max_rollouts_per_episode,
        max_total=None if args.max_total_rollouts == 0 else args.max_total_rollouts,
    )
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    kp = _parse_vector(args.kp, "--kp")
    kd = _parse_vector(args.kd, "--kd")
    flow_steps = model.flow_inference_steps if args.flow_steps is None else args.flow_steps
    flow_solver = model.flow_solver if args.flow_solver is None else args.flow_solver
    records = []
    selected_raw = {int(dataset.valid_indices[index]) for index in selected}
    episode_lookup = {}
    for fallback, episode in enumerate(dataset.episodes):
        episode_id = int(episode.get("episode_index", fallback))
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for raw in selected_raw:
            if start <= raw < end:
                episode_lookup[raw] = episode_id
    with torch.no_grad():
        for batch_number, cpu_batch in enumerate(loader):
            model_batch = _to_device(cpu_batch, device, ("q", "dq", "delta_q", "tau", "action", "action_mask"))
            source_noise = None
            if args.seed is not None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(args.seed) + batch_number * 1009)
                reference = model_batch[model.inputs[0]]
                source_noise = torch.randn(
                    (reference.shape[0], model.future_horizon, model.flow_dim),
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=device, dtype=reference.dtype)
            prediction = model.predict(model_batch, steps=flow_steps, solver=flow_solver, source_noise=source_noise)
            predicted = {
                key: _denormalize(normalizer, mode, key, prediction[f"{key}_pred"]).detach().cpu()
                for key in ("q", "dq", "tau")
            }
            q_gt = cpu_batch["q_future_raw"].float()
            dq_gt = cpu_batch["dq_future_raw"].float()
            tau_gt = cpu_batch["tau_future_raw"].float()
            q_des = _action_to_future(
                cpu_batch["expert_action_chunk_abs"].float(),
                cpu_batch["action_chunk_timestamp_ns"],
                cpu_batch["future_timestamp_ns"],
                interpolation=args.action_interpolation,
            )
            if args.dqdes_mode == "zero":
                dq_action = torch.zeros_like(cpu_batch["expert_action_chunk_abs"], dtype=torch.float32)
            else:
                dq_action = _derive_action_velocity(
                    cpu_batch["expert_action_chunk_abs"].float(),
                    cpu_batch["action_chunk_timestamp_ns"],
                )
            dq_des = _action_to_future(
                dq_action,
                cpu_batch["action_chunk_timestamp_ns"],
                cpu_batch["future_timestamp_ns"],
                interpolation=args.action_interpolation,
            )
            tau_nom_gt = decompose_nominal_torque(tau_gt, q_gt, dq_gt, q_des, dq_des, kp, kd)
            tau_nom_pred = decompose_nominal_torque(predicted["tau"], predicted["q"], predicted["dq"], q_des, dq_des, kp, kd)
            for index in range(q_gt.shape[0]):
                sample_idx = int(cpu_batch["sample_idx"][index])
                anchor_ns = int(cpu_batch["history_timestamp_ns"][index, -1])
                time_s = (cpu_batch["future_timestamp_ns"][index].numpy().astype(np.float64) - anchor_ns) * 1.0e-9
                record = {
                    "sample_idx": sample_idx,
                    "episode_index": episode_lookup[sample_idx],
                    "time_s": time_s,
                    "contact_phase": cpu_batch["contact_future"][index].reshape(-1).numpy().astype(np.int64),
                    "q_des": q_des[index].numpy(),
                    "q_gt": q_gt[index].numpy(),
                    "q_pred": predicted["q"][index].numpy(),
                    "dq_des": dq_des[index].numpy(),
                    "dq_gt": dq_gt[index].numpy(),
                    "dq_pred": predicted["dq"][index].numpy(),
                    "tau_total_gt": tau_gt[index].numpy(),
                    "tau_total_pred": predicted["tau"][index].numpy(),
                    "tau_nom_gt": tau_nom_gt[index].numpy(),
                    "tau_nom_pred": tau_nom_pred[index].numpy(),
                }
                records.append(record)
                sample_dir = output_dir / f"episode_{record['episode_index']:03d}" / f"anchor_{sample_idx:06d}"
                _plot_rollout(record, sample_dir / "torque_total_vs_nominal.png", args.dpi)
                _plot_target_alignment(record, sample_dir / "target_alignment.png", args.dpi)
                if args.save_arrays:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_dir / "nominal_torque.npz", **record)

    metrics = build_metrics(records)
    _plot_aggregate(records, metrics, output_dir / "aggregate_nominal_torque.png", args.dpi)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "weights": args.weights,
        "device": str(device),
        "flow_steps": int(flow_steps),
        "flow_solver": str(flow_solver),
        "dataset_root": str(dataset.root),
        "dataset_repo_id": dataset.repo_id,
        "num_rollouts": len(records),
        "selection": {
            "episodes": sorted(_episode_indices(args.episodes)) if _episode_indices(args.episodes) is not None else None,
            "sample_stride": dataset.future_horizon if args.sample_stride is None else args.sample_stride,
            "start_offset": args.start_offset,
            "max_rollouts_per_episode": args.max_rollouts_per_episode,
            "max_total_rollouts": args.max_total_rollouts,
        },
        "decomposition": {
            "formula": "tau_nom = tau_total - Kp * (q_des - q) - Kd * (dq_des - dq)",
            "kp": kp.tolist(),
            "kd": kd.tolist(),
            "q_des_source": "expert_action_chunk_abs (action.ee_pose)",
            "action_interpolation": args.action_interpolation,
            "dq_des_source": args.dqdes_mode,
            "dq_des_units": "rad/s",
            "future_state_rate_hz": float(dataset.high_fps),
        },
        "metrics": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=True)
    LOG.info("saved %d rollouts and figures to %s", len(records), output_dir)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/contact_world_model_opd_sweep/20260829_005818/teacher/checkpoints/epoch_0001200.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("test/outputs/contact_wm_nominal_torque"))
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--flow-solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--seed", type=int, default=1234, help="fixed Gaussian source seed; use --seed -1 for stochastic sources")
    parser.add_argument("--episodes", default=None, help="comma/space separated episode ids")
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--max-rollouts-per-episode", type=int, default=1, help="0 means no per-episode cap")
    parser.add_argument("--max-total-rollouts", type=int, default=12, help="0 means process all selected windows")
    parser.add_argument("--kp", default=','.join(str(value) for value in KP_DEFAULT), help="one scalar or seven comma-separated Kp values")
    parser.add_argument("--kd", default=','.join(str(value) for value in KD_DEFAULT), help="one scalar or seven comma-separated Kd values")
    parser.add_argument("--action-interpolation", choices=("previous", "nearest", "linear"), default="previous")
    parser.add_argument("--dqdes-mode", choices=("action-difference", "zero"), default="action-difference")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-arrays", dest="save_arrays", action="store_false")
    parser.set_defaults(save_arrays=True)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.flow_steps is not None and args.flow_steps <= 0:
        parser.error("batch size/flow steps must be positive and num-workers non-negative")
    if args.start_offset < 0 or args.max_rollouts_per_episode < 0 or args.max_total_rollouts < 0:
        parser.error("selection limits and start offset must be non-negative")
    if args.sample_stride is not None and args.sample_stride <= 0:
        parser.error("sample stride must be positive")
    if args.seed is not None and args.seed < 0:
        args.seed = None
    _parse_vector(args.kp, "--kp")
    _parse_vector(args.kd, "--kd")
    _episode_indices(args.episodes)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    summary = run(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
