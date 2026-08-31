"""Offline rollout inference and plots for the Contact World Model.

The model is rolled out from real 100 Hz history windows with the direct
25 Hz action chunk supplied by :class:`ContactWorldModelDataset`. Predictions
are denormalized and plotted as q/dq/delta_q/tau plus contact phase. No
derived state or dynamics quantity is introduced at inference time.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel
from train.nomalizer import Normalizer


log = logging.getLogger(__name__)

SIGNAL_LABELS = {
    "q": tuple(f"q{i}" for i in range(1, 8)),
    "dq": tuple(f"dq{i}" for i in range(1, 8)),
    "delta_q": tuple(f"delta_q{i}" for i in range(1, 8)),
    "tau": tuple(f"tau{i}" for i in range(1, 8)),
    "contact": ("free", "align", "contact"),
}
SIGNAL_UNITS = {
    "q": "rad",
    "dq": "rad/s",
    "delta_q": "rad",
    "tau": "N m",
    "contact": "phase",
}


def _deep_update(destination: dict, source: Mapping) -> dict:
    """Recursively update ``destination`` without mutating ``source``."""

    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
            nested = dict(destination[key])
            destination[key] = _deep_update(nested, value)
        else:
            destination[key] = copy.deepcopy(value)
    return destination


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rollout config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise TypeError("rollout config must contain a YAML mapping")
    return dict(config)


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a Contact World Model checkpoint file or directory."""

    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    candidates = sorted(path.glob("epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"Contact World Model checkpoint directory has no epoch_*.pt files: {path}")
    return max(candidates, key=lambda item: item.name).resolve()


def _checkpoint_normalizer(checkpoint: Mapping) -> Normalizer:
    payload = checkpoint.get("normalizer")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("stats"), Mapping):
        raise KeyError("world-model checkpoint is missing normalizer.stats")
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
    normalize_mode: str | None,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if normalize_mode is None or key not in normalizer.stats:
        return value
    functions = {
        "gaussian": normalizer.gaussian_denormalize,
        "limit": normalizer.limit_denormalize,
        "quantile": normalizer.quantile_denormalize,
    }
    if normalize_mode not in functions:
        raise ValueError(f"unsupported normalize_mode {normalize_mode!r}")
    return functions[normalize_mode](key, value)


def _requested_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable")
    return device


def load_model_and_dataset(
    rollout_config: Mapping,
) -> tuple[
    Path,
    Mapping,
    dict,
    ContactWorldModel,
    ContactWorldModelDataset,
    Normalizer,
    torch.device,
]:
    """Restore checkpoint weights/normalizer and apply dataset-only overrides."""

    checkpoint_path = resolve_checkpoint_path(rollout_config["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_version") != "contact_world_model_v1" or not any(str(key).startswith("state_encoders.") for key in (checkpoint.get("model") or {})):
        raise ValueError("checkpoint is not a canonical ContactWorldModel checkpoint")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise KeyError("world-model checkpoint is missing config")
    checkpoint_config = copy.deepcopy(dict(checkpoint_config))

    inference_config = rollout_config.get("inference") or {}
    device = _requested_device(str(inference_config.get("device", "cuda:0")))
    model = ContactWorldModel(checkpoint_config)
    weights = str(inference_config.get("weights", "ema")).lower()
    if weights in {"ema", "model"}:
        state_dict = checkpoint.get("model")
    elif weights in {"raw", "model_raw"}:
        state_dict = checkpoint.get("model_raw")
    else:
        raise ValueError("inference.weights must be ema or raw")
    if not isinstance(state_dict, Mapping):
        raise KeyError(f"checkpoint does not contain requested {weights!r} weights")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    effective_config = copy.deepcopy(checkpoint_config)
    dataset_overrides = rollout_config.get("dataset") or {}
    effective_config["dataloader"] = _deep_update(
        dict(effective_config.get("dataloader") or {}),
        dataset_overrides,
    )
    normalizer = _checkpoint_normalizer(checkpoint)
    dataset = ContactWorldModelDataset(
        effective_config,
        normalizer=normalizer,
        compute_normalizer=False,
    )
    if dataset.history_horizon != model.history_horizon:
        raise ValueError(
            "inference dataset history horizon differs from checkpoint model: "
            f"{dataset.history_horizon} != {model.history_horizon}"
        )
    if dataset.future_horizon != model.future_horizon:
        raise ValueError(
            "inference dataset prediction horizon differs from checkpoint model: "
            f"{dataset.future_horizon} != {model.future_horizon}"
        )
    if dataset.action_condition_horizon != model.action_condition_horizon:
        raise ValueError(
            "inference dataset action horizon differs from checkpoint model: "
            f"{dataset.action_condition_horizon} != "
            f"{model.action_condition_horizon}"
        )
    return (
        checkpoint_path,
        checkpoint,
        effective_config,
        model,
        dataset,
        normalizer,
        device,
    )


def _episode_index(episode: Mapping, fallback: int) -> int:
    return int(episode.get("episode_index", fallback))


def select_rollout_indices(
    dataset: ContactWorldModelDataset,
    selection_config: Mapping | None,
) -> list[int]:
    """Select dataset sample indices episode-by-episode at a fixed row stride."""

    selection_config = selection_config or {}
    requested = selection_config.get("episode_indices")
    requested_set = None if requested is None else {int(value) for value in requested}
    stride = int(selection_config.get("sample_stride", dataset.future_horizon))
    start_offset = int(selection_config.get("start_offset", 0))
    maximum = selection_config.get("max_rollouts_per_episode")
    maximum = None if maximum is None else int(maximum)
    maximum_total = selection_config.get("max_total_rollouts")
    maximum_total = None if maximum_total is None else int(maximum_total)
    if stride <= 0:
        raise ValueError("selection.sample_stride must be positive")
    if start_offset < 0:
        raise ValueError("selection.start_offset must be non-negative")
    if maximum is not None and maximum <= 0:
        raise ValueError("selection.max_rollouts_per_episode must be positive or null")
    if maximum_total is not None and maximum_total <= 0:
        raise ValueError("selection.max_total_rollouts must be positive or null")

    raw_to_dataset_index = {
        int(raw_index): dataset_index
        for dataset_index, raw_index in enumerate(dataset.valid_indices)
    }
    selected = []
    found_episodes = set()
    for fallback, episode in enumerate(dataset.episodes):
        episode_index = _episode_index(episode, fallback)
        if requested_set is not None and episode_index not in requested_set:
            continue
        found_episodes.add(episode_index)
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        candidates = [
            raw_to_dataset_index[raw_index]
            for raw_index in dataset.valid_indices
            if start <= int(raw_index) < end
        ]
        candidates = candidates[start_offset::stride]
        if maximum is not None:
            candidates = candidates[:maximum]
        selected.extend(candidates)
        if maximum_total is not None and len(selected) >= maximum_total:
            selected = selected[:maximum_total]
            break
    if requested_set is not None:
        missing = sorted(requested_set - found_episodes)
        if missing:
            raise ValueError(f"requested episode indices do not exist: {missing}")
    if not selected:
        raise ValueError("rollout selection produced no valid samples")
    return selected


def _to_device(batch: Mapping, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class RunningSquaredError:
    def __init__(self):
        self.sum_square: dict[str, torch.Tensor] = {}
        self.count: dict[str, int] = {}

    def update(self, name: str, data: torch.Tensor, prediction: torch.Tensor):
        if data.shape != prediction.shape:
            raise ValueError(
                f"metric shape mismatch for {name}: "
                f"{tuple(data.shape)} != {tuple(prediction.shape)}"
            )
        error_square = (prediction.detach().cpu() - data.detach().cpu()).square()
        per_dimension = error_square.reshape(-1, error_square.shape[-1]).sum(dim=0)
        if name not in self.sum_square:
            self.sum_square[name] = per_dimension
            self.count[name] = int(error_square.numel() // error_square.shape[-1])
        else:
            self.sum_square[name] += per_dimension
            self.count[name] += int(error_square.numel() // error_square.shape[-1])

    def result(self) -> dict:
        result = {}
        for name, sum_square in self.sum_square.items():
            per_dimension = torch.sqrt(sum_square / self.count[name])
            result[name] = {
                "rmse": float(torch.sqrt(sum_square.sum() / (self.count[name] * sum_square.numel()))),
                "rmse_per_dimension": [float(value) for value in per_dimension],
                "num_time_points": self.count[name],
            }
        return result


def plot_signal_comparison(
    time_s: np.ndarray,
    data: np.ndarray,
    prediction: np.ndarray,
    *,
    signal: str,
    title: str,
    output_path: str | Path,
    dpi: int = 160,
    columns: int = 2,
):
    """Save one multi-axis signal figure with fixed blue/yellow semantics."""

    if data.shape != prediction.shape or data.ndim != 2:
        raise ValueError("data and prediction must have equal [T, D] shapes")
    if time_s.ndim != 1 or time_s.shape[0] != data.shape[0]:
        raise ValueError("time_s must have shape [T]")
    labels = SIGNAL_LABELS.get(signal)
    if labels is None or len(labels) != data.shape[1]:
        labels = tuple(f"{signal}{index + 1}" for index in range(data.shape[1]))
    columns = max(1, min(int(columns), data.shape[1]))
    rows = int(math.ceil(data.shape[1] / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.2 * columns, 2.45 * rows),
        sharex=True,
        squeeze=False,
    )
    for index, axis in enumerate(axes.flat):
        if index >= data.shape[1]:
            axis.set_visible(False)
            continue
        axis.plot(
            time_s,
            data[:, index],
            color="tab:blue",
            linewidth=1.8,
            label="data",
        )
        axis.plot(
            time_s,
            prediction[:, index],
            color="#E6A700",
            linewidth=1.8,
            label="pred",
        )
        axis.set_title(labels[index])
        axis.set_ylabel(SIGNAL_UNITS.get(signal, "value"))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("future time from anchor (s)")
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def _sample_episode_lookup(dataset: ContactWorldModelDataset) -> dict[int, int]:
    result = {}
    for fallback, episode in enumerate(dataset.episodes):
        episode_index = _episode_index(episode, fallback)
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        for raw_index in dataset.valid_indices:
            raw_index = int(raw_index)
            if start <= raw_index < end:
                result[raw_index] = episode_index
    return result


def _write_json(path: Path, value: Mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run_rollout(rollout_config: Mapping) -> dict:
    (
        checkpoint_path,
        checkpoint,
        effective_config,
        model,
        dataset,
        normalizer,
        device,
    ) = load_model_and_dataset(rollout_config)
    selected_indices = select_rollout_indices(
        dataset,
        rollout_config.get("selection"),
    )
    inference_config = rollout_config.get("inference") or {}
    plot_config = rollout_config.get("plot") or {}
    batch_size = int(inference_config.get("batch_size", 8))
    num_workers = int(inference_config.get("num_workers", 0))
    flow_steps = int(inference_config.get("flow_steps", model.flow_inference_steps))
    flow_solver = str(inference_config.get("flow_solver", model.flow_solver)).lower()
    if batch_size <= 0 or num_workers < 0 or flow_steps <= 0:
        raise ValueError("batch_size/flow_steps must be positive and num_workers non-negative")

    output_dir = Path(
        plot_config.get("output_dir", "outputs/contact_world_model_rollout")
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(plot_config.get("dpi", 160))
    columns = int(plot_config.get("subplot_columns", 2))
    save_arrays = bool(plot_config.get("save_arrays", True))
    configured_signals = [
        str(value) for value in plot_config.get(
            "signals", ["q", "dq", "delta_q", "tau", "contact"]
        )
    ]
    unknown_signals = sorted(set(configured_signals) - set(SIGNAL_LABELS))
    if unknown_signals:
        raise ValueError(f"unknown plot signals: {unknown_signals}")

    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    normalize_mode = (effective_config.get("dataloader") or {}).get(
        "normalize_mode"
    )
    episode_lookup = _sample_episode_lookup(dataset)
    metrics = RunningSquaredError()
    model_inference_seconds = 0.0
    rollout_count = 0

    log.info(
        "rollout checkpoint=%s samples=%d device=%s flow=%s/%d",
        checkpoint_path,
        len(selected_indices),
        device,
        flow_solver,
        flow_steps,
    )
    with torch.no_grad():
        for cpu_batch in loader:
            batch = _to_device(cpu_batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model.predict(
                batch,
                steps=flow_steps,
                solver=flow_solver,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_inference_seconds += time.perf_counter() - started

            prediction = {
                key: _denormalize(
                    normalizer,
                    normalize_mode,
                    key,
                    output[f"{key}_pred"],
                )
                for key in ("q", "dq", "delta_q", "tau")
            }
            prediction["contact"] = output["contact_state_pred"]

            data = {
                "q": batch["q_future_raw"],
                "dq": batch["dq_future_raw"],
                "delta_q": batch["delta_q_future_raw"],
                "tau": batch["tau_future_raw"],
                "contact": batch["contact_future"],
            }
            for signal in configured_signals:
                metrics.update(signal, data[signal], prediction[signal])

            for batch_index in range(batch["q"].shape[0]):
                raw_index = int(cpu_batch["sample_idx"][batch_index])
                episode_index = episode_lookup[raw_index]
                sample_dir = (
                    output_dir
                    / f"episode_{episode_index:03d}"
                    / f"anchor_{raw_index:06d}"
                )
                future_ns = cpu_batch["future_timestamp_ns"][batch_index].numpy()
                anchor_ns = int(cpu_batch["history_timestamp_ns"][batch_index, -1])
                time_s = (future_ns.astype(np.float64) - anchor_ns) * 1.0e-9
                arrays = {"time_s": time_s}
                for signal in configured_signals:
                    data_value = data[signal][batch_index].detach().cpu().numpy()
                    prediction_value = (
                        prediction[signal][batch_index].detach().cpu().numpy()
                    )
                    arrays[f"{signal}_data"] = data_value
                    arrays[f"{signal}_pred"] = prediction_value
                    plot_signal_comparison(
                        time_s,
                        data_value,
                        prediction_value,
                        signal=signal,
                        title=(
                            f"episode {episode_index}, anchor row {raw_index}, "
                            f"{signal} rollout"
                        ),
                        output_path=sample_dir / f"{signal}.png",
                        dpi=dpi,
                        columns=columns,
                    )
                if save_arrays:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(sample_dir / "rollout.npz", **arrays)
                rollout_count += 1

    metric_result = metrics.result()
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "weights": str(inference_config.get("weights", "ema")),
        "device": str(device),
        "dataset_root": str(dataset.root),
        "dataset_repo_id": dataset.repo_id,
        "high_fps": dataset.high_fps,
        "history_horizon": dataset.history_horizon,
        "prediction_horizon": dataset.future_horizon,
        "action_condition_horizon": dataset.action_condition_horizon,
        "flow_steps": flow_steps,
        "flow_solver": flow_solver,
        "num_rollouts": rollout_count,
        "model_inference_total_s": model_inference_seconds,
        "model_inference_ms_per_rollout": 1000.0 * model_inference_seconds / rollout_count,
        "metrics": metric_result,
        "state_prediction": "direct q/dq/delta_q/tau outputs",
    }
    _write_json(output_dir / "metrics.json", summary)
    with (output_dir / "resolved_rollout_config.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(dict(rollout_config), stream, allow_unicode=True, sort_keys=False)
    log.info("saved %d rollouts and metrics to %s", rollout_count, output_dir)
    return summary


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run and plot 100 Hz Contact World Model rollouts."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/inference_cfg/contact_world_model_rollout.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    args = parse_args(argv)
    summary = run_rollout(load_yaml_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
