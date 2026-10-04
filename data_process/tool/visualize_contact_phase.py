"""Plot one episode's contact signal with contact/alignment/free phase regions.

Example::

    python data_process/tool/visualize_contact_phase.py \
        --h5 /path/to/episode_0001.h5 \
        --config config/train_cfg/pretrain/xarm/cwm_erase_board_100hz_40step.yaml

The default metric is tau_ext_l1. Contact is strictly above contact_threshold;
the preceding precontact_duration_s is alignment, with contact taking priority.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch
import yaml

from model.pinn_model.contact_gate import ContactGateConfig, contact_phase_labels_from_signal


PHASE_NAMES = ("free motion", "alignment", "contact")
PHASE_COLORS = ("#8dd3c7", "#fee8a5", "#f6a6a6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize free motion/alignment/contact phases for one H5 episode."
    )
    parser.add_argument("--h5", type=Path, required=True, help="Episode .h5/.hdf5 file")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional training YAML; supplies metric, contact threshold and timing",
    )
    parser.add_argument(
        "--metric", choices=("tau_ext_l1", "tau_ext_l2", "force_xyz_l2", "wrench_l2"),
        default=None,
    )
    parser.add_argument("--threshold", type=float, default=None, help="Strict contact threshold")
    parser.add_argument("--precontact-seconds", type=float, default=None, help="Alignment prefix in seconds")
    parser.add_argument("--timestamp-path", default="teleop/timestamp_us")
    parser.add_argument(
        "--timestamp-unit", choices=("s", "ms", "us", "ns"), default="us",
        help="Unit of --timestamp-path (default: us)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/contact_phase/contact_phase.png")
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--show", action="store_true", help="Open an interactive window")
    return parser.parse_args()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def _signal_from_h5(h5, metric: str) -> np.ndarray:
    if metric.startswith("tau_ext"):
        values = np.asarray(h5["teleop/tau_ext_cal"], dtype=np.float64)
        if metric == "tau_ext_l1":
            return np.abs(values).sum(axis=-1)
        return np.linalg.norm(values, axis=-1)
    values = np.asarray(h5["teleop/wrench_cal"], dtype=np.float64)
    if metric == "force_xyz_l2":
        values = values[..., :3]
    return np.linalg.norm(values, axis=-1)


def _phase_spans(time_s: np.ndarray, labels: np.ndarray):
    """Yield contiguous ``(start, end, phase)`` spans for background fills."""
    if labels.size == 0:
        return
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], labels.size]
    for start, end in zip(starts, ends):
        # Extend the final span to the last sample for a visually complete band.
        right = time_s[end] if end < time_s.size else time_s[-1]
        if end == labels.size and time_s.size > 1:
            right += time_s[-1] - time_s[-2]
        yield float(time_s[start]), float(right), int(labels[start])


def main() -> None:
    args = parse_args()
    # Import lazily so ``--help`` and argument validation remain usable in
    # headless environments where matplotlib's binary wheel is unavailable.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not args.h5.is_file():
        raise FileNotFoundError(args.h5)
    config = _load_config(args.config)
    contact_config = ContactGateConfig.from_config(config)
    metric = args.metric or (contact_config.metric if args.config else 'tau_ext_l1')
    contact_config = replace(contact_config,
        contact_threshold=contact_config.contact_threshold if args.threshold is None else args.threshold,
        precontact_duration_s=contact_config.precontact_duration_s if args.precontact_seconds is None else args.precontact_seconds)
    contact_config.validate()
    threshold = contact_config.contact_threshold
    import h5py

    with h5py.File(args.h5, "r") as h5:
        signal = _signal_from_h5(h5, metric)
        timestamps = np.asarray(h5[args.timestamp_path], dtype=np.float64).reshape(-1)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"signal must be a non-empty vector, got {signal.shape}")
    if timestamps.size != signal.size:
        raise ValueError(
            f"timestamp length {timestamps.size} does not match signal length {signal.size}"
        )
    # Convert timestamps to seconds and make the x-axis relative to frame 0.
    timestamp_scale = {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6, "ns": 1.0e-9}[args.timestamp_unit]
    time_s = (timestamps - timestamps[0]) * timestamp_scale
    if not np.isfinite(signal).all() or not np.isfinite(time_s).all():
        raise ValueError("signal and timestamps must be finite")
    labels = contact_phase_labels_from_signal(
        torch.from_numpy(signal.astype(np.float32)), [(0, len(signal))], contact_config,
        timestamps_s=time_s,
    )[:, 0].numpy().astype(np.int8)

    fig, (ax, phase_ax) = plt.subplots(
        2, 1, figsize=(12, 5.2), sharex=True,
        gridspec_kw={"height_ratios": (5, 0.55), "hspace": 0.04},
    )
    for left, right, phase in _phase_spans(time_s, labels):
        ax.axvspan(left, right, color=PHASE_COLORS[phase], alpha=0.30, lw=0)
        phase_ax.axvspan(left, right, color=PHASE_COLORS[phase], alpha=0.95, lw=0)
    ax.plot(time_s, signal, color="#303030", lw=1.1, label=metric)
    ax.axhline(threshold, color="#c44e52", ls="--", lw=1.0, label=f"contact > {threshold:g}")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    phase_ax.set_ylim(0, 1)
    phase_ax.set_yticks([])
    phase_ax.set_xlabel("time (s)")
    phase_ax.spines[["top", "right", "left"]].set_visible(False)
    phase_ax.tick_params(axis="x", length=3)
    legend = [
        Line2D([], [], color="#303030", lw=1.1, label=metric),
        Line2D([], [], color="#c44e52", ls="--", lw=1.0, label=f"contact > {threshold:g}"),
        *[Patch(facecolor=color, alpha=0.65, label=name) for name, color in zip(PHASE_NAMES, PHASE_COLORS)],
    ]
    ax.legend(handles=legend, loc="upper left", ncol=3, framealpha=0.9)
    title = args.title or f"{args.h5.name} — contact phase"
    ax.set_title(title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"saved: {args.output}")
    for phase, name in enumerate(PHASE_NAMES):
        count = int(np.count_nonzero(labels == phase))
        print(f"{name:>12}: {count:6d} frames ({100.0 * count / labels.size:5.1f}%)")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
