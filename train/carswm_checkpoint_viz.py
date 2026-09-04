"""Fixed-format checkpoint diagnostics for CARS-WM probability forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def render_checkpoint_summary(
    path: str | Path,
    records: Sequence[Mapping],
    metric_history: Sequence[Mapping],
    current_metrics: Mapping,
    scales: Mapping[str, Sequence[float]],
    *,
    wrist_joint_index: int,
    contact_names: Sequence[str],
) -> None:
    """Render the six panels used to compare saved checkpoints."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("checkpoint visualization requires at least one record")
    focus = next(
        (record for record in records if "pre" in str(record["name"]).lower()),
        records[0],
    )
    q_samples = np.asarray(focus["samples"]["q"])
    tau_samples = np.asarray(focus["samples"]["tau"])
    q_target = np.asarray(focus["targets"]["q"])
    tau_target = np.asarray(focus["targets"]["tau"])
    horizon = q_target.shape[0]
    time_axis = np.arange(1, horizon + 1)

    figure, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax_q, ax_tau, ax_contact, ax_scatter, ax_history, ax_calibration = axes.flat

    for sample in q_samples:
        ax_q.plot(time_axis, sample[:, wrist_joint_index], color="tab:blue", alpha=0.18)
    ax_q.plot(time_axis, q_target[:, wrist_joint_index], color="black", linewidth=2.0, label="GT")
    ax_q.set(title=f"A  q trajectories ({focus['name']})", xlabel="future step", ylabel="q [rad]")
    ax_q.set_ylim(*scales["q"])
    ax_q.legend()

    for sample in tau_samples:
        ax_tau.plot(time_axis, sample[:, wrist_joint_index], color="tab:red", alpha=0.18)
    ax_tau.plot(time_axis, tau_target[:, wrist_joint_index], color="black", linewidth=2.0, label="GT")
    ax_tau.set(title=f"B  tau trajectories ({focus['name']})", xlabel="future step", ylabel="tau [Nm]")
    ax_tau.set_ylim(*scales["tau"])
    ax_tau.legend()

    probabilities = np.asarray(focus["contact_probability"])
    mean_probability = probabilities.mean(axis=0)
    for phase, name in enumerate(contact_names):
        ax_contact.plot(time_axis, mean_probability[:, phase], label=name)
    contact_target = np.asarray(focus["contact_target"]).reshape(-1)
    ax_contact.step(
        time_axis,
        contact_target / max(len(contact_names) - 1, 1),
        where="mid",
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="GT phase (scaled)",
    )
    predicted_phase = probabilities.argmax(axis=-1)
    final_phase = len(contact_names) - 1
    onset = np.where(
        (predicted_phase == final_phase).any(axis=1),
        (predicted_phase == final_phase).argmax(axis=1) + 1,
        horizon + 1,
    )
    for value in onset:
        if value <= horizon:
            ax_contact.axvline(value, color="gray", alpha=0.08, linewidth=0.8)
    gt_contact = contact_target == final_phase
    if gt_contact.any():
        ax_contact.axvline(
            int(gt_contact.argmax()) + 1,
            color="black",
            linewidth=1.5,
            label="GT contact onset",
        )
    ax_contact.set(title="C  Mean contact probability and onset draws", xlabel="future step", ylabel="probability", ylim=(0.0, 1.0))
    ax_contact.legend(fontsize=8)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(records), 1)))
    for color, record in zip(colors, records):
        q_endpoint = np.asarray(record["samples"]["q"])[:, -1, wrist_joint_index]
        tau_endpoint = np.asarray(record["samples"]["tau"])[:, -1, wrist_joint_index]
        task_name = record.get("task_name", "task")
        phase_name = record.get("phase_name", record.get("name", "phase"))
        label = f"{task_name} / {phase_name}"
        ax_scatter.scatter(
            q_endpoint,
            tau_endpoint,
            alpha=0.55,
            s=18,
            color=color,
            label=label,
        )
        ax_scatter.scatter(
            np.asarray(record["targets"]["q"])[-1, wrist_joint_index],
            np.asarray(record["targets"]["tau"])[-1, wrist_joint_index],
            marker="x",
            s=70,
            linewidth=2,
            color=color,
        )
    ax_scatter.set(
        title="D  Phase-conditioned endpoint samples",
        xlabel="q endpoint, wrist joint [rad]",
        ylabel="tau endpoint, wrist joint [Nm]",
    )
    ax_scatter.set_xlim(*scales["q"])
    ax_scatter.set_ylim(*scales["tau"])
    ax_scatter.legend(fontsize=8)

    history = list(metric_history) + [dict(current_metrics)]
    steps = [int(value["step"]) for value in history]
    for key, label in (
        ("energy_score", "ES"),
        ("min_ade", "minADE"),
        ("min_fde", "minFDE"),
        ("sample_spread", "spread"),
    ):
        values = [value.get(key) for value in history]
        valid = [(step, value) for step, value in zip(steps, values) if value is not None]
        if valid:
            ax_history.plot([item[0] for item in valid], [item[1] for item in valid], marker="o", label=label)
    ax_history.set(title="E  Probability metrics by optimizer step", xlabel="optimizer step", ylabel="normalized distance")
    ax_history.legend(fontsize=8)

    bar_values = [
        float(current_metrics.get("coverage_90", float("nan"))),
        float(current_metrics.get("contact_nll", float("nan"))),
        float(current_metrics.get("contact_brier", float("nan"))),
        float(current_metrics.get("contact_macro_f1", float("nan"))),
    ]
    bar_labels = ["Coverage 90", "NLL", "Brier", "Macro-F1"]
    bars = ax_calibration.bar(bar_labels, bar_values, color=("tab:green", "tab:orange", "tab:red", "tab:blue"))
    for bar, value in zip(bars, bar_values):
        if np.isfinite(value):
            ax_calibration.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax_calibration.set(title="F  Coverage and contact calibration", ylabel="metric")
    ax_calibration.tick_params(axis="x", labelrotation=20)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"CARS-WM checkpoint step {int(current_metrics['step'])} | "
        f"K={int(current_metrics['num_samples'])} | seed={int(current_metrics['visualization_seed'])}"
    )
    figure.tight_layout()
    figure.savefig(Path(path), dpi=150)
    plt.close(figure)


__all__ = ["render_checkpoint_summary"]
