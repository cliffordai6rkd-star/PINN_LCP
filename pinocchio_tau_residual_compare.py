import argparse
import copy
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml

from data_process.dataloader import PINNDataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


JOINT_NAMES = [f"tau{i + 1}" for i in range(7)]
WRENCH_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
REFERENCE_FRAMES = {
    "LOCAL": pin.ReferenceFrame.LOCAL,
    "WORLD": pin.ReferenceFrame.WORLD,
    "LOCAL_WORLD_ALIGNED": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare tau_follower + Franka tau_ext against Pinocchio tau_id, "
            "and compare ATI wrench against Franka tau_ext mapped to endpoint wrench."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config/pinocchio.yaml"))
    parser.add_argument("--urdf", type=Path, default=Path("sim_mesh/franka_fr3/fr3_franka_hand.urdf"))
    parser.add_argument("--frame-name", default="fr3_hand")
    parser.add_argument(
        "--reference-frame",
        choices=sorted(REFERENCE_FRAMES.keys()),
        default="WORLD",
        help="Frame used by Pinocchio getFrameJacobian(). WORLD matches libfranka zeroJacobian style.",
    )
    parser.add_argument("--tau-ext-feature", default="observation.tau_ext")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pinocchio_tau_residual_compare"))
    parser.add_argument("--window-index", type=int, default=0, help="Index inside dataset horizon window.")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def to_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def load_yaml(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_config_with_tau_ext(config, tau_ext_feature):
    config = copy.deepcopy(config)
    dataloader_cfg = config.setdefault("dataloader", {})
    lowdim_keys = dict(dataloader_cfg.get("lowdim_keys") or {})

    required = {"q", "v", "a", "tau", "wrench"}
    missing = sorted(required - set(lowdim_keys.keys()))
    if missing:
        raise ValueError(f"config.dataloader.lowdim_keys missing required keys: {missing}")

    lowdim_keys.setdefault("tau_ext", tau_ext_feature)
    dataloader_cfg["lowdim_keys"] = lowdim_keys
    dataloader_cfg["load_images"] = False

    if dataloader_cfg.get("normalize_mode") is not None:
        log.warning("forcing normalize_mode=null because torque/wrench physics comparison needs raw values")
        dataloader_cfg["normalize_mode"] = None

    return config


def build_model(urdf_path, frame_name):
    full_model = pin.buildModelFromUrdf(str(urdf_path))
    locked_joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    locked_joint_ids = [full_model.getJointId(name) for name in locked_joint_names]
    model = pin.buildReducedModel(full_model, locked_joint_ids, pin.neutral(full_model))
    data = model.createData()

    frame_id = model.getFrameId(frame_name)
    if frame_id == len(model.frames):
        raise ValueError(f"frame not found: {frame_name}")
    return model, data, frame_id


def solve_endpoint_wrench(jacobian, joint_tau):
    return np.linalg.lstsq(jacobian.T, joint_tau, rcond=None)[0]


def plot_series(series_by_name, dim_names, title, save_path):
    fig, axes = plt.subplots(len(dim_names), 1, figsize=(14, 2.6 * len(dim_names)), sharex=True)
    if len(dim_names) == 1:
        axes = [axes]

    for dim, dim_name in enumerate(dim_names):
        ax = axes[dim]
        for label, values in series_by_name.items():
            ax.plot(values[:, dim], label=label, linewidth=1.1)
        ax.set_ylabel(dim_name)
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[0].set_title(title)
    axes[-1].set_xlabel("frame in episode")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_dimension_files(series_by_name, dim_names, output_dir, prefix):
    for dim, dim_name in enumerate(dim_names):
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        for label, values in series_by_name.items():
            ax.plot(values[:, dim], label=label, linewidth=1.2)
        ax.set_title(dim_name)
        ax.set_xlabel("frame in episode")
        ax.set_ylabel(dim_name)
        ax.grid(True)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{dim_name}.png", dpi=150)
        plt.close(fig)


def stack(values):
    return np.asarray(values, dtype=np.float64)


def main():
    args = parse_args()
    model, data, frame_id = build_model(args.urdf, args.frame_name)
    reference_frame = REFERENCE_FRAMES[args.reference_frame]

    config = load_yaml(args.config)
    dataset = PINNDataset(dataset_config_with_tau_ext(config, args.tau_ext_feature))
    raw_to_sample_idx = {
        raw_idx: sample_idx
        for sample_idx, raw_idx in enumerate(dataset.valid_indices)
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(dataset.dataset.meta.episodes)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    log.info(f"num episodes: {len(episodes)}")

    for ep_i, ep in enumerate(episodes):
        episode_index = int(ep.get("episode_index", ep_i))
        start_raw = int(ep["dataset_from_index"])
        end_raw = int(ep["dataset_to_index"])

        tau_id_rows = []
        tau_follow_rows = []
        tau_ext_rows = []
        tau_follow_plus_ext_rows = []
        measured_wrench_rows = []
        tau_ext_endpoint_wrench_rows = []

        for raw_idx in range(start_raw, end_raw):
            sample_idx = raw_to_sample_idx.get(raw_idx)
            if sample_idx is None:
                continue

            sample = dataset[sample_idx]
            window_index = args.window_index

            q = to_numpy_1d(sample["q"][window_index])
            v = to_numpy_1d(sample["v"][window_index])
            a = to_numpy_1d(sample["a"][window_index])
            tau_follow = to_numpy_1d(sample["tau"][window_index])
            tau_ext = to_numpy_1d(sample["tau_ext"][window_index])
            measured_wrench = to_numpy_1d(sample["wrench"][window_index])

            tau_id = pin.rnea(model, data, q, v, a)

            pin.computeJointJacobians(model, data, q)
            pin.framesForwardKinematics(model, data, q)
            jacobian = pin.getFrameJacobian(model, data, frame_id, reference_frame)

            tau_id_rows.append(tau_id)
            tau_follow_rows.append(tau_follow)
            tau_ext_rows.append(tau_ext)
            tau_follow_plus_ext_rows.append(tau_follow + tau_ext)
            measured_wrench_rows.append(measured_wrench)
            tau_ext_endpoint_wrench_rows.append(solve_endpoint_wrench(jacobian, tau_ext))

        if not tau_id_rows:
            log.warning(f"episode {episode_index} has no samples, skip")
            continue

        episode_dir = output_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        tau_id = stack(tau_id_rows)
        tau_follow = stack(tau_follow_rows)
        tau_ext = stack(tau_ext_rows)
        tau_follow_plus_ext = stack(tau_follow_plus_ext_rows)
        measured_wrench = stack(measured_wrench_rows)
        tau_ext_endpoint_wrench = stack(tau_ext_endpoint_wrench_rows)

        joint_series = {
            "pinocchio tau_id": tau_id,
            "tau_follower": tau_follow,
            "franka tau_ext": tau_ext,
            "tau_follower + tau_ext": tau_follow_plus_ext,
        }
        wrench_series = {
            "measured wrench": measured_wrench,
            "franka tau_ext -> endpoint wrench": tau_ext_endpoint_wrench,
        }

        plot_series(
            joint_series,
            JOINT_NAMES,
            "Joint torque comparison: tau_follower + tau_ext vs tau_id",
            episode_dir / "joint_tau_follow_plus_ext_vs_tau_id.png",
        )
        plot_dimension_files(joint_series, JOINT_NAMES, episode_dir, "joint_tau")
        plot_series(
            wrench_series,
            WRENCH_NAMES,
            f"ATI wrench vs tau_ext endpoint wrench ({args.reference_frame})",
            episode_dir / "ati_wrench_vs_tau_ext_endpoint_wrench.png",
        )
        plot_dimension_files(wrench_series, WRENCH_NAMES, episode_dir, "ati_vs_tau_ext_wrench")
        log.info(f"saved episode {episode_index} comparison: {episode_dir}")


if __name__ == "__main__":
    main()
