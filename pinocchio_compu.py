# pinocchio计算动力学线性项，映射成末端等效 wrench，并按 episode 画对比曲线。

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ATI wrench with Pinocchio equivalent endpoint wrench.")
    parser.add_argument("--config", type=Path, default=Path("config/pinocchio.yaml"))
    parser.add_argument("--urdf", type=Path, default=Path("sim_mesh/franka_fr3/fr3_franka_hand.urdf"))
    parser.add_argument("--frame-name", default="fr3_hand")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pinocchio_check"))
    return parser.parse_args()


def to_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def plot_episode(ati_list, tau_eq_list, diff_list, save_path):
    ati = np.asarray(ati_list)
    tau_eq = np.asarray(tau_eq_list)
    diff = np.asarray(diff_list)

    names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    for i, name in enumerate(names):
        axes[i].plot(ati[:, i], label="ATI wrench", linewidth=1.2)
        axes[i].plot(tau_eq[:, i], label="Pinocchio external eq wrench", linewidth=1.2)
        axes[i].plot(diff[:, i], label="ATI - Pinocchio", linewidth=0.9, alpha=0.75)
        axes[i].set_ylabel(name)
        axes[i].grid(True)
        axes[i].legend(loc="upper right")

    axes[-1].set_xlabel("frame in episode")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    full_model = pin.buildModelFromUrdf(str(args.urdf))
    locked_joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    locked_joint_ids = [full_model.getJointId(name) for name in locked_joint_names]
    model = pin.buildReducedModel(
        full_model,
        locked_joint_ids,
        pin.neutral(full_model),
    )
    data = model.createData()

    frame_id = model.getFrameId(args.frame_name)
    if frame_id == len(model.frames):
        raise ValueError(f"frame not found: {args.frame_name}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = PINNDataset(config)
    raw_to_sample_idx = {
        raw_idx: sample_idx
        for sample_idx, raw_idx in enumerate(dataset.valid_indices)
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(dataset.dataset.meta.episodes)
    log.info(f"num episodes: {len(episodes)}")

    for ep_i, ep in enumerate(episodes):
        episode_index = int(ep.get("episode_index", ep_i))
        start_raw = int(ep["dataset_from_index"])
        end_raw = int(ep["dataset_to_index"])

        ati_list = []
        tau_eq_list = []
        diff_list = []

        for raw_idx in range(start_raw, end_raw):
            sample_idx = raw_to_sample_idx.get(raw_idx)
            if sample_idx is None:
                continue

            sample = dataset[sample_idx]

            q = to_numpy_1d(sample["q"][0])
            v = to_numpy_1d(sample["v"][0])
            a = to_numpy_1d(sample["a"][0])
            tau = to_numpy_1d(sample["tau"][0])
            wrench = to_numpy_1d(sample["wrench"][0])

            tau_id = pin.rnea(model, data, q, v, a)

            pin.computeJointJacobians(model, data, q)
            pin.framesForwardKinematics(model, data, q)

            J = pin.getFrameJacobian(
                model,
                data,
                frame_id,
                pin.ReferenceFrame.LOCAL,
            )

            # tau_g = pin.rnea(
            #     model,
            #     data,
            #     q,
            #     np.zeros(model.nv),
            #     np.zeros(model.nv),
            # )

            tau_ext = tau_id - tau
            tau_eq_wrench = np.linalg.lstsq(J.T, tau_ext, rcond=None)[0]
            diff = wrench - tau_eq_wrench

            ati_list.append(wrench)
            tau_eq_list.append(tau_eq_wrench)
            diff_list.append(diff)

        if not ati_list:
            log.warning(f"episode {episode_index} has no samples, skip")
            continue

        save_path = output_dir / f"episode_{episode_index:03d}_wrench_compare.png"
        plot_episode(ati_list, tau_eq_list, diff_list, save_path)
        log.info(f"saved episode {episode_index} plot: {save_path}")


if __name__ == "__main__":
    main()
