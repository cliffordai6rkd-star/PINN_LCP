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


def plot_episode_variable(values, save_path, name="value", dim_names=None):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)

    dim = values.shape[1]
    if dim_names is None:
        dim_names = [f"{name}_{i}" for i in range(dim)]
    if len(dim_names) != dim:
        raise ValueError(f"dim_names length {len(dim_names)} does not match data dim {dim}")

    fig, axes = plt.subplots(dim, 1, figsize=(14, max(3, 2.6 * dim)), sharex=True)
    if dim == 1:
        axes = [axes]

    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        max_abs = float(np.max(np.abs(finite_values)))
    else:
        max_abs = 1.0
    if max_abs < 1e-12:
        max_abs = 1.0
    y_limit = max_abs * 1.05

    for i, dim_name in enumerate(dim_names):
        axes[i].plot(values[:, i], label=dim_name, linewidth=1.2)
        axes[i].set_ylim(-y_limit, y_limit)
        axes[i].set_ylabel(dim_name)
        axes[i].grid(True)
        axes[i].legend(loc="upper right")

    axes[0].set_title(name)
    axes[-1].set_xlabel("frame in episode")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_episode(ati_list, tau_eq_list, diff_list, save_path):
    names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    ati = np.asarray(ati_list)
    tau_eq = np.asarray(tau_eq_list)
    diff = np.asarray(diff_list)

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)

    for i, dim_name in enumerate(names):
        axes[i].plot(ati[:, i], label="ATI wrench", linewidth=1.2)
        axes[i].plot(tau_eq[:, i], label="Pinocchio external eq wrench", linewidth=1.2)
        axes[i].plot(diff[:, i], label="ATI - Pinocchio", linewidth=0.9, alpha=0.75)
        axes[i].set_ylabel(dim_name)
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

        tau_ext_m_list  = []
        tau_contact_c_list  = []
        tau_id_list = []
        ati_list = []
        tau_list =[] 
        a_list = []
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
            tau_ext = to_numpy_1d(sample["tau_ext"][0])
            wrench = to_numpy_1d(sample["wrench"][0])

            tau_id = pin.rnea(model, data, q, v, a)

            pin.computeJointJacobians(model, data, q)
            pin.framesForwardKinematics(model, data, q)

            J = pin.getFrameJacobian(
                model,
                data,
                frame_id,
                pin.ReferenceFrame.WORLD,
            )

            # tau_g = pin.rnea(
            #     model,
            #     data,
            #     q,
            #     np.zeros(model.nv),
            #     np.zeros(model.nv),
            # )

            tau_ext_m = tau_ext
            tau_contact_c = - tau_id + tau 
            # tau_ati = J.T @ wrench

            # ati = wrench
            diff = np.linalg.lstsq(J.T, tau_contact_c, rcond=None)[0]
            # tau_ext_c = np.linalg.lstsq(J.T, tau_ext_c, rcond=None)[0]

            

            tau_ext_m_list.append(tau_ext_m)
            tau_contact_c_list.append(tau_contact_c)
            tau_id_list.append(tau_id)
            ati_list.append(wrench)
            tau_list.append(tau)
            a_list.append(a)
            diff_list.append(diff)

    

        episode_dir = output_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        tau_dim_names = [f"tau{i + 1}" for i in range(7)]
        wrench_dim_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
        plot_episode_variable(
            tau_ext_m_list,
            episode_dir / "tau_ext_m.png",
            name="tau_ext_m",
            dim_names=tau_dim_names,
        )
        plot_episode_variable(
            tau_contact_c_list,
            episode_dir / "tau_contact_c.png",
            name="tau_contact_c",
            dim_names=tau_dim_names,
        )
        plot_episode_variable(
            a_list,
            episode_dir / "a_.png",
            name="a",
            dim_names=tau_dim_names,
        )
        plot_episode_variable(
            tau_id_list,
            episode_dir / "tau_id.png",
            name="tau_id",
            dim_names=tau_dim_names,
        )
        plot_episode_variable(
            ati_list,
            episode_dir / "ati.png",
            name="ati",
            dim_names=wrench_dim_names,
        )
        # plot_episode_variable(
        #     tau_list,
        #     episode_dir / "tau_.png",
        #     name="tau",
        #     dim_names=wrench_dim_names,
        # )
        plot_episode_variable(
            diff_list,
            episode_dir / "diff.png",
            name="diff",
            dim_names=wrench_dim_names,
        )
        log.info(f"saved episode {episode_index} plots: {episode_dir}")


if __name__ == "__main__":
    main()
