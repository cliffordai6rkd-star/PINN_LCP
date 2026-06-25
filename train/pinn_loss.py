from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class PinnLossCalculator:
    def __init__(self, config):
        self.config = config
        self.loss_config = config.get("loss") or {}

        self.weights = {
            "data": float(self.loss_config.get("data_weight", 1.0)),
            "dynamics": float(self.loss_config.get("dynamics_weight", 0.0)),
            "friction_cone": float(self.loss_config.get("friction_cone_weight", 0.0)),
            "normal_complementarity": float(
                self.loss_config.get("normal_complementarity_weight", 0.0)
            ),
            "tangent_complementarity": float(
                self.loss_config.get("tangent_complementarity_weight", 0.0)
            ),
        }
        self.mu = float(self.loss_config.get("friction_mu", 0.5))
        self.eps = float(self.loss_config.get("eps", 1e-8))
        self.pin_model = None
        self.pin_data = None
        self.pin_frame_id = None
        self.pin_reference_frame = None

    def __call__(self, out, batch):
        wrench_pred = out["wrench_pred"]

        loss_dict = {}
        total_loss = wrench_pred.new_zeros(())

        data_loss = self.data_supervision_loss(wrench_pred, batch)
        total_loss = total_loss + self.weights["data"] * data_loss
        loss_dict["data_loss"] = data_loss.detach()

        if self.weights["dynamics"] > 0.0:
            dynamics_loss = self.dynamics_loss(wrench_pred, batch)
        else:
            dynamics_loss = wrench_pred.new_zeros(())
        total_loss = total_loss + self.weights["dynamics"] * dynamics_loss
        loss_dict["dynamics_loss"] = dynamics_loss.detach()

        friction_cone_loss = self.friction_cone_loss(wrench_pred, batch)
        total_loss = total_loss + self.weights["friction_cone"] * friction_cone_loss
        loss_dict["friction_cone_loss"] = friction_cone_loss.detach()

        normal_loss = self.normal_complementarity_loss(wrench_pred, batch)
        total_loss = total_loss + self.weights["normal_complementarity"] * normal_loss
        loss_dict["normal_complementarity_loss"] = normal_loss.detach()

        tangent_loss = self.tangent_complementarity_loss(wrench_pred, batch)
        total_loss = total_loss + self.weights["tangent_complementarity"] * tangent_loss
        loss_dict["tangent_complementarity_loss"] = tangent_loss.detach()
        loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict

    def data_supervision_loss(self, wrench_pred, batch):
        wrench_target = self.get_wrench_target(batch, wrench_pred)
        wrench_pred, wrench_target = self.align_prediction_and_target(
            wrench_pred,
            wrench_target,
        )
        return F.mse_loss(wrench_pred, wrench_target)

    def dynamics_loss(self, wrench_pred, batch):
        q = self.get_future_dynamics_tensor(batch, "q")
        v = self.get_future_dynamics_tensor(batch, "v")
        a = self.get_future_dynamics_tensor(batch, "a")
        tau = self.get_future_dynamics_tensor(batch, "tau")

        q, v, a, tau = self.align_dynamics_inputs(q, v, a, tau)
        self.check_same_time_length(
            wrench_pred=wrench_pred,
            q=q,
            v=v,
            a=a,
            tau=tau,
        )

        wrench_target = self.pinocchio_wrench_from_batch(
            q=q,
            v=v,
            a=a,
            tau=tau,
            device=wrench_pred.device,
            dtype=wrench_pred.dtype,
        )
        return F.mse_loss(wrench_pred, wrench_target)

    def get_future_dynamics_tensor(self, batch, key):
        candidates = (f"{key}_future_raw", f"{key}_future")
        for candidate in candidates:
            if candidate in batch:
                return batch[candidate]
        raise KeyError(
            "dynamics loss requires future state tensors from dataloader; "
            f"missing one of {candidates}"
        )

    def pinocchio_wrench_from_batch(self, q, v, a, tau, device, dtype):
        self.setup_pinocchio()

        q_np = q.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()
        a_np = a.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()

        batch_size, time_steps = q_np.shape[:2]
        wrench = np.zeros((batch_size, time_steps, 6), dtype=np.float64)

        for batch_idx in range(batch_size):
            for time_idx in range(time_steps):
                wrench[batch_idx, time_idx] = self.pinocchio_wrench_one_step(
                    q_np[batch_idx, time_idx],
                    v_np[batch_idx, time_idx],
                    a_np[batch_idx, time_idx],
                    tau_np[batch_idx, time_idx],
                )

        return torch.as_tensor(wrench, device=device, dtype=dtype)

    def pinocchio_wrench_one_step(self, q, v, a, tau):
        import pinocchio as pin

        tau_id = pin.rnea(self.pin_model, self.pin_data, q, v, a)

        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        jacobian = pin.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.pin_frame_id,
            self.pin_reference_frame,
        )

        tau_source = self.loss_config.get("pinocchio_tau_source", "tau_ext")
        if tau_source == "tau_ext":
            generalized_force = tau_id - tau
        elif tau_source == "tau":
            generalized_force = tau
        elif tau_source == "tau_minus_tau_id":
            generalized_force = tau - tau_id
        else:
            raise ValueError(f"unknown pinocchio_tau_source: {tau_source}")

        # 用最小二乘求解 J(q)^T * wrench = generalized_force。
        wrench, *_ = np.linalg.lstsq(jacobian.T, generalized_force, rcond=None)
        wrench_sign = float(self.loss_config.get("pinocchio_wrench_sign", 1.0))
        return wrench_sign * wrench

    def setup_pinocchio(self):
        if self.pin_model is not None:
            return

        import pinocchio as pin

        urdf_path = Path(
            self.loss_config.get(
                "pinocchio_urdf_path",
                "sim_mesh/franka_fr3/fr3_franka_hand.urdf",
            )
        )
        full_model = pin.buildModelFromUrdf(str(urdf_path))

        locked_joint_names = self.loss_config.get(
            "pinocchio_locked_joint_names",
            ["fr3_finger_joint1", "fr3_finger_joint2"],
        )
        locked_joint_ids = []
        for joint_name in locked_joint_names:
            joint_id = full_model.getJointId(joint_name)
            if joint_id == full_model.njoints:
                raise ValueError(f"pinocchio joint not found: {joint_name}")
            locked_joint_ids.append(joint_id)

        if locked_joint_ids:
            self.pin_model = pin.buildReducedModel(
                full_model,
                locked_joint_ids,
                pin.neutral(full_model),
            )
        else:
            self.pin_model = full_model
        self.pin_data = self.pin_model.createData()

        frame_name = self.loss_config.get("pinocchio_frame_name", "fr3_hand")
        self.pin_frame_id = self.pin_model.getFrameId(frame_name)
        if self.pin_frame_id == len(self.pin_model.frames):
            raise ValueError(f"pinocchio frame not found: {frame_name}")

        reference_frame = self.loss_config.get("pinocchio_reference_frame", "LOCAL")
        reference_frames = {
            "LOCAL": pin.ReferenceFrame.LOCAL,
            "WORLD": pin.ReferenceFrame.WORLD,
            "LOCAL_WORLD_ALIGNED": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        }
        if reference_frame not in reference_frames:
            raise ValueError(f"unknown pinocchio_reference_frame: {reference_frame}")
        self.pin_reference_frame = reference_frames[reference_frame]

    def friction_cone_loss(self, wrench_pred, batch):
        if "contact_normal" not in batch:
            return wrench_pred.new_zeros(())

        force = wrench_pred[..., :3]
        normal = batch["contact_normal"]
        force, normal = self.align_time(force, normal)
        normal = F.normalize(normal, dim=-1, eps=self.eps)

        normal_force = (force * normal).sum(dim=-1, keepdim=True)
        tangent_force = force - normal_force * normal

        # 摩擦锥: ||F_t|| <= mu * F_n，同时要求法向力非负。
        cone_violation = tangent_force.norm(dim=-1, keepdim=True) - self.mu * normal_force
        normal_negative = F.relu(-normal_force)
        return F.relu(cone_violation).pow(2).mean() + normal_negative.pow(2).mean()

    def normal_complementarity_loss(self, wrench_pred, batch):
        if "contact_normal" not in batch or "contact_phi" not in batch:
            return wrench_pred.new_zeros(())

        force = wrench_pred[..., :3]
        normal = batch["contact_normal"]
        phi = self.ensure_scalar_time_series(batch["contact_phi"])

        force, normal = self.align_time(force, normal)
        force, phi = self.align_time(force, phi)
        normal = F.normalize(normal, dim=-1, eps=self.eps)
        phi = self.ensure_last_dim(phi)

        normal_force = (force * normal).sum(dim=-1, keepdim=True)

        # 法向互补: 0 <= F_n ⟂ phi(q) >= 0。
        force_positive_loss = F.relu(-normal_force).pow(2).mean()
        gap_positive_loss = F.relu(-phi).pow(2).mean()
        complementarity_loss = (normal_force * phi).pow(2).mean()
        return force_positive_loss + gap_positive_loss + complementarity_loss

    def tangent_complementarity_loss(self, wrench_pred, batch):
        if "contact_normal" not in batch or "contact_tangent_velocity" not in batch:
            return wrench_pred.new_zeros(())

        force = wrench_pred[..., :3]
        normal = batch["contact_normal"]
        tangent_velocity = batch["contact_tangent_velocity"]

        force, normal = self.align_time(force, normal)
        force, tangent_velocity = self.align_time(force, tangent_velocity)
        normal = F.normalize(normal, dim=-1, eps=self.eps)

        normal_force = (force * normal).sum(dim=-1, keepdim=True)
        tangent_force = force - normal_force * normal
        tangent_force_norm = tangent_force.norm(dim=-1, keepdim=True)
        tangent_velocity_norm = tangent_velocity.norm(dim=-1, keepdim=True)

        # 当前 model_v1 只输出 wrench，没有单独输出 Stewart-Trinkle 的 gamma。
        # 这里先用摩擦锥余量作为 gamma 的 surrogate，后续若模型输出 gamma 可直接替换。
        gamma = F.relu(self.mu * normal_force - tangent_force_norm)
        return (gamma * tangent_velocity_norm).pow(2).mean()

    def get_wrench_target(self, batch, wrench_pred):
        target_key = self.loss_config.get("wrench_target_key")
        if target_key is not None and target_key in batch:
            return batch[target_key]

        for key in ("wrench_target", "wrench_future", "wrench"):
            if key in batch:
                return batch[key]

        raise KeyError("batch must contain wrench_target, wrench_future, or wrench")

    def align_prediction_and_target(self, pred, target):
        target = self.ensure_time_dim(target)

        if pred.shape[1] == target.shape[1]:
            return pred, target

        if target.shape[1] == 1:
            return pred, target.expand(-1, pred.shape[1], -1)

        # dataloader 暂时只有历史窗口时，用最后一帧 wrench 监督未来预测，先保证训练链路可跑。
        target = target[:, -1:, :].expand(-1, pred.shape[1], -1)
        return pred, target

    def align_dynamics_inputs(self, q, v, a, tau):
        q = self.ensure_time_dim(q)
        v = self.ensure_time_dim(v)
        a = self.ensure_time_dim(a)
        tau = self.ensure_time_dim(tau)

        time_lengths = [q.shape[1], v.shape[1], a.shape[1], tau.shape[1]]
        strict = bool(self.loss_config.get("strict_dynamics_time_align", True))
        if strict and len(set(time_lengths)) != 1:
            raise ValueError(
                "q/v/a/tau must share the same time length for dynamics loss, "
                f"got {time_lengths}"
            )

        min_len = min(time_lengths)
        return (
            q[:, :min_len],
            v[:, :min_len],
            a[:, :min_len],
            tau[:, :min_len],
        )

    def check_same_time_length(self, wrench_pred, q, v, a, tau):
        time_lengths = {
            "wrench_pred": wrench_pred.shape[1],
            "q_future": q.shape[1],
            "v_future": v.shape[1],
            "a_future": a.shape[1],
            "tau_future": tau.shape[1],
        }
        if len(set(time_lengths.values())) != 1:
            raise ValueError(
                "dynamics loss requires future q/v/a/tau and wrench_pred "
                f"to share the same time length, got {time_lengths}"
            )

    def align_time(self, reference, value):
        value = self.ensure_time_dim(value)
        if reference.shape[1] == value.shape[1]:
            return reference, value

        if value.shape[1] == 1:
            return reference, value.expand(-1, reference.shape[1], *value.shape[2:])

        min_len = min(reference.shape[1], value.shape[1])
        return reference[:, :min_len], value[:, :min_len]

    def ensure_time_dim(self, x):
        if x.ndim == 2:
            return x.unsqueeze(1)
        return x

    def ensure_last_dim(self, x):
        if x.ndim == 2:
            return x.unsqueeze(-1)
        return x

    def ensure_scalar_time_series(self, x):
        if x.ndim == 2:
            return x.unsqueeze(-1)
        return x

    def has_keys(self, batch, keys):
        return all(key in batch for key in keys)
