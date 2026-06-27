import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin


log = logging.getLogger(__name__)


@dataclass
class TorqueQPCommand:
    tau: np.ndarray
    qddot: np.ndarray
    wrench: np.ndarray
    pose_error: np.ndarray
    success: bool
    status: str
    cost: float


class ContactInverseDynamicsQPController:
    """Torque-level contact inverse dynamics QP for the FR3 arm.

    The QP solves for joint acceleration, actuator torque, and a 6D endpoint
    wrench while enforcing inverse dynamics as an equality constraint:

        M(q) qddot + h(q, v) = tau + J(q)^T wrench

    The current implementation can use SciPy SLSQP for box and linear contact
    constraints. If SciPy is unavailable, it falls back to an equality
    constrained dense QP and clips the final torque command.
    """

    def __init__(self, config):
        self.config = config
        urdf_path = config.get(
            "qp_urdf_path",
            config.get("ik_urdf_path", "sim_mesh/franka_fr3/fr3_pika_gripper_ati.urdf"),
        )
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.ee_frame_name = config.get(
            "qp_ee_frame_name",
            config.get("ik_ee_frame_name", "pika_gripper_ee"),
        )
        self.arm_joint_names = config.get(
            "qp_arm_joint_names",
            config.get(
                "ik_arm_joint_names",
                [
                    "fr3_joint1",
                    "fr3_joint2",
                    "fr3_joint3",
                    "fr3_joint4",
                    "fr3_joint5",
                    "fr3_joint6",
                    "fr3_joint7",
                ],
            ),
        )
        default_locked_joints = config.get(
            "ik_gripper_joint_names",
            ["gripper_left_joint", "gripper_right_joint"],
        )
        self.lock_joint_names = config.get("qp_lock_joint_names", default_locked_joints)
        self.reference_frame_name = str(config.get("qp_reference_frame", "LOCAL")).upper()
        self.reference_frame = self._parse_reference_frame(self.reference_frame_name)

        self.dt = float(config.get("qp_dt", config.get("control_dt", 0.01)))
        if self.dt <= 0.0:
            raise ValueError(f"qp_dt/control_dt must be positive, got {self.dt}")

        self.task_kp = self._vector_config(
            "qp_task_kp",
            [120.0, 120.0, 120.0, 50.0, 50.0, 50.0],
            6,
        )
        self.task_kd = self._vector_config(
            "qp_task_kd",
            [24.0, 24.0, 24.0, 10.0, 10.0, 10.0],
            6,
        )
        self.motion_weight = self._vector_config(
            "qp_motion_weight",
            [1.0, 1.0, 1.0, 0.4, 0.4, 0.4],
            6,
        )
        self.force_weight = self._vector_config(
            "qp_force_weight",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            6,
        )
        self.qddot_weight = float(config.get("qp_qddot_weight", 1.0e-4))
        self.torque_weight = float(config.get("qp_torque_weight", 1.0e-5))
        self.wrench_weight = float(config.get("qp_wrench_regularization", 1.0e-6))
        self.lock_wrench_without_target = bool(config.get("qp_lock_wrench_without_target", True))
        self.posture_weight = float(config.get("qp_posture_weight", 1.0e-3))
        self.posture_kp = float(config.get("qp_posture_kp", 8.0))
        self.posture_kd = float(config.get("qp_posture_kd", 2.0))
        self.regularization = float(config.get("qp_regularization", 1.0e-9))
        self.wrench_sign = float(config.get("qp_wrench_sign", 1.0))
        self.max_slsqp_iter = int(config.get("qp_max_slsqp_iter", 80))
        self.slsqp_ftol = float(config.get("qp_slsqp_ftol", 1.0e-6))

        self.torque_limit = self._vector_config(
            "qp_torque_limit",
            [40.0, 40.0, 40.0, 40.0, 12.0, 12.0, 12.0],
            len(self.arm_joint_names),
        )
        self.qddot_limit = self._vector_config(
            "qp_qddot_limit",
            [30.0] * len(self.arm_joint_names),
            len(self.arm_joint_names),
        )
        self.qvel_limit = self._optional_vector_config("qp_qvel_limit", len(self.arm_joint_names))

        self.contact_normal = self._optional_vector_config("qp_contact_normal", 3)
        if self.contact_normal is not None:
            norm = np.linalg.norm(self.contact_normal)
            if norm < 1.0e-8:
                raise ValueError("qp_contact_normal must be non-zero")
            self.contact_normal = self.contact_normal / norm
        self.friction_mu = float(config.get("qp_friction_mu", 0.0))
        self.min_normal_force = config.get("qp_min_normal_force", None)
        self.max_normal_force = config.get("qp_max_normal_force", None)

        self.model = self._build_model()
        self._configure_gravity()
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.frame_id == len(self.model.frames):
            raise ValueError(f"QP frame not found in URDF: {self.ee_frame_name}")

        self.arm_q_indices = self._joint_q_indices(self.arm_joint_names)
        self.nv = self.model.nv
        if self.nv != len(self.arm_joint_names):
            raise ValueError(
                f"QP reduced model nv={self.nv}, expected {len(self.arm_joint_names)}. "
                "Check qp_lock_joint_names and qp_arm_joint_names."
            )

        self.lower_limits = np.asarray(self.model.lowerPositionLimit, dtype=np.float64)
        self.upper_limits = np.asarray(self.model.upperPositionLimit, dtype=np.float64)
        self.q_nominal = self._make_nominal_q()

        log.info(
            "loaded torque QP model from %s, frame=%s, nq=%d, nv=%d, locked_joints=%s",
            self.urdf_path,
            self.ee_frame_name,
            self.model.nq,
            self.model.nv,
            self.lock_joint_names,
        )

    def _build_model(self):
        full_model = pin.buildModelFromUrdf(str(self.urdf_path))
        locked_joint_ids = [
            full_model.getJointId(name)
            for name in self.lock_joint_names
            if full_model.existJointName(name)
        ]
        if not locked_joint_ids:
            return full_model

        locked_reference = pin.neutral(full_model)
        return pin.buildReducedModel(full_model, locked_joint_ids, locked_reference)

    def _configure_gravity(self):
        gravity = self.config.get("qp_gravity", None)
        if gravity is None:
            return
        gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
        self.model.gravity = pin.Motion(np.concatenate([gravity, np.zeros(3, dtype=np.float64)]))

    def _joint_q_indices(self, joint_names):
        indices = []
        for joint_name in joint_names:
            if not self.model.existJointName(joint_name):
                raise ValueError(f"joint not found in QP model: {joint_name}")
            joint_id = self.model.getJointId(joint_name)
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"expected 1-DoF joint for {joint_name}, got nq={joint.nq}, nv={joint.nv}"
                )
            indices.append(joint.idx_q)
        return indices

    def _make_nominal_q(self):
        q_nominal = np.asarray(pin.neutral(self.model), dtype=np.float64).reshape(-1)
        nominal_arm_q = self.config.get("qp_nominal_arm_q", self.config.get("q_reset", None))
        if nominal_arm_q is not None:
            nominal_arm_q = np.asarray(nominal_arm_q, dtype=np.float64).reshape(-1)
            if nominal_arm_q.shape[0] != len(self.arm_q_indices):
                raise ValueError(
                    f"qp_nominal_arm_q/q_reset length {nominal_arm_q.shape[0]} "
                    f"!= expected {len(self.arm_q_indices)}"
                )
            for value, idx in zip(nominal_arm_q, self.arm_q_indices):
                q_nominal[idx] = value
        return self.clip_q(q_nominal)

    @staticmethod
    def _parse_reference_frame(name):
        if name == "LOCAL":
            return pin.ReferenceFrame.LOCAL
        if name == "WORLD":
            return pin.ReferenceFrame.WORLD
        if name == "LOCAL_WORLD_ALIGNED":
            return pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        raise ValueError(f"unknown qp_reference_frame: {name}")

    def _vector_config(self, key, default, size):
        value = np.asarray(self.config.get(key, default), dtype=np.float64).reshape(-1)
        if value.shape[0] != size:
            raise ValueError(f"{key} length {value.shape[0]} != expected {size}")
        return value

    def _optional_vector_config(self, key, size):
        value = self.config.get(key, None)
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if value.shape[0] != size:
            raise ValueError(f"{key} length {value.shape[0]} != expected {size}")
        return value

    def clip_q(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        finite = np.isfinite(self.lower_limits) & np.isfinite(self.upper_limits)
        q[finite] = np.clip(q[finite], self.lower_limits[finite], self.upper_limits[finite])
        return q

    def pose_to_se3(self, ee_pose):
        ee_pose = np.asarray(ee_pose, dtype=np.float64).reshape(7)
        quat = ee_pose[3:7]
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError(f"invalid target quaternion: {quat}")
        quat = quat / norm
        x, y, z, w = quat
        quat_pin = pin.Quaternion(w, x, y, z)
        quat_pin.normalize()
        return pin.SE3(quat_pin.toRotationMatrix(), ee_pose[:3])

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(self.model.nq)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.frame_id].copy()

    def solve(self, q, v, target_pose, desired_wrench=None, desired_twist=None):
        q = self.clip_q(np.asarray(q, dtype=np.float64).reshape(self.model.nq))
        v = np.asarray(v, dtype=np.float64).reshape(self.model.nv)
        target_se3 = target_pose if isinstance(target_pose, pin.SE3) else self.pose_to_se3(target_pose)

        M, h, J, jdot_v, pose_error, current_twist = self._compute_terms(q, v, target_se3)
        desired_acc = self._desired_task_acceleration(
            pose_error=pose_error,
            current_twist=current_twist,
            desired_twist=desired_twist,
        )
        has_desired_wrench = desired_wrench is not None
        desired_wrench = self._desired_wrench(desired_wrench)
        lock_wrench_zero = (
            self.lock_wrench_without_target
            and not has_desired_wrench
            and self.contact_normal is None
            and np.all(self.force_weight <= 0.0)
        )
        qddot_posture = self.posture_kp * (self.q_nominal - q) - self.posture_kd * v

        H, g = self._build_objective(
            J=J,
            jdot_v=jdot_v,
            desired_acc=desired_acc,
            desired_wrench=desired_wrench,
            qddot_posture=qddot_posture,
        )
        E, e = self._build_dynamics_constraint(M=M, h=h, J=J)
        lower, upper = self._build_bounds(q=q, v=v, lock_wrench_zero=lock_wrench_zero)
        linear_constraints = self._build_linear_constraints(E=E, e=e)

        z0 = self._solve_equality_qp(H, g, E, e)
        z0 = np.minimum(np.maximum(z0, lower), upper)
        z, success, status = self._solve_with_scipy(
            H=H,
            g=g,
            z0=z0,
            lower=lower,
            upper=upper,
            constraints=linear_constraints,
        )
        if z is None:
            qddot, _, wrench = self._unpack_z(z0)
            tau = M @ qddot + h - self.wrench_sign * J.T @ wrench
            z = self._pack_z(qddot=qddot, tau=tau, wrench=wrench)
            z = np.minimum(np.maximum(z, lower), upper)
            success = False
            status = "equality_qp_fallback"

        qddot, tau, wrench = self._unpack_z(z)
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        cost = float(0.5 * z @ H @ z + g @ z)
        return TorqueQPCommand(
            tau=tau,
            qddot=qddot,
            wrench=wrench,
            pose_error=pose_error,
            success=success,
            status=status,
            cost=cost,
        )

    def _compute_terms(self, q, v, target_pose):
        pin.computeAllTerms(self.model, self.data, q, v)
        M = np.asarray(self.data.M, dtype=np.float64).copy()
        M = 0.5 * (M + M.T)
        h = np.asarray(pin.nonLinearEffects(self.model, self.data, q, v), dtype=np.float64).reshape(-1)

        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        current_pose = self.data.oMf[self.frame_id].copy()
        pose_error = pin.log(current_pose.actInv(target_pose)).vector
        J = np.asarray(
            pin.getFrameJacobian(
                self.model,
                self.data,
                self.frame_id,
                self.reference_frame,
            ),
            dtype=np.float64,
        )
        current_twist = J @ v
        jdot_v = self._frame_jdot_v(q, v)
        return M, h, J, jdot_v, pose_error, current_twist

    def _frame_jdot_v(self, q, v):
        zero_acc = np.zeros(self.model.nv, dtype=np.float64)
        try:
            pin.forwardKinematics(self.model, self.data, q, v, zero_acc)
            pin.updateFramePlacements(self.model, self.data)
            acc = pin.getFrameClassicalAcceleration(
                self.model,
                self.data,
                self.frame_id,
                self.reference_frame,
            )
            return np.asarray(acc.vector, dtype=np.float64).reshape(6)
        except Exception as exc:
            log.debug("falling back to zero Jdot*v: %s", exc)
            return np.zeros(6, dtype=np.float64)

    def _desired_task_acceleration(self, pose_error, current_twist, desired_twist):
        if desired_twist is None:
            desired_twist = np.zeros(6, dtype=np.float64)
        else:
            desired_twist = np.asarray(desired_twist, dtype=np.float64).reshape(6)
        return self.task_kp * pose_error + self.task_kd * (desired_twist - current_twist)

    @staticmethod
    def _desired_wrench(desired_wrench):
        if desired_wrench is None:
            return np.zeros(6, dtype=np.float64)
        return np.asarray(desired_wrench, dtype=np.float64).reshape(6)

    def _build_objective(self, J, jdot_v, desired_acc, desired_wrench, qddot_posture):
        n = self.nv
        nz = 2 * n + 6
        rows = []
        rhs = []

        motion_scale = np.sqrt(np.maximum(self.motion_weight, 0.0))
        A_motion = np.zeros((6, nz), dtype=np.float64)
        A_motion[:, :n] = J
        rows.append(motion_scale[:, None] * A_motion)
        rhs.append(motion_scale * (desired_acc - jdot_v))

        force_scale = np.sqrt(np.maximum(self.force_weight, 0.0))
        A_force = np.zeros((6, nz), dtype=np.float64)
        A_force[:, 2 * n : 2 * n + 6] = np.eye(6)
        rows.append(force_scale[:, None] * A_force)
        rhs.append(force_scale * desired_wrench)

        if self.posture_weight > 0.0:
            A_posture = np.zeros((n, nz), dtype=np.float64)
            A_posture[:, :n] = np.eye(n)
            scale = np.sqrt(self.posture_weight)
            rows.append(scale * A_posture)
            rhs.append(scale * qddot_posture)

        if self.qddot_weight > 0.0:
            A_qddot = np.zeros((n, nz), dtype=np.float64)
            A_qddot[:, :n] = np.eye(n)
            rows.append(np.sqrt(self.qddot_weight) * A_qddot)
            rhs.append(np.zeros(n, dtype=np.float64))

        if self.torque_weight > 0.0:
            A_tau = np.zeros((n, nz), dtype=np.float64)
            A_tau[:, n : 2 * n] = np.eye(n)
            rows.append(np.sqrt(self.torque_weight) * A_tau)
            rhs.append(np.zeros(n, dtype=np.float64))

        if self.wrench_weight > 0.0:
            A_wrench = np.zeros((6, nz), dtype=np.float64)
            A_wrench[:, 2 * n : 2 * n + 6] = np.eye(6)
            rows.append(np.sqrt(self.wrench_weight) * A_wrench)
            rhs.append(np.zeros(6, dtype=np.float64))

        A = np.vstack(rows)
        b = np.concatenate(rhs)
        H = A.T @ A + self.regularization * np.eye(nz)
        g = -(A.T @ b)
        return H, g

    def _build_dynamics_constraint(self, M, h, J):
        n = self.nv
        nz = 2 * n + 6
        E = np.zeros((n, nz), dtype=np.float64)
        E[:, :n] = M
        E[:, n : 2 * n] = -np.eye(n)
        E[:, 2 * n : 2 * n + 6] = -self.wrench_sign * J.T
        e = -h
        return E, e

    def _build_bounds(self, q, v, lock_wrench_zero=False):
        n = self.nv
        nz = 2 * n + 6
        lower = np.full(nz, -np.inf, dtype=np.float64)
        upper = np.full(nz, np.inf, dtype=np.float64)

        qddot_lower = -self.qddot_limit.copy()
        qddot_upper = self.qddot_limit.copy()

        finite_lower = np.isfinite(self.lower_limits)
        finite_upper = np.isfinite(self.upper_limits)
        qddot_from_q_lower = 2.0 * (self.lower_limits - q - v * self.dt) / (self.dt**2)
        qddot_from_q_upper = 2.0 * (self.upper_limits - q - v * self.dt) / (self.dt**2)
        qddot_lower[finite_lower] = np.maximum(qddot_lower[finite_lower], qddot_from_q_lower[finite_lower])
        qddot_upper[finite_upper] = np.minimum(qddot_upper[finite_upper], qddot_from_q_upper[finite_upper])

        if self.qvel_limit is not None:
            qddot_lower = np.maximum(qddot_lower, (-self.qvel_limit - v) / self.dt)
            qddot_upper = np.minimum(qddot_upper, (self.qvel_limit - v) / self.dt)

        infeasible = qddot_lower > qddot_upper
        if np.any(infeasible):
            qddot_mid = 0.5 * (qddot_lower[infeasible] + qddot_upper[infeasible])
            qddot_lower[infeasible] = qddot_mid
            qddot_upper[infeasible] = qddot_mid

        lower[:n] = qddot_lower
        upper[:n] = qddot_upper
        lower[n : 2 * n] = -self.torque_limit
        upper[n : 2 * n] = self.torque_limit
        if lock_wrench_zero:
            lower[2 * n : 2 * n + 6] = 0.0
            upper[2 * n : 2 * n + 6] = 0.0
        return lower, upper

    def _build_linear_constraints(self, E, e):
        constraints = [{"A": E, "lower": e, "upper": e}]

        if self.contact_normal is None:
            return constraints

        n = self.nv
        nz = 2 * n + 6
        normal = self.contact_normal
        tangent1, tangent2 = self._orthonormal_tangents(normal)

        A_rows = []
        lower = []
        upper = []

        row = np.zeros(nz, dtype=np.float64)
        row[2 * n : 2 * n + 3] = normal
        A_rows.append(row)
        lower.append(-np.inf if self.min_normal_force is None else float(self.min_normal_force))
        upper.append(np.inf if self.max_normal_force is None else float(self.max_normal_force))

        if self.friction_mu > 0.0:
            for tangent in (tangent1, tangent2):
                row_pos = np.zeros(nz, dtype=np.float64)
                row_pos[2 * n : 2 * n + 3] = tangent - self.friction_mu * normal
                A_rows.append(row_pos)
                lower.append(-np.inf)
                upper.append(0.0)

                row_neg = np.zeros(nz, dtype=np.float64)
                row_neg[2 * n : 2 * n + 3] = -tangent - self.friction_mu * normal
                A_rows.append(row_neg)
                lower.append(-np.inf)
                upper.append(0.0)

        constraints.append(
            {
                "A": np.vstack(A_rows),
                "lower": np.asarray(lower, dtype=np.float64),
                "upper": np.asarray(upper, dtype=np.float64),
            }
        )
        return constraints

    @staticmethod
    def _orthonormal_tangents(normal):
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(normal @ basis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent1 = basis - float(basis @ normal) * normal
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        tangent2 /= np.linalg.norm(tangent2)
        return tangent1, tangent2

    def _solve_equality_qp(self, H, g, E, e):
        nz = H.shape[0]
        neq = E.shape[0]
        KKT = np.block(
            [
                [H, E.T],
                [E, np.zeros((neq, neq), dtype=np.float64)],
            ]
        )
        rhs = np.concatenate([-g, e])
        try:
            solution = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        return solution[:nz]

    def _solve_with_scipy(self, H, g, z0, lower, upper, constraints):
        try:
            from scipy.optimize import Bounds, LinearConstraint, minimize
        except Exception as exc:
            log.debug("SciPy QP path unavailable: %s", exc)
            return None, False, "scipy_unavailable"

        scipy_constraints = [
            LinearConstraint(item["A"], item["lower"], item["upper"])
            for item in constraints
        ]
        bounds = Bounds(lower, upper)

        def objective(z):
            return float(0.5 * z @ H @ z + g @ z)

        def gradient(z):
            return H @ z + g

        result = minimize(
            objective,
            z0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=scipy_constraints,
            options={
                "ftol": self.slsqp_ftol,
                "maxiter": self.max_slsqp_iter,
                "disp": False,
            },
        )
        if not result.success:
            log.debug("SLSQP did not converge: %s", result.message)
        return np.asarray(result.x, dtype=np.float64), bool(result.success), str(result.message)

    def _unpack_z(self, z):
        n = self.nv
        qddot = np.asarray(z[:n], dtype=np.float64).copy()
        tau = np.asarray(z[n : 2 * n], dtype=np.float64).copy()
        wrench = np.asarray(z[2 * n : 2 * n + 6], dtype=np.float64).copy()
        return qddot, tau, wrench

    def _pack_z(self, qddot, tau, wrench):
        return np.concatenate(
            [
                np.asarray(qddot, dtype=np.float64).reshape(self.nv),
                np.asarray(tau, dtype=np.float64).reshape(self.nv),
                np.asarray(wrench, dtype=np.float64).reshape(6),
            ]
        )
