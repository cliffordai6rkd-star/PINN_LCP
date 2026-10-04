"""Shared ingestion-time FK for absolute joint-action conditions."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_process.tool.VA_h5_v3 import homogeneous_pose_to_xyz_quat_xyzw


def normalize_action_fk(value: Mapping[str, Any] | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("action_fk must be a mapping")
    if not value.get("enabled", True):
        return None
    allowed = {"enabled", "joint_key", "pose_key", "urdf_path", "base_frame", "frame_name", "joint_names"}
    if set(value) - allowed:
        raise ValueError(f"Unknown action_fk options: {sorted(set(value) - allowed)}")
    if not value.get("urdf_path"):
        raise ValueError("action_fk.urdf_path is required; FK must use the dataset robot's URDF")
    result = {
        "joint_key": "action.joint",
        "pose_key": "action.ee_pose",
        "base_frame": "link_base",
        "frame_name": "link_eef",
        **{key: item for key, item in value.items() if key != "enabled"},
    }
    for key in ("joint_key", "pose_key", "base_frame", "frame_name"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"action_fk.{key} must be a nonempty string")
    if result["joint_key"] == result["pose_key"]:
        raise ValueError("action_fk pose_key must differ from joint_key")
    return result


def compute_eeposes(
    joints: Any,
    urdf_path: Path,
    *,
    base_frame: str = "link_base",
    frame_name: str = "link_eef",
    joint_names=None,
) -> np.ndarray:
    """Compute metres + unit xyzw quaternion (qw >= 0) from radian joints.

    All moving joints must be scalar and supplied in joint_names order. When
    omitted, the URDF's joint order is used. Fixed tool offsets are included.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or not len(joints) or not joints.shape[1]:
        raise ValueError(f"Joint values must have nonempty shape (N, D), got {joints.shape}.")
    if not np.isfinite(joints).all():
        raise ValueError("Joint values contain non-finite values.")
    urdf_path = Path(urdf_path).expanduser()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Missing URDF: {urdf_path}")
    try:
        import pinocchio as pin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Joint-action FK requires pinocchio.") from exc

    model = pin.buildModelFromUrdf(str(urdf_path))
    names = list(model.names[1:]) if joint_names is None else list(joint_names)
    if (len(names) != joints.shape[1] or len(set(names)) != len(names)
            or set(names) != set(model.names[1:])):
        raise ValueError("joint_names must list every moving URDF joint exactly once and match the action dimension.")
    if model.nq != len(names) or model.nv != len(names):
        raise ValueError("FK requires scalar URDF joints with one configuration coordinate each.")
    indexes = []
    for name in names:
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"URDF joint {name!r} must have one configuration coordinate.")
        indexes.append(joint.idx_q)
    for name in (base_frame, frame_name):
        if not model.existFrame(name):
            raise ValueError(f"URDF does not contain frame {name!r}.")

    base_id, frame_id = model.getFrameId(base_frame), model.getFrameId(frame_name)
    data = model.createData()
    configuration = np.zeros(model.nq)
    transforms = np.empty((len(joints), 4, 4), dtype=np.float64)
    for index, values in enumerate(joints):
        configuration[indexes] = values
        pin.framesForwardKinematics(model, data, configuration)
        transforms[index] = (data.oMf[base_id].inverse() * data.oMf[frame_id]).homogeneous
    return homogeneous_pose_to_xyz_quat_xyzw(transforms, np)


def poses_from_joint_actions(joints: Any, config: Mapping[str, Any]) -> np.ndarray:
    """FK unique joint actions once, then restore held or packed source rows."""
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim not in (2, 3):
        raise ValueError(f"Joint actions must have [N,D] or [N,S,D], got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError(f"{config['joint_key']} contains non-finite joint actions")
    unique, inverse = np.unique(joints.reshape(-1, joints.shape[-1]), axis=0, return_inverse=True)
    poses = compute_eeposes(
        unique, Path(config["urdf_path"]), base_frame=config["base_frame"],
        frame_name=config["frame_name"], joint_names=config.get("joint_names"),
    )
    return poses[inverse].reshape(*joints.shape[:-1], 7)
