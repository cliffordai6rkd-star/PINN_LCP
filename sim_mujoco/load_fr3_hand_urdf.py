import argparse
import logging
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


FRAME_MODES = {
    "none": mujoco.mjtFrame.mjFRAME_NONE,
    "body": mujoco.mjtFrame.mjFRAME_BODY,
    "site": mujoco.mjtFrame.mjFRAME_SITE,
    "geom": mujoco.mjtFrame.mjFRAME_GEOM,
    "world": mujoco.mjtFrame.mjFRAME_WORLD,
    "camera": mujoco.mjtFrame.mjFRAME_CAMERA,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Load FR3 Franka hand scene in MuJoCo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("sim_mesh/franka_fr3/franka_ati_hand.urdf"),
        help="URDF or MJCF path to load.",
    )
    parser.add_argument(
        "--keyframe",
        default="home",
        help="Keyframe used to initialize qpos when --qpos is not provided.",
    )
    parser.add_argument(
        "--qpos",
        type=float,
        nargs="+",
        default=None,
        help="Optional explicit qpos. Must contain model.nq values.",
    )
    parser.add_argument(
        "--frame-mode",
        choices=sorted(FRAME_MODES),
        default="body",
        help="Coordinate frame axes to show in the viewer.",
    )
    parser.add_argument(
        "--check-y-align",
        action="store_true",
        help="Check whether ATI local y axis is aligned with Franka local y axis.",
    )
    parser.add_argument(
        "--ati-frame",
        default="ft_ati_m8_link",
        help="Body or site name used as ATI frame for --check-y-align.",
    )
    parser.add_argument(
        "--franka-frame",
        default="fr3_hand",
        help="Body or site name used as Franka frame for --check-y-align.",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=None,
        help="Optional path for MuJoCo's compiled MJCF XML.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Only load and print model info.")
    parser.add_argument("--duration", type=float, default=None, help="Viewer duration in seconds.")
    parser.add_argument("--step", action="store_true", help="Run physics instead of holding the fixed qpos.")
    return parser.parse_args()


def name_of(model, obj_type, idx):
    name = mujoco.mj_id2name(model, obj_type, idx)
    return name if name is not None else ""


def print_model_info(model):
    log.info("nq=%s nv=%s nu=%s nbody=%s njnt=%s ngeom=%s nsite=%s", model.nq, model.nv, model.nu, model.nbody, model.njnt, model.ngeom, model.nsite)

    log.info("joints:")
    for idx in range(model.njnt):
        log.info("  %02d %s", idx, name_of(model, mujoco.mjtObj.mjOBJ_JOINT, idx))

    log.info("actuators:")
    for idx in range(model.nu):
        log.info("  %02d %s", idx, name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx))

    log.info("sites:")
    for idx in range(model.nsite):
        log.info("  %02d %s", idx, name_of(model, mujoco.mjtObj.mjOBJ_SITE, idx))


def apply_initial_qpos(model, data, keyframe_name, qpos):
    if qpos is not None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (model.nq,):
            raise ValueError(f"--qpos expects {model.nq} values, got {qpos.size}")
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        log.info("initialized qpos from --qpos")
        return

    if not keyframe_name:
        mujoco.mj_forward(model, data)
        return

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        log.warning("keyframe not found: %s; using model default qpos", keyframe_name)
        mujoco.mj_forward(model, data)
        return

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    log.info("initialized qpos from keyframe: %s", keyframe_name)


def frame_rotation(model, data, frame_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    if body_id >= 0:
        return data.xmat[body_id].reshape(3, 3), "body"

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id >= 0:
        return data.site_xmat[site_id].reshape(3, 3), "site"

    raise ValueError(f"frame not found as body or site: {frame_name}")


def check_y_axis_alignment(model, data, ati_frame, franka_frame):
    ati_rotation, ati_kind = frame_rotation(model, data, ati_frame)
    franka_rotation, franka_kind = frame_rotation(model, data, franka_frame)

    ati_y_world = ati_rotation[:, 1]
    franka_y_world = franka_rotation[:, 1]
    cos_angle = float(np.clip(np.dot(ati_y_world, franka_y_world), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    log.info("ATI frame %s (%s) y axis in world: %s", ati_frame, ati_kind, np.round(ati_y_world, 6))
    log.info(
        "Franka frame %s (%s) y axis in world: %s",
        franka_frame,
        franka_kind,
        np.round(franka_y_world, 6),
    )
    log.info("y-axis dot=%+.6f angle=%.3f deg", cos_angle, angle_deg)
    if abs(cos_angle) > 0.999:
        log.info("result: y axes are parallel; sign=%s", "same" if cos_angle > 0 else "opposite")
    else:
        log.info("result: y axes are not aligned")


def save_compiled_xml(model, output_path):
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output_path), model)
    log.info("saved compiled MJCF: %s", output_path)


def launch_viewer(model, data, duration, frame_mode, step):
    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.opt.frame = FRAME_MODES[frame_mode]

        while viewer.is_running():
            if step:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            if duration is not None and time.monotonic() - start >= duration:
                break
            time.sleep(0.01)


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    log.info("loading model: %s", model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    apply_initial_qpos(model, data, args.keyframe, args.qpos)
    print_model_info(model)

    if args.check_y_align:
        check_y_axis_alignment(model, data, args.ati_frame, args.franka_frame)

    if args.save_xml is not None:
        save_compiled_xml(model, args.save_xml)

    if not args.no_viewer:
        launch_viewer(model, data, args.duration, args.frame_mode, args.step)


if __name__ == "__main__":
    main()
