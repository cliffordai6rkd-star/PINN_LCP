import argparse
import logging
import time
from pathlib import Path

import mujoco
import mujoco.viewer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Load FR3 Franka hand URDF in MuJoCo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("sim_mesh/franka_fr3/fr3_franka_hand.urdf"),
        help="URDF or MJCF path to load.",
    )
    parser.add_argument(
        "--save-xml",
        type=Path,
        default=None,
        help="Optional path for MuJoCo's compiled MJCF XML.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Only load and print model info.")
    parser.add_argument("--duration", type=float, default=None, help="Viewer duration in seconds.")
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


def save_compiled_xml(model, output_path):
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output_path), model)
    log.info("saved compiled MJCF: %s", output_path)


def launch_viewer(model, data, duration):
    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if duration is not None and time.monotonic() - start >= duration:
                break


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    log.info("loading model: %s", model_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print_model_info(model)

    if args.save_xml is not None:
        save_compiled_xml(model, args.save_xml)

    if not args.no_viewer:
        launch_viewer(model, data, args.duration)


if __name__ == "__main__":
    main()
