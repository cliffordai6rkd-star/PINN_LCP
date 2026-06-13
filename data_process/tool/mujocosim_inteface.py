import mujoco
import mujoco.viewer
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)




class MujocoSim_interface_fr3:
    def __init__(self,config):
        model_path = config.get("model_path", "sim_mesh/franka_fr3/fr3_pika_gripper_ati.urdf")
        self.model_path = Path(model_path).expanduser().resolve()
        output_path = config.get("output_path", None)
        self.output_path = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else None
        )
        self.print_info = config.get("print_info", True)
        self.save_mjcf = config.get("save_mjcf", False)
        self.show_sim = config.get("show_sim", True)
        
        self.model = None
        self.data = None

        self.world_frame_size = config.get("world_frame_size", 0.2)
        self.quick_replay = config.get("quick_replay", True)

    # Load .urdf or .xml.
    def load_model(self):
        log.info(f"loading model from {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def ensure_loaded(self):
        if self.model is None or self.data is None:
            self.load_model()

    def name_of(self, obj_type, idx):
        self.ensure_loaded()
        name = mujoco.mj_id2name(self.model, obj_type, idx)
        return name if name is not None else ""

    def print_model_info(self):
        if not self.print_info:
            return
        self.ensure_loaded()

        log.info(f"nq: {self.model.nq}")
        log.info(f"nv: {self.model.nv}")
        log.info(f"nu: {self.model.nu}")
        log.info(f"nbody: {self.model.nbody}")
        log.info(f"njnt: {self.model.njnt}")
        log.info(f"ngeom: {self.model.ngeom}")
        log.info(f"nsite: {self.model.nsite}")

        log.info("Bodies:")
        for i in range(self.model.nbody):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_BODY, i)}")

        log.info("Joints:")
        for i in range(self.model.njnt):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_JOINT, i)}")

        log.info("Sites:")
        for i in range(self.model.nsite):
            log.info(f"{i} {self.name_of(mujoco.mjtObj.mjOBJ_SITE, i)}")

    def save_compiled_mjcf(self):
        if not self.save_mjcf:
            return None

        self.ensure_loaded()
        path = self.output_path
        if path is None:
            path = self.model_path.with_suffix(".compiled.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(path), self.model)
        self._fix_compiled_meshdir(path)
        log.info(f"saved compiled MJCF to {path}")
        return path

    def show_model(self):
        if not self.show_sim:
            return
    
        self.ensure_loaded()
    
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.sync()
            while viewer.is_running():
                viewer.sync()

    def set_joint_positions(self, q, joint_names=None):
        self.ensure_loaded()
        # log.info(f"joint command::{q}")
        if joint_names is None:
            joint_names = [
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ]
    
        if len(q) != len(joint_names):
            raise ValueError(f"q length {len(q)} != joint_names length {len(joint_names)}")
    
        for value, joint_name in zip(q, joint_names):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"joint not found: {joint_name}")
    
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = value

        mujoco.mj_forward(self.model, self.data)

    def play_joint_sequence(self, q_seq, dt):
        self.ensure_loaded()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for q in q_seq:
                # if not viewer.is_running():
                #     break

                self.set_joint_positions(q)
                viewer.sync()
                if self.quick_replay is False:
                    time.sleep(dt)

    





def main():
    import yaml

    with open("dataset/config/sim_cfg/test.yaml", "r") as f:
        config = yaml.safe_load(f)
    viewer = MujocoSim_interface_fr3(config)
    viewer.load_model()
    log.info("MuJoCo compile OK")
    viewer.print_model_info()
    viewer.save_compiled_mjcf()
    viewer.set_joint_positions(config["q"])
    viewer.show_model()


if __name__ == "__main__":
    main()
