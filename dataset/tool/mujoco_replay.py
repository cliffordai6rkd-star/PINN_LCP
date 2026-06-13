import logging
import time
from pathlib import Path

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset.tool.mujocosim_inteface import MujocoSim_interface_fr3



logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Sim_replayer:
    def __init__(self, config):
        self.config = config

    def load_q_sequence(self):
        dataset = LeRobotDataset(
            repo_id=self.config["repo_id"],
            root=Path(self.config["root"]),
            video_backend=self.config.get("video_backend", "torchcodec"),
        )
        fps = getattr(dataset, "fps", None)
        fps = float(fps)
        log.info(f"data collect fps:{fps} ")

        episode_index = int(self.config.get("episode_index", 0))
        log.info(f"replaying episdoe{episode_index}")
        q_key = self.config.get("q_key", "observation.joint")

        episode = dataset.meta.episodes[episode_index]
        start_idx = int(episode["dataset_from_index"])
        end_idx = int(episode["dataset_to_index"])

        q_seq = []
        for idx in range(start_idx, end_idx):
            frame = dataset.hf_dataset[idx]
            q = frame[q_key]

            if hasattr(q, "detach"):
                q = q.detach().cpu().numpy()

            q_seq.append(q)

        return q_seq, fps

def main():
        with open("dataset/config/sim_cfg/test.yaml", "r") as f:
            config = yaml.safe_load(f)

        viewer = MujocoSim_interface_fr3(config)
        viewer.load_model()
        viewer.print_model_info()
        viewer.save_compiled_mjcf()

        relayer = Sim_replayer(config)
        q_seq,fps = relayer.load_q_sequence()
        viewer.play_joint_sequence(q_seq, dt=1.0 / fps)

if __name__ == "__main__":
     main()