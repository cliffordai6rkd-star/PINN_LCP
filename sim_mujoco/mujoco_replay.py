import logging
import time
from pathlib import Path

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim_mujoco.mujocosim_inteface import MujocoSim_interface_fr3



logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class Sim_replayer:
    def __init__(self, config):
        self.config = config

    def load_q_sequence(self, ep_idx):
        dataset = LeRobotDataset(
            repo_id=self.config["repo_id"],
            root=Path(self.config["root"]),
            video_backend=self.config.get("video_backend", "torchcodec"),
        )
        fps = getattr(dataset, "fps", None)
        fps = float(fps)
        # log.info(f"data collect fps:{fps} ")

        
        q_seq = []
        
        
        q_key = self.config.get("q_key", "observation.joint")
        episode = dataset.meta.episodes[ep_idx]
        start_frame_idx = int(episode["dataset_from_index"])
        end_frame_idx = int(episode["dataset_to_index"])
        
        for idx in range(start_frame_idx, end_frame_idx):
            
            frame = dataset.hf_dataset[idx]
            q = frame[q_key]
            # log.info(f"q commond{q}")
            # time.sleep(1/fps)
            if hasattr(q, "detach"):
                q = q.detach().cpu().numpy()
            q_seq.append(q)

        return q_seq, fps
        

    def replayer(self, config):
        episode_start = int(self.config.get("episode_start", 1) - 1)
        episode_end = int(self.config.get("episode_end", 10) - 1)
        
        log.info(f"replaying episode from{episode_start} to {episode_end}")
        viewer = MujocoSim_interface_fr3(config)
        viewer.load_model()
        viewer.print_model_info()
        viewer.save_compiled_mjcf()

        all_q_seqs = []
        fps = None

        for idx in range(episode_start, episode_end):
            log.info(f"loading episode {idx}")
            q_seq, fps = self.load_q_sequence(idx)
            all_q_seqs.append(q_seq)

        viewer.play_joint_sequences(all_q_seqs, dt=1.0 / fps)

if __name__ == "__main__":
    with open("config/sim_cfg/test.yaml", "r") as f:
                config = yaml.safe_load(f)
    replayer = Sim_replayer(config)
    replayer.replayer(config)