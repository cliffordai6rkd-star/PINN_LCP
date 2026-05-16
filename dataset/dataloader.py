#Q: 我们研究的方程到底是以t为自变量的ODE还是以q,v,u等为自变量的PDE?
# 若是以t为自变量则应该引入时序建模   若不是则ft是以机器人状态为输入  对应输出ft

# 变量: q v u(action without gripper)
#      wrench(lambda) Fx Fy Fz τx τy τz
# img -> phi\miu


import torch
import os
import argparse
import yaml
import logging as log

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from dataset.nomalizer import Normalizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert standalone .h5/.hdf5 episode files to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/tool/config/dataset_test_cfg.yaml"),
        help="Path to the config",
    )
    return parser.parse_args()

class PINNDataset(torch.utils.data.Dataset):
    def __init__(self,config):
        # repo_id, root, 
        self.config = config 
        self.data_config = config.get("dataloader")
        self.repo_id = self.data_config.get("repo_id",None)
        self.root = Path(self.data_config.get("root", None))
        self.video_backend = self.data_config.get("video_backend", "torchcodec")
        if not self.repo_id or not self.root:
            raise ValueError(f"miss lerobotv3 dataset repo_id and root")
        # self.repo_path = os.path.join(self.root, self.repo_id)
        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            video_backend=self.video_backend
        )
        self.dt = float(1/30)  # 采样frequency 30Hz

        self.valid_indices = []
        episodes = self.dataset.meta.episodes
        for ep in episodes:
            start_idx = int(ep["dataset_from_index"])
            end_idx = int(ep["dataset_to_index"])
            for idx in range(start_idx, end_idx - 1):
                self.valid_indices.append(idx)

        self.normalize_lowdim_keys = self.data_config.get("normalize_lowdim_keys",None)
        if self.normalize_lowdim_keys is None:
            raise ValueError(f"miss normalize lowdim keys")
        self.lowdim_keys = self.data_config.get("lowdim_keys", {})
        self.image_keys = self.data_config.get("image_keys", {})
        self.normalize_mode = None
        self.is_normalize = False
        self.normalizer = None
        self.normalize_fuc = None
        self.normalize_mode = self.data_config.get("normalize_mode", "gaussian")
        
        if self.normalize_mode is not None:
            self.is_normalize = True
            self.normalizer = Normalizer.stats_from_dataset(
                dataset = self.dataset,
                valid_indices = self.valid_indices,
                lowdim_keys = self.lowdim_keys,
                normalize_keys = self.normalize_lowdim_keys,
            )
            if self.normalize_mode == "gaussian":
                self.normalize_fuc = self.normalizer.gaussian_normalize
            elif self.normalize_mode == "limit":
                self.normalize_fuc = self.normalizer.limit_normalize
            elif self.normalize_mode == "quantile":
                self.normalize_fuc = self.normalizer.quantile_normalize
            else:
                raise ValueError(f"unknown normalize mode")
        
    def __len__(self):
        return len(self.valid_indices) 
    
    def __getitem__(self, idx):

        valid_idx = self.valid_indices[idx]
        cur = self.dataset[valid_idx]
        nxt = self.dataset[valid_idx+1]

        sample = {}
        
        # ("q", "observation.joint") —> key = "q" , dataset_key = "observation.joint"
        for key, dataset_key in self.lowdim_keys.items():
            sample[f"{key}_cur"] = cur[dataset_key]
            sample[f"{key}_nxt"] = nxt[dataset_key]


        

        for key in self.normalize_lowdim_keys:
            if self.is_normalize:
                sample[f"{key}_cur"] = self.normalize_fuc(key, sample[f"{key}_cur"])
                sample[f"{key}_nxt"] = self.normalize_fuc(key, sample[f"{key}_nxt"])


        for key, dataset_key in self.image_keys.items():
            sample[f"image_{key}_cur"] = cur[dataset_key]
            sample[f"image_{key}_nxt"] = nxt[dataset_key]

        return sample


   

        
if __name__ == "__main__":

    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"config :{config}")
    dataset = PINNDataset(config)
    log.info(f"dataset len : {len(dataset)}")
    assert len(dataset) > 0

    sample = dataset[0]
    log.info(f"sample keys: {sample.keys()}")
    log.info(f"sample success")

    loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers = 4)
    
    batch = next(iter(loader))

    for k, v in batch.items():
        log.info(f"batch data shape : {k, v.shape, v.dtype}")
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"{k} has nan or inf" # 检查是否有非法数值

    