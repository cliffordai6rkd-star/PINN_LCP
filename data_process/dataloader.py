# 模型学习一个长度为horizon的window  不仅学习点到点的映射关系 也学习力的变化趋势
# 变量: q v u(action without gripper, tau)
#      wrench(lambda) Fx Fy Fz τx τy τz
# img -> phi\miu -> loss


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
from train.nomalizer import Normalizer

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
    def __init__(self, config,
                    normalizer=None,
                    normalize_mode=None,
                    compute_normalizer=True,):
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
        self.stats_dataset =  self.dataset.hf_dataset
        # self.dt = float(1/30)  # 采样frequency 30Hz

        self.horizon = int(self.data_config.get("horizon", 1))
        self.history_horizon = int(self.data_config.get("history_horizon", self.horizon))
        self.future_horizon = int(self.data_config.get("future_horizon", self.horizon))

        self.valid_indices = []
        self.raw_idx_to_episode_start = {}
        self.raw_idx_to_episode_end = {}
        self._build_valid_indices()

        self.load_image = bool(self.data_config.get("load_images",True))

        self.normalize_lowdim_keys = self.data_config.get("normalize_lowdim_keys",None)
        if self.normalize_lowdim_keys is None:
            self.normalize_lowdim_keys = []
        self.lowdim_keys = self.data_config.get("lowdim_keys", {})
        if self.load_image:
            self.image_keys = self.data_config.get("image_keys", {})
        else:
            self.image_keys = {}
        self.normalize_mode = None
        self.is_normalize = False
        self.normalizer = None
        self.normalize_fuc = None
        self.normalize_mode = normalize_mode
        if self.normalize_mode is None:
            self.normalize_mode = self.data_config.get("normalize_mode", "gaussian")

        if self.normalize_mode is not None:
            self.is_normalize = True

            if normalizer is not None:
                self.normalizer = normalizer
            elif compute_normalizer:
                self.normalizer = Normalizer.stats_from_dataset(
                    dataset = self.stats_dataset,
                    valid_indices = self.valid_indices,
                    lowdim_keys = self.lowdim_keys,
                    normalize_keys = self.normalize_lowdim_keys,
                )
            else:
                raise ValueError("normalizer is required when compute_normalizer=False")

            if self.normalize_mode == "gaussian":
                self.normalize_fuc = self.normalizer.gaussian_normalize
            elif self.normalize_mode == "limit":
                self.normalize_fuc = self.normalizer.limit_normalize
            elif self.normalize_mode == "quantile":
                self.normalize_fuc = self.normalizer.quantile_normalize
            else:
                raise ValueError(f"unknown normalize mode: {self.normalize_mode}")

    def _build_valid_indices(self):
        episodes = self.dataset.meta.episodes

        for ep in episodes:
            start_idx = int(ep["dataset_from_index"])
            end_idx = int(ep["dataset_to_index"])

            for idx in range(start_idx, end_idx):
                self.valid_indices.append(idx)
                self.raw_idx_to_episode_start[idx] = start_idx
                self.raw_idx_to_episode_end[idx] = end_idx

    def __len__(self):
        return len(self.valid_indices) 
    
    def __getitem__(self, idx):

        raw_idx = self.valid_indices[idx]
        episode_start = self.raw_idx_to_episode_start[raw_idx]

        # 构造horizon窗口  用max来结局解决开头几帧的问题  相当于padding
        frame_indices = [
                max(episode_start, raw_idx - self.horizon + 1 + offset)
                for offset in range(self.horizon)
        ]


        frames = [self._read_frame(i) for i in frame_indices]

        sample = {}
        
        # ("q", "observation.joint") —> key = "q" , dataset_key = "observation.joint"
        for key, dataset_key in self.lowdim_keys.items():
            seq = [frame[dataset_key] for frame in frames]
            sample[f"{key}"] = torch.stack(seq, dim=0)

        for key in self.normalize_lowdim_keys:
            if self.is_normalize:
                sample[f"{key}"] = self.normalize_fuc(key, sample[f"{key}"])
    
        for key, dataset_key in self.image_keys.items():
            seq = [frame[dataset_key] for frame in frames]
            sample[f"image_{key}"] = torch.stack(seq, dim=0)            

        return sample
    
    def _read_frame(self, i):
        if self.load_image:
            return self.dataset[i]
        return self.dataset.hf_dataset[i]


class PINNHistoryFutureDataset(PINNDataset):
    def _build_valid_indices(self):
        episodes = self.dataset.meta.episodes
        self.pad_future = bool(self.data_config.get("pad_future", False))

        for ep in episodes:
            start_idx = int(ep["dataset_from_index"])
            end_idx = int(ep["dataset_to_index"])
            if self.pad_future:
                valid_end = end_idx
            else:
                valid_end = end_idx - self.future_horizon

            for idx in range(start_idx, max(start_idx, valid_end)):
                self.valid_indices.append(idx)
                self.raw_idx_to_episode_start[idx] = start_idx
                self.raw_idx_to_episode_end[idx] = end_idx

    def __getitem__(self, idx):
        raw_idx = self.valid_indices[idx]
        episode_start = self.raw_idx_to_episode_start[raw_idx]
        episode_end = self.raw_idx_to_episode_end[raw_idx]

        # raw_idx 是历史观测窗口的最后一帧，未来窗口从 raw_idx + 1 开始。
        history_indices = [
            max(episode_start, raw_idx - self.history_horizon + 1 + offset)
            for offset in range(self.history_horizon)
        ]
        future_indices = [
            min(episode_end - 1, raw_idx + 1 + offset)
            for offset in range(self.future_horizon)
        ]

        history_frames = [self._read_frame(i) for i in history_indices]
        future_frames = [self._read_frame(i) for i in future_indices]

        sample = {
            "raw_idx": torch.tensor(raw_idx, dtype=torch.long),
            "history_indices": torch.tensor(history_indices, dtype=torch.long),
            "future_indices": torch.tensor(future_indices, dtype=torch.long),
        }

        pinocchio_raw_keys = set(
            self.data_config.get("pinocchio_raw_keys", ["q", "v", "a", "tau"])
        )

        for key, dataset_key in self.lowdim_keys.items():
            history_seq = torch.stack([frame[dataset_key] for frame in history_frames], dim=0)
            future_seq = torch.stack([frame[dataset_key] for frame in future_frames], dim=0)

            sample[key] = history_seq
            sample[f"{key}_future"] = future_seq

            # Pinocchio 必须使用真实物理量；如果后续对 q/v/a/tau 做归一化，这里保留原始未来量。
            if key in pinocchio_raw_keys:
                sample[f"{key}_future_raw"] = future_seq.clone()

        for key in self.normalize_lowdim_keys:
            if self.is_normalize:
                sample[key] = self.normalize_fuc(key, sample[key])
                future_key = f"{key}_future"
                if future_key in sample:
                    sample[future_key] = self.normalize_fuc(key, sample[future_key])

        for key, dataset_key in self.image_keys.items():
            seq = [frame[dataset_key] for frame in history_frames]
            sample[f"image_{key}"] = torch.stack(seq, dim=0)

        return sample
        
if __name__ == "__main__":





    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"config :{config}")
    dataset = PINNDataset(config)
    log.info(f"dataset len : {len(dataset)}")
    assert len(dataset) > 0

    log.info(f"dataset len: {len(dataset)}")
    log.info(f"horizon: {dataset.horizon}")
    log.info(f"num episodes: {len(dataset.dataset.meta.episodes)}")
    log.info(f"first valid index: {dataset.valid_indices[0]}")


    for i in range(1,100):
        sample = dataset[i]
        log.info(f"sample keys: {sample.keys()}")
        log.info(f"sample success")
        # for k, v in sample.items():
        #     log.info(f"sample {k}: shape={v.shape}, dtype={v.dtype}")
        #     if torch.is_tensor(v) and v.is_floating_point():
        #         assert torch.isfinite(v).all(), f"{k} has nan or inf"

        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=4 ,
                                            shuffle=False,
                                            num_workers=4)

        batch = next(iter(loader))

        for k, v in batch.items():
            log.info(f"batch data shape : {k, v.shape, v.dtype}")
            if torch.is_tensor(v) and v.is_floating_point():
                assert torch.isfinite(v).all(), f"{k} has nan or inf" # 检查是否有非法数值

    
