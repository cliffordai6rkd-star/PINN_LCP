import torch
import logging
import time
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, stats, eps=1e-6):
        self.stats = stats
        self.eps = eps


    @classmethod
    def stats_from_dataset(cls, dataset, valid_indices, eps=1e-6):
        start_time = time.perf_counter()
        log.info(f"start computing normalizer stats, num_samples={len(valid_indices)}")

        q_list = []
        v_list = []
        action_list = []
        # wrench_list = []

        for idx in tqdm(valid_indices, desc="compute normalizer stats", unit="sample"):
            cur = dataset[idx]

            q = cur["observation.joint"]
            v = cur["observation.velocity"]
            action = cur["action.ee_pose"]
            torque = cur["observation.torque"]
            

            # wrench = cur["observation.ft_window"]

            q_list.append(q)
            v_list.append(v)
            action_list.append(u)
            # wrench_list.append(wrench)

        log.info("finished loading samples, start stacking tensors")

        q_all = torch.stack(q_list, dim=0)
        log.info(f"q_all shape={q_all.shape}")
        v_all = torch.stack(v_list, dim=0)
        log.info(f"v_all shape={v_all.shape}")
        u_all = torch.stack(u_list, dim=0)
        log.info(f"u_all shape={u_all.shape}")
        # wrench_all = torch.stack(wrench_list, dim=0)
        # log.info(f"u_all shape={wrench_all.shape}")

        stats = {
            "q": {
                "mean": q_all.mean(dim=0),
                "std": q_all.std(dim=0),
                "min": q_all.min(dim=0).values,
                "max": q_all.max(dim=0).values,
                "q01": torch.quantile(q_all, 0.01, dim=0),
                "q99": torch.quantile(q_all, 0.99, dim=0) 
            },
            "v": {
                "mean": v_all.mean(dim=0),
                "std": v_all.std(dim=0),
                "min": v_all.min(dim=0).values,
                "max": v_all.max(dim=0).values,
                "q01": torch.quantile(v_all, 0.01, dim=0),
                "q99": torch.quantile(v_all, 0.99, dim=0) 
            },
            "u": {
                "mean": u_all.mean(dim=0),
                "std": u_all.std(dim=0),
                "min": u_all.min(dim=0).values,
                "max": u_all.max(dim=0).values,
                "q01": torch.quantile(u_all, 0.01, dim=0),
                "q99": torch.quantile(u_all, 0.99, dim=0) 
            },
            # "wrench":{
            #     "mean": wrench_all.mean(dim=0),
            #     "std": wrench_all.std(dim=0),
            #     "min": wrench_all.min(dim=0).values,
            #     "max": wrench_all.max(dim=0).values,
            #     "q01": torch.quantile(wrench_all, 0.01, dim=0),
            #     "q99": torch.quantile(wrench_all, 0.99, dim=0) 
            # }
        }
        read_time = time.perf_counter() - start_time
        log.info("normalizer stats computed successfully")
        log.info(f"stats calculating time :{read_time:.2f}s")
        return cls(stats, eps=eps)

    
    def gaussian_normalize(self, key, x):
        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"]
        return (x - mean) / (std + self.eps)

    def limit_normalize(self, key, x):
        min_v = self.stats[key]["min"]
        max_v = self.stats[key]["max"]
        return 2 * (x - min_v) / (max_v - min_v + self.eps) - 1

    def quantile_normalize(self, key ,x):
        q01 = self.stats[key]["q01"]
        q99 = self.stats[key]["q99"]
        x = 2 * (x - q01) / (q99 - q01 + self.eps) - 1
        return torch.clamp(x, -1.0, 1.0)

