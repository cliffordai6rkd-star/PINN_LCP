import torch
import logging
import time
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class Normalizer:
    def __init__(self, stats, eps=1e-6,):

        self.stats = stats
        self.eps = eps


    @classmethod
    def stats_from_dataset(cls, dataset, valid_indices,lowdim_keys, normalize_keys, eps=1e-6):
        start_time = time.perf_counter()
        log.info(f"start computing normalizer stats, num_samples={len(valid_indices)}")

        stats = {}
        buffers = {key:[] for key in normalize_keys}

        for idx in tqdm(valid_indices, desc="compute normalizer stats", unit="sample"):
            cur = dataset[idx]
            for key in normalize_keys:
                dataset_key = lowdim_keys[key]
                buffers[key].append(cur[dataset_key])

        for key, values in buffers.items():
            x = torch.stack(values, dim=0)
            stats[key] = {
                "mean": x.mean(dim=0),
                "std": x.std(dim=0),
                "min": x.min(dim=0).values,
                "max": x.max(dim=0).values,
                "q01": torch.quantile(x, 0.01, dim=0),
                "q99": torch.quantile(x, 0.99, dim=0),
            }

        log.info("finished loading samples, start stacking tensors")

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

