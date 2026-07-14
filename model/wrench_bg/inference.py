
import torch
import yaml
import logging
import argparse
import copy

from pathlib import Path
from pinn_model.wrench_bg.wrench_background_v2 import Wrench_Background_V2
from data_process.dataloader import PINNDataset
from train.nomalizer import Normalizer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("WrecnhBgInfrencer")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/inference_cfg/wrench_bg_inference.yaml"),
    )
    return parser.parse_args()

class WrecnhBgInferencer:
    def __init__(self, config):
        self.config = config
        self.inference_cfg = config.get("inferencer", {})

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.ckpt_path = Path(self.inference_cfg.get("ckpt_path", None))
        if self.ckpt_path is None:
            raise ValueError(f"miss ckpt path")
        self.get_ckpt()
        self.load_dataset()
        

    def _build_dataset_config(self, ckpt_cfg):
        dataset_config = copy.deepcopy(ckpt_cfg)
        dataloader_cfg = dataset_config["dataloader"]

        dataset_override = self.config.get("dataset") or {}
        for key in ("root", "repo_id", "video_backend", "load_images"):
            if key in dataset_override:
                dataloader_cfg[key] = dataset_override[key]

        return dataset_config
    
    def get_ckpt(self):
        log.info(f"Loading checkpoint------------------------")
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        ckpt_cfg = ckpt["config"]
        log.info(f"Found checkpoint at {self.ckpt_path}")
        log.info(f"checkpoint config {ckpt_cfg}")

        model = Wrench_Background_V2(ckpt_cfg).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        self.ckpt = ckpt

        normalizer_state = ckpt.get("normalizer")
        if normalizer_state is None:
            raise KeyError("checkpoint missing normalizer stats")

        self.ckpt_normalizer = Normalizer(
            stats=normalizer_state["stats"],
            eps=normalizer_state.get("eps", 1e-6),
        )
        self.ckpt_normalize_mode = normalizer_state.get(
            "normalize_mode",
            ckpt_cfg["dataloader"].get("normalize_mode"),
        )
        self.ckpt_cfg = ckpt_cfg
        self.dataset_config = self._build_dataset_config(ckpt_cfg)
        self.model = model

        return model, ckpt_cfg, ckpt
    
    def inferecer_one_step(self, batch):
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            out = self.model(batch)

        return out["wrench_pred"]
    
    def load_dataset(self):
        log.info("Loading dataset")
        self.dataset = PINNDataset(
            self.dataset_config,
            normalizer=self.ckpt_normalizer,
            normalize_mode=self.ckpt_normalize_mode,
            compute_normalizer=False,
        )
        self.normalizer = self.dataset.normalizer
        log.info(f"dataset normalize_mode: {self.dataset.normalize_mode}")
        log.info(f"dataset normalizer from checkpoint: {self.dataset.normalizer is self.ckpt_normalizer}")
        log.info(f"model active_inputs: {self.model.active_inputs}")
        log.info("using checkpoint normalizer for inference dataset")

        self.lerobot_dataset = self.dataset.dataset
        self.hf_dataset = self.dataset.dataset.hf_dataset
        dataloader_cfg = self.dataset_config["dataloader"]
        log.info(f"dataset root: {dataloader_cfg.get('root')}")
        log.info(f"dataset repo_id: {dataloader_cfg.get('repo_id')}")
        self.raw_to_sample_idx = {
            raw_idx: sample_idx
            for sample_idx, raw_idx in enumerate(self.dataset.valid_indices)
        }

        log.info(f"dataset length: {len(self.dataset)}")
        log.info(f"num episodes: {len(self.lerobot_dataset.meta.episodes)}")

    def denormalize_wrench(self, wrench_norm):
        normalize_mode = self.ckpt_normalize_mode

        if normalize_mode == "gaussian":
            return self.normalizer.gaussian_denormalize("wrench", wrench_norm)
        if normalize_mode == "limit":
            return self.normalizer.limit_denormalize("wrench", wrench_norm)
        if normalize_mode == "quantile":
            return self.normalizer.quantile_denormalize("wrench", wrench_norm)

        raise ValueError(f"unknown normalize mode: {normalize_mode}")
    
    def infer_one_sample(self, idx):
        sample = self.dataset[idx]

        batch = {
            k: v.unsqueeze(0)
            for k, v in sample.items()
            if k in self.model.active_inputs
        }

        wrench_bg_norm = self.inferecer_one_step(batch)
        wrench_bg = self.denormalize_wrench(wrench_bg_norm)
        if wrench_bg.ndim == 3:
            wrench_bg = wrench_bg[:, -1, :]

        raw_idx = self.dataset.valid_indices[idx]
        raw_frame = self.hf_dataset[raw_idx]
        wrench_key = self.dataset_config["dataloader"]["lowdim_keys"]["wrench"]
        raw_wrench = raw_frame[wrench_key].unsqueeze(0).to(wrench_bg.device)

        lambda_wrench = raw_wrench - wrench_bg

        return {
            "raw_idx": raw_idx,
            "raw_wrench": raw_wrench.squeeze(0).cpu(),
            "wrench_bg": wrench_bg.squeeze(0).cpu(),
            "lambda_wrench": lambda_wrench.squeeze(0).cpu(),
        }

    def infer_one_episode(self, episode_idx):
        episode = self.lerobot_dataset.meta.episodes[episode_idx]

        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])

        raw_indices = []
        raw_wrench_list = []
        wrench_bg_list = []
        lambda_wrench_list = []

        for raw_idx in range(start, end):
            if raw_idx not in self.raw_to_sample_idx:
                continue

            sample_idx = self.raw_to_sample_idx[raw_idx]
            result = self.infer_one_sample(sample_idx)

            raw_indices.append(raw_idx)
            raw_wrench_list.append(result["raw_wrench"])
            wrench_bg_list.append(result["wrench_bg"])
            lambda_wrench_list.append(result["lambda_wrench"])

        episode_result = {
            "episode_idx": episode_idx,
            "raw_indices": raw_indices,
            "raw_wrench": torch.stack(raw_wrench_list, dim=0),
            "wrench_bg": torch.stack(wrench_bg_list, dim=0),
            "lambda_wrench": torch.stack(lambda_wrench_list, dim=0),
        }

        log.info(f"episode {episode_idx} raw_wrench shape: {episode_result['raw_wrench'].shape}")
        log.info(f"episode {episode_idx} wrench_bg shape: {episode_result['wrench_bg'].shape}")
        log.info(f"episode {episode_idx} lambda_wrench shape: {episode_result['lambda_wrench'].shape}")

        return episode_result
    
    def plot_episode_result(self, episode_result):
        output_dir = self.inference_cfg.get("inference_plot_save_path") or "outputs/wrench_bg/inference_plots"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        episode_idx = episode_result["episode_idx"]

        raw = episode_result["raw_wrench"]
        bg = episode_result["wrench_bg"]
        lam = episode_result["lambda_wrench"]
        if raw.ndim != 2:
            raise ValueError(f"expected raw_wrench shape [T, 6], got {tuple(raw.shape)}")
        if bg.ndim != 2:
            raise ValueError(f"expected wrench_bg shape [T, 6], got {tuple(bg.shape)}")
        if lam.ndim != 2:
            raise ValueError(f"expected lambda_wrench shape [T, 6], got {tuple(lam.shape)}")

        raw_last = raw
        bg_last = bg
        lam_last = lam

        names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

        def use_data_ylim(axes, *arrays):
            values = torch.cat([array.reshape(-1) for array in arrays])
            finite_values = values[torch.isfinite(values)]
            if finite_values.numel() == 0:
                return
            max_abs = float(finite_values.abs().max().item())
            if max_abs <= 0:
                max_abs = 1.0
            pad = max_abs * 0.05
            ylim = (-max_abs - pad, max_abs + pad)
            for axis in axes:
                axis.set_ylim(ylim)

        # 图 1：lambda 单独曲线
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

        for i, name in enumerate(names):
            axes[i].plot(lam_last[:, i], label=f"lambda {name}", color="tab:red")
            axes[i].set_ylabel(name)
            axes[i].grid(True)
            axes[i].legend(loc="upper right")

        use_data_ylim(axes, lam_last)
        axes[-1].set_xlabel("frame")
        fig.tight_layout()

        path_lambda = output_dir / f"episode_{episode_idx:03d}_lambda.png"
        fig.savefig(path_lambda)
        plt.close(fig)

        # 图 2：raw wrench、wrench_bg 和 lambda 对比
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

        for i, name in enumerate(names):
            axes[i].plot(raw_last[:, i], label=f"raw {name}", color="tab:blue")
            axes[i].plot(bg_last[:, i], label=f"wrench_bg {name}", color="gold")
            axes[i].plot(lam_last[:, i], label=f"lambda {name}", color="tab:red")
            axes[i].set_ylabel(name)
            axes[i].grid(True)
            axes[i].legend(loc="upper right")

        use_data_ylim(axes, raw_last, bg_last, lam_last)
        axes[-1].set_xlabel("frame")
        fig.tight_layout()

        path_compare = output_dir / f"episode_{episode_idx:03d}_raw_vs_lambda.png"
        fig.savefig(path_compare)
        plt.close(fig)

        log.info(f"saved plot: {path_lambda}")
        log.info(f"saved plot: {path_compare}")

if __name__ == "__main__":

    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"wrench background train config:{config}")

    inferencer = WrecnhBgInferencer(config)

    # inferencer.test_one_sample()
    num_episodes = config.get("num_episodes", 100)

    for episode_idx in range(num_episodes):
        log.info(f"plotting episode {episode_idx}/{num_episodes - 1}")
        episode_result = inferencer.infer_one_episode(episode_idx)
        inferencer.plot_episode_result(episode_result)
