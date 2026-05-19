import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("dataset/config/dataset_test_cfg.yaml"),
    )
    return parser.parse_args()

class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.train_config = config.get("train") or {}

        if torch.cuda.is_available():
            self.device = self.train_config.get("device", "cuda:0")
        else :
            self.device = "cpu"

        self.val_ratio = float(self.train_config.get("val_ratio", 0.1))
        self.seed = int(self.train_config.get("seed", 42))
        self.val_loader = None

        self.batch_size = int(self.train_config.get("batch_size", 64))
        self.num_workers = int(self.train_config.get("num_workers", 4))
        self.lr = float(self.train_config.get("lr", 1e-4))
        self.weight_decay = float(self.train_config.get("weight_decay", 1e-4))
        self.num_epochs = int(self.train_config.get("num_epochs", 20))
        self.monitor_key = self.train_config.get("monitor_key", "val_loss")
        self.top_k = int(self.train_config.get("top_k", 3))
        self.best_checkpoints = []

        self.dataset = None
        self.loader = None
        self.model = None
        self.optimizer = None

        self.output_dir = Path(self.train_config.get("output_dir", "data/outputs/wrench_background"))
        self.ckpt_dir = self.output_dir / "checkpoints"

        self.global_step = 0
        self.loss_history = []

        self.last_summary = None

    def batch_to_device(self, batch):
        new_batch = {}

        for k, v in batch.items():
            if torch.is_tensor(v):
                # log.info(f"{v} is tensor")
                new_batch[k] = v.to(self.device)
            else:
                log.warning(f"{v} is not tensor")
                new_batch[k] = v

        return new_batch
    
    def build_dataset(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError

    def compute_loss(self, batch):
        raise NotImplementedError
    
    def setup(self):
        self.dataset = self.build_dataset()

        if self.val_ratio > 0:
            val_size = int(len(self.dataset) * self.val_ratio)
            train_size = len(self.dataset) - val_size

            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset,
                [train_size, val_size],
                generator=generator,
            )
        else:
            train_dataset = self.dataset
            val_dataset = None

        self.loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        if val_dataset is not None:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        self.model = self.build_model().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        num_steps = 0
        pbar = tqdm(
            self.loader,
            desc=f"train epoch {epoch}",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(pbar):
            batch = self.batch_to_device(batch)

            loss, out = self.compute_loss(batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.global_step += 1

            total_loss += loss.item()
            num_steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "step": self.global_step,
            })

        return total_loss / max(num_steps, 1)
    
    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        if self.val_loader is None:
            return None 

        self.model.eval()

        total_loss = 0.0
        num_steps = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"val epoch {epoch}",
            unit="batch",
            leave=False,
        )

        for batch in pbar:
            batch = self.batch_to_device(batch)
            loss, out = self.compute_loss(batch)

            total_loss += loss.item()
            num_steps += 1

            pbar.set_postfix({
                "val_loss": f"{loss.item():.6f}",
            })

        val_loss = total_loss / max(num_steps, 1)
        # log.info(f"epoch={epoch} val_loss={val_loss:.6f}")
        return val_loss
    
    def train(self):
        self.setup()

        for epoch in range(self.num_epochs):
            avg_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)
            # if val_loss is None:
            #     log.info(f"epoch={epoch} avg_loss={avg_loss:.6f}")
            # else:
            #     log.info(f"epoch={epoch} avg_loss={avg_loss:.6f} val_loss={val_loss:.6f}")

            self.loss_history.append({
                "epoch": epoch,
                "global_step": self.global_step,
                "avg_loss": avg_loss,
                "val_loss": val_loss,
            })

            self.save_loss_plot()

            metrics = {
                "avg_loss": avg_loss,
                "val_loss": val_loss,
            }

            self.save_topk_checkpoint(epoch, metrics)

        self.last_summary = {
            "num_epochs": self.num_epochs,
            "global_step": self.global_step,
            "last_loss": self.loss_history[-1]["avg_loss"] if self.loss_history else None,
            "last_val_loss": self.loss_history[-1]["val_loss"] if self.loss_history else None,
            "monitor_key": self.monitor_key,
            "top_k": self.top_k,
            "best_checkpoints": self.best_checkpoints,
            "output_dir": self.output_dir,
            "ckpt_dir": self.ckpt_dir,
        }

        return self.last_summary
        
    
    def save_loss_plot(self):
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.output_dir.mkdir(parents=True, exist_ok=True)

        steps = [item["global_step"] for item in self.loss_history]
        epochs = [item["epoch"] for item in self.loss_history]
        losses = [item["avg_loss"] for item in self.loss_history]
        val_losses = [item.get("val_loss") for item in self.loss_history]
        has_val = any(loss is not None for loss in val_losses)

        plt.figure()
        plt.plot(epochs, losses, label="train")
        plt.xlabel("epoch")
        plt.ylabel("avg_loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / "loss_epoch.png"
        # plt.savefig(path)
        plt.close()

        if has_val:
            plt.figure()
            plt.plot(epochs, val_losses, label="val", color="tab:orange")
            plt.xlabel("epoch")
            plt.ylabel("val_loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            path = self.output_dir / "val_loss_epoch.png"
            # plt.savefig(path)
            plt.close()

        plt.figure()
        plt.plot(steps, losses, label="train")
        if has_val:
            plt.plot(steps, val_losses, label="val")
        plt.xlabel("steps")
        plt.ylabel("avg_loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = self.output_dir / "loss_steps.png"
        # plt.savefig(path)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, losses, label="train")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("avg_loss")
        axes[0].set_title("Train Loss / Epoch")
        axes[0].legend()
        axes[0].grid(True)

        if has_val:
            axes[1].plot(epochs, val_losses, label="val", color="tab:orange")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val_loss")
        axes[1].set_title("Val Loss / Epoch")
        if has_val:
            axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(steps, losses, label="train")
        if has_val:
            axes[2].plot(steps, val_losses, label="val")
        axes[2].set_xlabel("steps")
        axes[2].set_ylabel("loss")
        axes[2].set_title("Loss / Steps")
        axes[2].legend()
        axes[2].grid(True)

        fig.tight_layout()
        path = self.output_dir / "loss_summary.png"
        fig.savefig(path)
        plt.close(fig)

        # log.info(f"saved loss plot: {path}")
        
    def save_checkpoint(self, epoch, avg_loss, val_loss=None):

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_loss": val_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt, path)
        # log.info(f"saved checkpoint: {path}")
        
    def save_topk_checkpoint(self, epoch, metrics):
        score = metrics.get(self.monitor_key)

        if score is None:
            log.warning(f"monitor_key={self.monitor_key} is None, skip checkpoint")
            return

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = self.ckpt_dir / f"epoch_{epoch:03d}_{self.monitor_key}_{score:.4f}.pt"

        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "monitor_key": self.monitor_key,
            "monitor_score": score,
            "metrics": metrics,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        torch.save(ckpt, path)
        # log.info(f"saved checkpoint: {path}")

        self.best_checkpoints.append({
            "score": score,
            "path": path,
            "epoch": epoch,
        })

        self.best_checkpoints.sort(key=lambda item: item["score"])

        while len(self.best_checkpoints) > self.top_k:
            removed = self.best_checkpoints.pop(-1)
            removed_path = removed["path"]
            if removed_path.exists():
                removed_path.unlink()
                # log.info(f"removed checkpoint: {removed_path}")

    def format_summary(self, summary):
        lines = []
        lines.append("Training finished")
        lines.append(f"num_epochs: {summary['num_epochs']}")
        lines.append(f"global_step: {summary['global_step']}")
        lines.append(f"last_loss: {summary['last_loss']}")
        lines.append(f"last_val_loss: {summary['last_val_loss']}")
        lines.append(f"monitor_key: {summary['monitor_key']}")
        lines.append(f"output_dir: {summary['output_dir']}")
        lines.append(f"ckpt_dir: {summary['ckpt_dir']}")
        lines.append("best_checkpoints:")
    
        for item in summary["best_checkpoints"]:
            lines.append(
                f"  epoch={item['epoch']} score={item['score']:.6f} path={item['path']}"
            )
    
        return "\n".join(lines)