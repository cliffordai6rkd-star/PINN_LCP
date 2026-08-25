from types import SimpleNamespace

import torch

from train.base_trainer import BaseTrainer


class _Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = torch.arange(3, dtype=torch.float32)[:, None]
        self.normalizer = SimpleNamespace(stats={}, eps=1.0e-6)
        self.filter_config = {}
        self.sample_rate_hz = 100.0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"x": self.values[index]}


class _Trainer(BaseTrainer):
    def build_dataset(self):
        return _Dataset()

    def build_model(self):
        return torch.nn.Linear(1, 1)

    def compute_loss(self, batch):
        prediction = self.model(batch["x"])
        loss = prediction.square().mean()
        return loss, {"loss_dict": {"total_loss": loss.detach()}}


def test_step_training_retains_latest_topk_and_latest_file(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 100,
            "max_train_steps": 5,
            "checkpoint_every_steps": 2,
            "top_k": 2,
            "save_latest_checkpoint": True,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 5, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.save_loss_plot = lambda: None
    summary = trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.glob("step_*.pt")) == [
        "step_00000004.pt",
        "step_00000005.pt",
    ]
    latest = torch.load(
        checkpoint_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert latest["global_step"] == 5
    assert summary["max_train_steps"] == 5
    assert summary["checkpoint_every_steps"] == 2
    assert [item["global_step"] for item in summary["best_checkpoints"]] == [4, 5]
    assert trainer.scheduler.last_epoch == 5


def test_step_training_defaults_null_top_k(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 100,
            "max_train_steps": 2,
            "checkpoint_every_steps": 2,
            "top_k": None,
            "save_latest_checkpoint": True,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 2, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)

    assert trainer.top_k == 3


def test_epoch_training_retains_latest_scheduled_topk(tmp_path):
    config = {
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "num_epochs": 6,
            "checkpoint_every_epochs": 2,
            "top_k": 3,
            "save_latest_checkpoint": False,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "scheduler": {"name": "cosine", "T_max": 6, "eta_min": 0.0},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        }
    }
    trainer = _Trainer(config)
    trainer.save_loss_plot = lambda: None
    summary = trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert sorted(path.name for path in checkpoint_dir.glob("epoch_*.pt")) == [
        "epoch_0000002.pt",
        "epoch_0000004.pt",
        "epoch_0000006.pt",
    ]
    assert [item["epoch"] for item in summary["best_checkpoints"]] == [2, 4, 6]
    assert summary["checkpoint_every_epochs"] == 2
    assert not (checkpoint_dir / "latest.pt").exists()
