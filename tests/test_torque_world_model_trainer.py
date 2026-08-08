from types import SimpleNamespace

import torch

from train.trainer.torque_world_model_train import TorqueWorldModelTrainer


class _SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.valid_indices = list(range(len(samples)))
        self.raw_idx_to_episode_start = {index: 0 for index in self.valid_indices}
        self.dataset = SimpleNamespace(
            meta=SimpleNamespace(
                episodes=[
                    {
                        "dataset_from_index": 0,
                        "dataset_to_index": len(samples),
                    }
                ]
            )
        )
        self.normalizer = None

    @property
    def horizon(self):
        return 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _config():
    return {
        "dataloader": {
            "state_history_horizon": 4,
            "action_chunk_horizon": 5,
            "prediction_horizon": 3,
            "normalize_mode": None,
        },
        "model": {
            "joint_dim": 2,
            "action_dim": 7,
            "hidden_dim": 16,
            "num_layers": 1,
            "attention_heads": 4,
            "flow_layers": 1,
            "flow_attention_heads": 4,
            "flow_ffn_multiplier": 2,
            "dropout": 0.0,
            "state_estimator": {
                "sampling_dt": 0.1,
                "q_mean_window_samples": 1,
                "q_lowpass_cutoff_hz": None,
                "dq_lowpass_cutoff_hz": None,
                "ddq_lowpass_cutoff_hz": None,
            },
        },
        "contact_gate": {
            "enabled": True,
            "positive_class_weight": 1.0,
        },
        "loss": {
            "wrench_weight": 0.0,
            "standardize_derived_residuals": False,
        },
        "train": {
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "val_ratio": 0.0,
            "num_epochs": 1,
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        },
    }


def _sample(seed):
    generator = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(4, 2, generator=generator),
        "tau": torch.randn(4, 2, generator=generator),
        "target_relative_pose": torch.randn(5, 7, generator=generator),
        "target_relative_pose_mask": torch.ones(5),
        "q_future": torch.randn(3, 2, generator=generator),
        "tau_future": torch.randn(3, 2, generator=generator),
        "dq_future_raw": torch.randn(3, 2, generator=generator),
        "ddq_future_raw": torch.randn(3, 2, generator=generator),
        "contact_future": torch.tensor([[0.0], [1.0], [1.0]]),
    }


def test_synthetic_dataset_batch_runs_model_loss_and_optimizer_step():
    config = _config()
    trainer = TorqueWorldModelTrainer(config)
    trainer.dataset = _SyntheticDataset([_sample(1), _sample(2)])
    trainer.model = trainer.build_model()
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                trainer.dataset, batch_size=2, shuffle=False
            )
        )
    )

    loss, out = trainer.compute_loss(batch)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainer.optimizer.step()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert out["q_pred"].shape == (2, 3, 2)
    assert out["tau_pred"].shape == (2, 3, 2)
    assert out["contact_probability"].shape == (2, 3, 1)
    assert "loss_dict" in out
