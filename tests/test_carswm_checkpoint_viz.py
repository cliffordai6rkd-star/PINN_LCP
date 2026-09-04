import json

import pytest
import torch

from model.pinn_model.contact_world_model import ContactWorldModel
from train.trainer.contact_world_model_train import ContactWorldModelTrainer


def _config(tmp_path):
    return {
        "dataloader": {
            "state_history_horizon": 3,
            "prediction_horizon": 4,
            "action_condition_horizon": 2,
            "high_fps": 100,
            "normalize_mode": None,
        },
        "model": {
            "inputs": ["q", "dq", "delta_q", "tau"],
            "joint_dim": 2,
            "action_dim": 2,
            "contact_state_count": 3,
            "hidden_dim": 8,
            "state_layers": 1,
            "action_layers": 1,
            "flow_layers": 1,
            "flow_attention_heads": 2,
            "flow_ffn_multiplier": 2,
            "flow_inference_steps": 1,
            "flow_solver": "euler",
            "dropout": 0.0,
        },
        "loss": {"dt": 0.01},
        "train": {
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "val_ratio": 0.0,
            "output_dir": str(tmp_path),
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
            "checkpoint_visualization": {
                "enabled": True,
                "num_samples": 4,
                "seed": 2027,
                "denormalize_for_plot": False,
                "wrist_joint_index": 1,
                "flow_steps": 1,
                "flow_solver": "euler",
            },
        },
    }


def _batch():
    value = {
        key: torch.randn(1, 3, 2)
        for key in ("q", "dq", "delta_q", "tau")
    }
    value.update(
        {
            f"{key}_future": torch.randn(1, 4, 2)
            for key in ("q", "dq", "delta_q", "tau")
        }
    )
    value.update(
        action=torch.randn(1, 2, 2),
        action_mask=torch.ones(1, 2),
        contact_future=torch.tensor([[[0.0], [1.0], [2.0], [2.0]]]),
    )
    return value


def test_checkpoint_visualization_is_fixed_and_writes_six_panel_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    config = _config(tmp_path)
    trainer = ContactWorldModelTrainer(config)
    trainer.model = ContactWorldModel(config).eval()
    trainer.ema = None
    trainer._checkpoint_visualization_records = [
        {"name": "pre_contact", "index": 7, "raw_index": 11, "batch": _batch()}
    ]
    generator = torch.Generator(device="cpu").manual_seed(2027)
    trainer._checkpoint_visualization_noise = {
        "pre_contact": torch.randn(1, 4, 4, 8, generator=generator)
    }

    trainer.global_step = 10
    trainer._save_checkpoint_visualization(0, tmp_path / "step_00000010.pt")
    trainer.global_step = 20
    trainer._save_checkpoint_visualization(1, tmp_path / "step_00000020.pt")

    output = tmp_path / "checkpoint_viz"
    first = json.loads((output / "step_00000010_metrics.json").read_text())
    second = json.loads((output / "step_00000020_metrics.json").read_text())
    assert first["visualization_seed"] == second["visualization_seed"] == 2027
    assert first["energy_score"] == pytest.approx(second["energy_score"])
    assert first["q_energy_score"] == pytest.approx(second["q_energy_score"])
    assert first["tau_energy_score"] == pytest.approx(second["tau_energy_score"])
    for name in (
        "step_00000010_summary.png",
        "step_00000020_summary.png",
        "latest_summary.png",
        "plot_scales.json",
    ):
        assert (output / name).stat().st_size > 0
