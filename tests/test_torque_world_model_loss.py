import torch

from model.pinn_model.torque_world_model import TorqueWorldModel
from train.torque_world_model_loss import TorqueWorldModelLoss


def _config(*, physics=False):
    return {
        "dataloader": {
            "history_horizon": 4,
            "future_horizon": 3,
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
            "flow_weight": 1.0,
            "q_weight": 1.0,
            "tau_weight": 1.0,
            "dq_weight": 0.2,
            "ddq_weight": 0.1,
            "contact_weight": 1.0,
            "wrench_weight": 1.0 if physics else 0.0,
            "standardize_derived_residuals": False,
            "standardize_wrench_residual": False,
        },
        "physics": {
            "tau_f_checkpoint_path": "unused-test-checkpoint.pt",
            "wrench_damping": 0.1,
            "soft_contact_gate": True,
        },
    }


def _batch():
    generator = torch.Generator().manual_seed(3)
    return {
        "q": torch.randn(2, 4, 2, generator=generator),
        "tau": torch.randn(2, 4, 2, generator=generator),
        "target_relative_pose": torch.randn(2, 5, 7, generator=generator),
        "target_relative_pose_mask": torch.ones(2, 5),
        "q_future": torch.randn(2, 3, 2, generator=generator),
        "tau_future": torch.randn(2, 3, 2, generator=generator),
        "dq_future_raw": torch.randn(2, 3, 2, generator=generator),
        "ddq_future_raw": torch.randn(2, 3, 2, generator=generator),
        "contact_future": torch.randint(
            0, 2, (2, 3, 1), generator=generator
        ).float(),
    }


def test_data_only_loss_backpropagates_through_flow_q_tau_and_contact():
    config = _config()
    model = TorqueWorldModel(config).train()
    calculator = TorqueWorldModelLoss(config)
    batch = _batch()

    out = model(batch, flow_time=0.4)
    loss, metrics = calculator(out, batch)
    loss.backward()

    assert loss.ndim == 0
    assert set(("flow_loss", "q_loss", "tau_loss", "dq_loss", "ddq_loss", "contact_loss")) <= set(metrics)
    assert model.state_encoder.weight_ih_l0.grad is not None
    assert model.action_encoder.weight_ih_l0.grad is not None
    assert model.flow_output[-1].weight.grad is not None


def test_configured_dt_drives_q_derived_velocity_and_acceleration():
    calculator = TorqueWorldModelLoss(_config())
    q_history = torch.tensor(
        [[[-0.3, -0.3], [-0.2, -0.2], [-0.1, -0.1], [0.0, 0.0]]]
    )
    q_future = torch.tensor(
        [[[0.1, 0.1], [0.3, 0.3], [0.6, 0.6]]]
    )
    out = {"q_pred": q_future}
    batch = {"q": q_history}

    _, _, dq, ddq = calculator._derived_state(out, batch)

    torch.testing.assert_close(
        dq,
        torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        ddq,
        torch.tensor([[[0.0, 0.0], [10.0, 10.0], [10.0, 10.0]]]),
        atol=1e-4,
        rtol=0.0,
    )


class _FakeTauF:
    def __call__(self, history, future):
        del history
        return 0.05 * (
            future["q"] + future["dq"] + future["ddq"] + future["tau"]
        )


def test_local_rnea_wrench_loss_reaches_q_tau_and_contact():
    config = _config(physics=True)
    calculator = TorqueWorldModelLoss(config)
    calculator.tau_f_predictor = _FakeTauF()
    batch = _batch()
    batch_size, horizon, joints = batch["q_future"].shape
    identity = torch.eye(joints).expand(batch_size, horizon, joints, joints)
    batch.update(
        {
            "q_future_raw": batch["q_future"].clone(),
            "rnea_tau_id_future": torch.zeros(batch_size, horizon, joints),
            "rnea_d_tau_d_q_future": 0.2 * identity,
            "rnea_d_tau_d_dq_future": 0.1 * identity,
            "rnea_d_tau_d_ddq_future": 0.5 * identity,
            "frame_jacobian_future": identity.clone(),
            "wrench_future_raw": torch.zeros(batch_size, horizon, joints),
        }
    )
    q_pred = batch["q_future"].clone().requires_grad_()
    tau_pred = batch["tau_future"].clone().requires_grad_()
    contact_logits = torch.zeros(batch_size, horizon, 1, requires_grad=True)
    flow_velocity = torch.zeros(
        batch_size, horizon, 2 * joints + 1, requires_grad=True
    )
    out = {
        "q_pred": q_pred,
        "tau_pred": tau_pred,
        "contact_logits": contact_logits,
        "contact_probability": torch.sigmoid(contact_logits),
        "flow_velocity_pred": flow_velocity,
        "flow_velocity_target": torch.ones_like(flow_velocity),
    }

    loss, metrics = calculator(out, batch)
    loss.backward()

    assert metrics["wrench_loss"] > 0
    assert q_pred.grad is not None and q_pred.grad.abs().sum() > 0
    assert tau_pred.grad is not None and tau_pred.grad.abs().sum() > 0
    assert contact_logits.grad is not None and contact_logits.grad.abs().sum() > 0
    assert "wrench_pred" in out and "tau_f_pred" in out
