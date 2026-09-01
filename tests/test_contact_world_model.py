import pytest
import torch

from model.pinn_model.contact_world_model import (
    ContactWorldModel,
    PREDICTED_STATE_STREAMS,
    SUPPORTED_STATE_STREAMS,
)
from train.contact_world_model_loss import ContactWorldModelLoss


def config(inputs=None):
    return {
        "dataloader": {"state_history_horizon": 5, "prediction_horizon": 4, "action_condition_horizon": 3, "high_fps": 100, "normalize_mode": None},
        "model": {"inputs": list(inputs or SUPPORTED_STATE_STREAMS), "joint_dim": 2, "action_dim": 2, "contact_state_count": 3, "hidden_dim": 8, "state_layers": 1, "action_layers": 1, "attention_heads": 2, "flow_layers": 1, "flow_attention_heads": 2, "flow_ffn_multiplier": 2, "flow_inference_steps": 2, "flow_solver": "heun", "flow_source_mode": "gaussian", "dropout": 0.0},
        "loss": {"dt": 0.01, "kinematic_consistency_weight": 0.01, "ddq_smoothness_weight": 0.01},
    }


def batch(cfg):
    b, h, f, d, a = 2, 5, 4, 2, 3
    out = {key: torch.randn(b, h, d) for key in cfg["model"]["inputs"]}
    out.update({f"{key}_future": torch.randn(b, f, d) for key in PREDICTED_STATE_STREAMS})
    out.update(action=torch.randn(b, a, d), action_mask=torch.ones(b, a, dtype=torch.bool), contact_future=torch.randint(0, 3, (b, f, 1)).float())
    return out


@pytest.mark.parametrize(
    "inputs,count",
    [
        (["q", "dq", "delta_q", "tau"], 4),
        (["q", "delta_q", "tau"], 3),
        (["q", "tau"], 2),
    ],
)
def test_independent_state_encoders_and_fused_shapes(inputs, count):
    cfg = config(inputs)
    model = ContactWorldModel(cfg)
    assert len(model.state_encoders) == count
    assert tuple(model.state_encoders) == tuple(inputs)
    assert model.state_encoders[inputs[0]] is not model.state_encoders[inputs[-1]]
    first = {id(parameter) for parameter in model.state_encoders[inputs[0]].parameters()}
    assert first.isdisjoint(id(parameter) for parameter in model.state_encoders[inputs[-1]].parameters())
    output = model(batch(cfg), flow_time=0.5)
    assert output["state_features"].shape == (2, count, 8)
    assert output["state_tokens"].shape == (2, count, 8)
    assert output["condition_memory"].shape == (2, count + 3, 8)
    assert output["action_gates"].shape == (2, count, 1)
    assert torch.all((output["action_gates"] >= 0) & (output["action_gates"] <= 1))
    assert model.state_to_action_attention.embed_dim == 8


def test_flow_targets_and_contact_are_separate():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape == (2, 4, 11)
    assert output["contact_logits"].shape == (2, 4, 3)
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    loss.backward()
    assert "flow_contact_loss" not in metrics
    assert "contact_loss" in metrics


def test_missing_selected_state_is_rejected_without_zero_fill():
    cfg = config(["q", "tau"])
    values = batch(cfg)
    del values["tau"]
    with pytest.raises(KeyError, match="tau"):
        ContactWorldModel(cfg)(values)


def test_selected_streams_define_continuous_output_contract():
    cfg = config(["q", "delta_q", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("dq", None)
    values.pop("dq_future", None)
    output = model(values, flow_time=0.5)
    assert model.predicted_state_streams == ("q", "delta_q", "tau")
    assert model.PREDICTED_STATE_STREAMS == model.predicted_state_streams
    assert model.TARGET_KEYS == ("q_future", "delta_q_future", "tau_future", "contact_future")
    assert model.flow_dim == 3 * 2 + 3
    assert output["flow_velocity_pred"].shape == (2, 4, 9)
    assert all(f"{key}_pred" in output for key in ("q", "delta_q", "tau"))
    assert "dq_pred" not in output
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
    assert "dq_loss" not in metrics


def test_contract_allows_ablation_without_q():
    cfg = config(["dq", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("q", None)
    values.pop("q_future", None)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape[-1] == 2 * 2 + 3
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
