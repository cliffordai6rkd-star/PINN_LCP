import pytest
import torch

from model.pinn_model.contact_world_model import (
    ContactWorldModel,
    PREDICTED_STATE_STREAMS,
    SUPPORTED_STATE_STREAMS,
)
from train.contact_world_model_loss import ContactWorldModelLoss


def config(inputs=None, outputs=None):
    result = {
        "dataloader": {"state_history_horizon": 5, "prediction_horizon": 4, "action_condition_horizon": 3, "high_fps": 100, "normalize_mode": None},
        "model": {"inputs": list(inputs or SUPPORTED_STATE_STREAMS), "joint_dim": 2, "action_dim": 2, "contact_state_count": 3, "hidden_dim": 8, "state_layers": 1, "action_layers": 1, "flow_layers": 1, "flow_attention_heads": 2, "flow_ffn_multiplier": 2, "flow_inference_steps": 2, "flow_solver": "heun", "flow_source_mode": "gaussian", "dropout": 0.0},
        "loss": {"dt": 0.01, "kinematic_consistency_weight": 0.01, "ddq_smoothness_weight": 0.01},
    }
    if outputs is not None:
        result["model"]["outputs"] = list(outputs)
    return result


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
    assert output["state_tokens"].shape == (2, count, 8)
    assert output["state_action_tokens"].shape == (2, count, 8)
    assert output["condition_memory"].shape == (2, count + 3, 8)
    assert output["action_tokens"].shape == (2, 3, 8)
    torch.testing.assert_close(
        output["condition_memory"],
        torch.cat((output["state_action_tokens"], output["action_tokens"]), dim=1),
    )
    assert hasattr(model, "state_to_action_attention")
    assert output["action_time"].shape == (2, 3)
    assert output["future_time"].shape == (2, 4)


def test_flow_block_uses_self_attention_cross_attention_and_ffn():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)
    output["flow_velocity_pred"].square().mean().backward()
    block = model.flow_blocks[0]
    assert any(parameter.grad is not None for parameter in block.self_attention.parameters())
    assert any(parameter.grad is not None for parameter in block.condition_attention.parameters())
    assert any(parameter.grad is not None for parameter in block.ffn.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.state_to_action_attention.parameters()
    )
    assert any(
        parameter.grad is not None
        for encoder in model.state_encoders.values()
        for parameter in encoder.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.action_encoder.parameters())


def test_attention_state_pooling_reads_full_history():
    cfg = config()
    cfg["model"]["state_pooling"] = "attention"
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)

    assert model.state_pooling == "attention"
    assert hasattr(model, "state_pool_attention")
    assert tuple(model.state_pool_queries) == tuple(cfg["model"]["inputs"])
    assert output["state_tokens"].shape == (2, 4, 8)

    output["flow_velocity_pred"].square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.state_pool_attention.parameters()
    )
    assert model.checkpoint_contract()["architecture"]["state_token"] == (
        "attention_pooling"
    )


def test_checkpoint_contract_identifies_simplified_token_architecture():
    model = ContactWorldModel(config())
    contract = model.checkpoint_contract()
    assert model.MODEL_VERSION == "carswm_v3"
    assert contract["schema_version"] == 3
    assert contract["architecture"] == {
        "condition_encoder": "modality_gru_action_gru_state_to_action_cross_attention",
        "state_token": "final_gru_hidden",
        "flow_decoder": "self_attention_cross_attention_ffn",
        "condition_memory": "state_action_aware_state_plus_raw_action",
        "action_time_encoding": "physical_seconds_fourier_mlp",
        "future_time_encoding": "physical_seconds_fourier_mlp",
    }
    assert contract["action"]["dataset_alignment"] == "previous"
    incompatible = dict(contract)
    incompatible["model_version"] = "carswm_v1"
    with pytest.raises(ValueError, match="contract mismatch"):
        model.validate_checkpoint_contract(incompatible)


def test_flow_targets_and_contact_are_separate():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape == (2, 4, 8)
    assert output["flow_target_state"].shape[-1] == 8
    assert output["flow_source_state"].shape[-1] == 8
    assert output["contact_logits"].shape == (2, 4, 3)
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    loss.backward()
    assert "flow_contact_loss" not in metrics
    assert "contact_loss" in metrics
    assert any(parameter.grad is not None for parameter in model.contact_head.parameters())


def test_physical_time_changes_action_tokens_and_contact_alignment():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    first = model.encode_conditions(values)
    values["action_time"] = torch.tensor(
        [[0.10, 0.20, 0.30], [0.10, 0.20, 0.30]]
    )
    shifted = model.encode_conditions(values)
    assert not torch.equal(first["action_tokens"], shifted["action_tokens"])
    assert not torch.equal(first["action_time"], shifted["action_time"])
    assert shifted["future_time"].shape == (2, 4)


def test_physical_time_prefers_recorded_irregular_timestamps():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    values["history_timestamp_ns"] = torch.tensor(
        [
            [100_000_000, 110_000_000, 120_000_000, 130_000_000, 140_000_000],
            [200_000_000, 210_000_000, 220_000_000, 230_000_000, 240_000_000],
        ],
        dtype=torch.int64,
    )
    values["action_chunk_timestamp_ns"] = torch.tensor(
        [
            [181_000_000, 223_000_000, 281_000_000],
            [281_000_000, 323_000_000, 381_000_000],
        ],
        dtype=torch.int64,
    )
    values["future_timestamp_ns"] = torch.tensor(
        [
            [151_000_000, 162_000_000, 177_000_000, 201_000_000],
            [251_000_000, 262_000_000, 277_000_000, 301_000_000],
        ],
        dtype=torch.int64,
    )
    encoded = model.encode_conditions(values)
    torch.testing.assert_close(
        encoded["action_time"],
        torch.tensor([[0.041, 0.083, 0.141], [0.041, 0.083, 0.141]]),
    )
    torch.testing.assert_close(
        encoded["future_time"],
        torch.tensor([[0.011, 0.022, 0.037, 0.061], [0.011, 0.022, 0.037, 0.061]]),
    )


def test_zoh_action_alignment_uses_physical_seconds():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    encoded = {
        "action_tokens": torch.tensor([[[10.0], [20.0], [30.0]]]),
        "action_time": torch.tensor([[0.04, 0.08, 0.14]]),
        "future_time": torch.tensor([[0.01, 0.04, 0.05, 0.10]]),
    }
    aligned = model._time_aligned_action_features(encoded)
    torch.testing.assert_close(
        aligned,
        torch.tensor([[[10.0], [10.0], [10.0], [20.0]]]),
    )


def test_contact_head_depends_on_its_generated_continuous_sample():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    encoded = model.encode_conditions(values)
    continuous = torch.randn(2, 4, model.flow_dim, requires_grad=True)
    model.contact_logits(continuous, encoded).square().mean().backward()
    assert continuous.grad is not None
    assert torch.any(continuous.grad != 0)


def test_contact_head_uses_configured_phase_count():
    cfg = config()
    cfg["model"]["contact_state_count"] = 4
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values["contact_future"] = torch.randint(0, 4, values["contact_future"].shape).float()
    output = model(values, flow_time=0.5)
    assert output["contact_logits"].shape == (2, 4, 4)
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


def test_contact_target_outside_configured_phase_count_is_rejected():
    cfg = config()
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values["contact_future"][0, 0, 0] = 3.0
    output = model(values, flow_time=0.5)
    with pytest.raises(ValueError, match="outside model.contact_state_count"):
        ContactWorldModelLoss(cfg)(output, values)


def test_different_source_noise_generates_different_futures():
    cfg = config()
    model = ContactWorldModel(cfg).eval()
    values = batch(cfg)
    first = model.predict(values, source_noise=torch.zeros(2, 4, 8))
    second = model.predict(values, source_noise=torch.ones(2, 4, 8))
    assert not torch.equal(first["flow_state_pred"], second["flow_state_pred"])
    assert first["contact_logits"].shape == (2, 4, 3)


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
    assert model.flow_dim == 3 * 2
    assert output["flow_velocity_pred"].shape == (2, 4, 6)
    assert all(f"{key}_pred" in output for key in ("q", "delta_q", "tau"))
    assert "dq_pred" not in output
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
    assert "dq_loss" not in metrics


def test_teacher_inputs_and_outputs_are_independent():
    cfg = config(
        inputs=["q", "dq", "delta_q", "tau"],
        outputs=["q", "tau"],
    )
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("dq_future")
    values.pop("delta_q_future")

    output = model(values, flow_time=0.5)

    assert tuple(model.state_encoders) == ("q", "dq", "delta_q", "tau")
    assert model.outputs == ("q", "tau")
    assert model.predicted_state_streams == ("q", "tau")
    assert model.PREDICTED_STATE_STREAMS == ("q", "tau")
    assert model.CONDITION_KEYS == (
        "q", "dq", "delta_q", "tau", "action", "action_mask",
        "action_time", "future_time"
    )
    assert model.TARGET_KEYS == ("q_future", "tau_future", "contact_future")
    assert model.flow_dim == 2 * model.joint_dim
    assert output["flow_velocity_pred"].shape == (2, 4, 4)
    assert "q_pred" in output and "tau_pred" in output
    assert "dq_pred" not in output and "delta_q_pred" not in output
    loss, metrics = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)
    assert "q_loss" in metrics and "tau_loss" in metrics
    assert "dq_loss" not in metrics and "delta_q_loss" not in metrics

    contract = model.checkpoint_contract()
    assert contract["input_state_streams"] == ["q", "dq", "delta_q", "tau"]
    assert contract["predicted_continuous_streams"] == ["q", "tau"]


def test_teacher_can_predict_a_stream_not_used_as_history():
    cfg = config(inputs=["q", "dq"], outputs=["q", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    output = model(values, flow_time=0.5)

    assert tuple(model.state_encoders) == ("q", "dq")
    assert model.predicted_state_streams == ("q", "tau")
    assert "tau" not in values
    assert output["tau_pred"].shape == (2, 4, 2)
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "outputs,match",
    [
        ([], "at least one"),
        (["q", "q"], "duplicates"),
        (["q", "temperature"], "unsupported"),
    ],
)
def test_invalid_teacher_outputs_are_rejected(outputs, match):
    cfg = config(outputs=outputs)
    with pytest.raises(ValueError, match=match):
        ContactWorldModel(cfg)
    with pytest.raises(ValueError, match=match):
        ContactWorldModelLoss(cfg)


def test_null_outputs_fall_back_to_inputs():
    cfg = config(inputs=["q", "tau"])
    cfg["model"]["outputs"] = None
    model = ContactWorldModel(cfg)
    calculator = ContactWorldModelLoss(cfg)

    assert model.outputs == ("q", "tau")
    assert calculator.predicted_state_streams == ("q", "tau")


def test_disabled_cross_stream_regularizers_allow_reduced_outputs():
    cfg = config(inputs=["q", "dq", "delta_q", "tau"], outputs=["tau"])
    cfg["loss"]["kinematic_consistency_weight"] = 0.0
    cfg["loss"]["ddq_smoothness_weight"] = 0.0
    values = batch(cfg)
    values.pop("q_future")
    values.pop("dq_future")
    values.pop("delta_q_future")
    model = ContactWorldModel(cfg)

    loss, _ = ContactWorldModelLoss(cfg)(model(values, flow_time=0.5), values)

    assert torch.isfinite(loss)


def test_contract_allows_ablation_without_q():
    cfg = config(["dq", "tau"])
    model = ContactWorldModel(cfg)
    values = batch(cfg)
    values.pop("q", None)
    values.pop("q_future", None)
    output = model(values, flow_time=0.5)
    assert output["flow_velocity_pred"].shape[-1] == 2 * 2
    loss, _ = ContactWorldModelLoss(cfg)(output, values)
    assert torch.isfinite(loss)


def test_endpoint_schedule_and_delta_q_contract():
    cfg = config()
    calculator = ContactWorldModelLoss(cfg)
    calculator.set_global_step(0, 100)
    assert calculator.endpoint_weight == pytest.approx(0.1)
    calculator.set_global_step(15, 100)
    assert calculator.endpoint_weight == pytest.approx(0.05)
    calculator.set_global_step(30, 100)
    assert calculator.endpoint_weight == pytest.approx(0.0)
    assert calculator.delta_q_consistency_weight == 0.0


def test_kinematic_loss_is_zero_for_trapezoidal_integration():
    cfg = config()
    calculator = ContactWorldModelLoss(cfg)
    values = batch(cfg)
    values["q"][:] = 0.0
    values["dq"][:] = 1.0
    q = torch.arange(1, 5, dtype=torch.float32)[None, :, None].repeat(2, 1, 2) * 0.01
    out = {"q_pred": q, "dq_pred": torch.ones_like(q)}
    assert torch.max(calculator._kinematic_consistency(out, values)) < 1.0e-8
