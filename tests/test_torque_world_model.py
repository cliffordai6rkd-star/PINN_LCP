import pytest
import torch

from model.pinn_model.torque_world_model import TorqueWorldModel


def _config(*, history=5, future=3, action_horizon=None):
    data = {
        "state_history_horizon": history,
        "prediction_horizon": future,
    }
    if action_horizon is not None:
        data["action_condition_horizon"] = action_horizon
    return {
        "dataloader": data,
        "model": {
            "joint_dim": 7,
            "action_dim": 7,
            "hidden_dim": 24,
            "num_layers": 1,
            "attention_heads": 4,
            "flow_layers": 2,
            "flow_attention_heads": 4,
            "flow_ffn_multiplier": 2,
            "flow_inference_steps": 2,
            "flow_solver": "heun",
            "contact_logit_scale": 3.0,
            "dropout": 0.0,
        },
    }


def _condition_batch(*, batch_size=2, history=5, action_horizon=7):
    generator = torch.Generator().manual_seed(31)
    return {
        "q": torch.randn(batch_size, history, 7, generator=generator),
        "tau": torch.randn(batch_size, history, 7, generator=generator),
        "target_relative_pose": torch.randn(
            batch_size, action_horizon, 7, generator=generator
        ),
        "target_relative_pose_mask": torch.tensor(
            [[1] * (action_horizon - 2) + [0, 0]] * batch_size,
            dtype=torch.float32,
        ),
    }


def _training_batch(*, batch_size=2, history=5, action_horizon=7, future=3):
    batch = _condition_batch(
        batch_size=batch_size,
        history=history,
        action_horizon=action_horizon,
    )
    generator = torch.Generator().manual_seed(47)
    batch.update(
        {
            "q_future": torch.randn(
                batch_size, future, 7, generator=generator
            ),
            "tau_future": torch.randn(
                batch_size, future, 7, generator=generator
            ),
            "contact_future": torch.randint(
                0,
                2,
                (batch_size, future, 1),
                generator=generator,
            ).float(),
        }
    )
    return batch


def test_cfm_shapes_and_independent_history_action_future_horizons():
    model = TorqueWorldModel(_config(history=5, future=3)).eval()
    batch = _training_batch(history=5, action_horizon=7, future=3)
    output = model(batch, flow_time=0.25)

    assert output["flow_velocity_pred"].shape == (2, 3, 15)
    assert output["flow_velocity_target"].shape == (2, 3, 15)
    assert output["q_pred"].shape == (2, 3, 7)
    assert output["tau_pred"].shape == (2, 3, 7)
    assert output["contact_logits"].shape == (2, 3, 1)
    assert output["state_features"].shape == (2, 5, 24)
    assert output["action_features"].shape == (2, 7, 24)
    assert output["condition_memory"].shape == (2, 12, 24)
    assert output["state_action_attention_weights"].shape == (2, 4, 5, 7)

    target = torch.cat(
        (
            batch["q_future"],
            batch["tau_future"],
            (2.0 * batch["contact_future"] - 1.0) * 3.0,
        ),
        dim=-1,
    )
    source = torch.cat(
        (
            batch["q"][:, -3:],
            batch["tau"][:, -3:],
            torch.zeros(2, 3, 1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(output["flow_target_state"], target)
    torch.testing.assert_close(output["flow_source_state"], source)
    torch.testing.assert_close(output["flow_velocity_target"], target - source)
    torch.testing.assert_close(
        output["flow_interpolated"], 0.75 * source + 0.25 * target
    )


def test_state_tokens_query_action_tokens_and_both_encoders_get_gradients():
    model = TorqueWorldModel(_config()).train()
    batch = _training_batch()
    output = model(batch, flow_time=0.4)

    # Attention axes make the intended Q/K direction directly observable:
    # query length is state H=5, key/value length is action A=7.
    weights = output["state_action_attention_weights"]
    assert weights.shape[-2:] == (5, 7)
    torch.testing.assert_close(weights[..., -2:], torch.zeros_like(weights[..., -2:]))
    torch.testing.assert_close(
        output["condition_memory"][:, 5:], output["action_features"]
    )

    output["flow_velocity_pred"].square().mean().backward()
    for parameter in (
        model.state_encoder.weight_ih_l0,
        model.action_encoder.weight_ih_l0,
        model.state_queries_action.in_proj_weight,
        model.flow_blocks[0].condition_attention.in_proj_weight,
    ):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


def test_predict_uses_only_documented_condition_inputs():
    model = TorqueWorldModel(_config()).eval()
    batch = _condition_batch()

    baseline = model.predict(batch, solver="euler", steps=2)
    batch.update(
        {
            "dq": torch.randn(2, 5, 7),
            "ddq": torch.randn(2, 5, 7),
            "wrench": torch.randn(2, 5, 6),
            "contact_future": torch.randn(2, 3, 1),
        }
    )
    with_supervision_only_fields = model.predict(
        batch, solver="euler", steps=2
    )

    torch.testing.assert_close(
        baseline["flow_state_pred"],
        with_supervision_only_fields["flow_state_pred"],
    )
    assert TorqueWorldModel.CONDITION_KEYS == (
        "q",
        "tau",
        "target_relative_pose",
        "target_relative_pose_mask",
    )
    assert not any(
        token in name
        for name, _ in model.named_modules()
        for token in ("wrench", "contact_encoder", "dq", "ddq")
    )


def test_predict_is_deterministic_from_the_same_history_source():
    model = TorqueWorldModel(_config()).eval()
    batch = _condition_batch()

    first = model.predict(batch, steps=2)
    repeated = model.predict(batch, steps=2)

    torch.testing.assert_close(first["flow_state_pred"], repeated["flow_state_pred"])
    expected_source = torch.cat(
        (batch["q"][:, -3:], batch["tau"][:, -3:], torch.zeros(2, 3, 1)),
        dim=-1,
    )
    torch.testing.assert_close(first["flow_source_state"], expected_source)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"hidden_dim": 22}, "divisible"),
        ({"flow_solver": "rk4"}, "euler"),
        ({"contact_logit_scale": 0.0}, "positive"),
    ],
)
def test_invalid_model_config_is_rejected(change, message):
    config = _config()
    config["model"].update(change)
    with pytest.raises(ValueError, match=message):
        TorqueWorldModel(config)


def test_configured_action_horizon_is_validated_without_coupling_to_future():
    model = TorqueWorldModel(_config(action_horizon=7)).eval()
    model.predict(_condition_batch(action_horizon=7))

    with pytest.raises(ValueError, match="horizon 7"):
        model.predict(
            _condition_batch(action_horizon=6),
        )


def test_history_and_prediction_horizons_are_independent_in_both_directions():
    model = TorqueWorldModel(_config(history=2, future=3)).eval()
    batch = _condition_batch(history=2)
    output = model.predict(batch, steps=1, solver="euler")

    assert output["flow_source_state"].shape == (2, 3, 15)
    torch.testing.assert_close(
        output["flow_source_state"][:, 0, :7], batch["q"][:, 0]
    )
    torch.testing.assert_close(
        output["flow_source_state"][:, 1, :7], batch["q"][:, 0]
    )
    torch.testing.assert_close(
        output["flow_source_state"][:, -1, :7], batch["q"][:, -1]
    )
