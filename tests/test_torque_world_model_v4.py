from types import SimpleNamespace
from unittest.mock import patch

import torch

from data_process.world_model_dataset import TorqueWorldModelDataset
from model.pinn_model.contact_gate import (
    ContactGateConfig,
    batched_hysteresis_three_phase_mask,
    hysteresis_three_phase_mask,
)
from model.pinn_model.torque_world_model import TorqueWorldModel
from train.torque_world_model_loss import TorqueWorldModelLoss
from train.trainer.torque_world_model_opd_train import TorqueWorldModelOPDTrainer


def _config(inputs=None):
    return {
        "dataloader": {
            "state_history_horizon": 5,
            "prediction_horizon": 4,
            "action_condition_horizon": 3,
            "high_fps": 100,
        },
        "model": {
            "inputs": inputs or ["q", "dq", "delta_q", "tau"],
            "joint_dim": 2,
            "action_dim": 2,
            "contact_state_count": 3,
            "hidden_dim": 16,
            "state_layers": 1,
            "action_layers": 1,
            "attention_heads": 4,
            "flow_layers": 1,
            "flow_attention_heads": 4,
            "flow_inference_steps": 2,
            "dropout": 0.0,
        },
        "loss": {
            "dt": 0.01,
            "kinematic_consistency_weight": 0.05,
            "ddq_smoothness_weight": 0.001,
        },
    }


def _batch(config):
    generator = torch.Generator().manual_seed(7)
    history = config["dataloader"]["state_history_horizon"]
    future = config["dataloader"]["prediction_horizon"]
    action_horizon = config["dataloader"]["action_condition_horizon"]
    batch = {
        key: torch.randn(2, history, 2, generator=generator)
        for key in ("q", "dq", "delta_q", "tau")
    }
    batch.update(
        {
            f"{key}_future": torch.randn(2, future, 2, generator=generator)
            for key in ("q", "dq", "delta_q", "tau")
        }
    )
    batch["action"] = torch.randn(2, action_horizon, 2, generator=generator)
    batch["action_mask"] = torch.ones(2, action_horizon, dtype=torch.bool)
    batch["contact"] = torch.zeros(2, history, 1)
    batch["contact_future"] = torch.randint(
        0, 3, (2, future, 1), generator=generator
    ).float()
    return batch


def test_configured_inputs_condition_history_but_all_four_streams_are_predicted():
    config = _config(inputs=["q", "tau"])
    model = TorqueWorldModel(config).train()
    batch = _batch(config)

    output = model(batch, flow_time=0.4)
    loss, metrics = TorqueWorldModelLoss(config)(output, batch)
    loss.backward()

    assert model.inputs == ("q", "tau")
    assert model.state_input_dim == 4
    assert output["flow_state_pred"].shape == (2, 4, 11)
    for key in ("q", "dq", "delta_q", "tau"):
        assert output[f"{key}_pred"].shape == (2, 4, 2)
        assert f"{key}_loss" in metrics
    assert "flow_contact_loss" not in metrics
    assert model.state_encoder.weight_ih_l0.grad is not None


def test_flow_source_is_injected_gaussian_noise_and_reusable_for_opd():
    config = _config()
    model = TorqueWorldModel(config).eval()
    batch = _batch(config)
    source_noise = torch.randn(
        2, config["dataloader"]["prediction_horizon"], model.flow_dim
    )

    first = model.predict(batch, steps=1, source_noise=source_noise)
    second = model.predict(batch, steps=1, source_noise=source_noise)

    torch.testing.assert_close(first["flow_source_state"], source_noise)
    torch.testing.assert_close(first["flow_source_noise"], source_noise)
    torch.testing.assert_close(first["flow_state_pred"], second["flow_state_pred"])
    with torch.no_grad():
        different = model.predict(
            batch,
            steps=1,
            source_noise=torch.randn_like(source_noise),
        )
    assert not torch.allclose(first["flow_source_state"], different["flow_source_state"])


def test_delta_q_is_directly_supervised_and_not_reconstructed_from_q():
    config = _config()
    calculator = TorqueWorldModelLoss(config)
    batch = _batch(config)
    output = {
        f"{key}_pred": batch[f"{key}_future"].clone()
        for key in ("q", "dq", "delta_q", "tau")
    }

    direct = calculator._direct_losses(output, batch)
    assert direct["delta_q"].item() == 0.0
    output["q_pred"] = output["q_pred"] + 100.0
    direct = calculator._direct_losses(output, batch)
    assert direct["delta_q"].item() == 0.0
    assert direct["q"].item() > 0.0


def test_velocity_smoothness_penalizes_jerk_not_constant_acceleration():
    config = _config()
    calculator = TorqueWorldModelLoss(config)
    # Linear dq has constant ddq and therefore zero jerk.
    linear_dq = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    oscillating_dq = torch.tensor([0.0, 1.0, -1.0, 2.0]).reshape(1, 4, 1)

    assert calculator._ddq_smoothness({"dq_pred": linear_dq}).item() == 0.0
    assert calculator._ddq_smoothness({"dq_pred": oscillating_dq}).item() > 0.0


def test_batched_three_phase_hysteresis_matches_independent_rows():
    generator = torch.Generator().manual_seed(23)
    signal = torch.rand(9, 37, generator=generator) * 2.0
    signal[:, ::7] = 0.2
    signal[:, 3::11] = 1.6

    for consecutive_frames in (1, 2, 5):
        for backfill in (False, True):
            expected = torch.stack(
                [
                    hysteresis_three_phase_mask(
                        row,
                        off_threshold=0.3,
                        on_threshold=1.5,
                        consecutive_frames=consecutive_frames,
                        backfill=backfill,
                    )
                    for row in signal
                ]
            )
            actual = batched_hysteresis_three_phase_mask(
                signal,
                off_threshold=0.3,
                on_threshold=1.5,
                consecutive_frames=consecutive_frames,
                backfill=backfill,
            )
            torch.testing.assert_close(actual, expected)
            assert actual.device == signal.device


def test_opd_rollout_contact_uses_tau_free_and_backpropagates_physical_term():
    config = _config()
    config["contact_gate"] = {
        "enabled": True,
        "label_mode": "three_phase",
        "metric": "tau_ext_l2",
        "thresholds": {"tau_ext_l2": {"off": 0.2, "on": 0.6}},
        "consecutive_frames": 2,
    }
    trainer = TorqueWorldModelOPDTrainer.__new__(TorqueWorldModelOPDTrainer)
    trainer.model = TorqueWorldModel(config)
    trainer.loss_calculator = TorqueWorldModelLoss(config)
    trainer.rollout_contact_enabled = True
    trainer.rollout_contact_weight = 0.2
    trainer.rollout_contact_physical_weight = 0.02
    trainer.rollout_contact_horizon = 4
    trainer.rollout_contact_backfill = False
    trainer.rollout_contact_temperature = 0.05
    trainer.rollout_contact_gate = ContactGateConfig.from_config(config)

    class _FakeTauFree:
        active_inputs = ("q", "dq", "delta_q")
        history_horizon = 5

        def __call__(self, history, future):
            del history
            return 0.1 * future["q"]

    trainer.tau_free_predictor = _FakeTauFree()
    batch = _batch(config)
    student_out = {
        f"{key}_pred": batch[f"{key}_future"].clone().requires_grad_()
        for key in ("q", "dq", "delta_q", "tau")
    }
    student_out["contact_logits"] = torch.randn(2, 4, 3, requires_grad=True)

    loss, metrics = trainer._rollout_contact_loss(batch, student_out)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["rollout_contact_ce"])
    assert student_out["tau_pred"].grad is not None
    assert student_out["q_pred"].grad is not None
    assert student_out["contact_logits"].grad is not None


class _FakeHFDataset:
    def __init__(self, columns):
        self.columns = columns

    def with_format(self, *args, **kwargs):
        del args, kwargs
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.columns
        return {key: value[index] for key, value in self.columns.items()}


class _FakeLeRobotDataset:
    columns = None
    episodes = None

    def __init__(self, **kwargs):
        del kwargs
        self.hf_dataset = _FakeHFDataset(self.columns)
        self.meta = SimpleNamespace(episodes=self.episodes)


def test_v3_action_index_builds_exact_consecutive_25hz_chunk():
    rows = 20
    joint_dim = 2
    action_index = torch.arange(rows).div(2, rounding_mode="floor")
    held_action = action_index[:, None].float().expand(rows, joint_dim)
    _FakeLeRobotDataset.columns = {
        "observation.joint": torch.zeros(rows, joint_dim),
        "observation.velocity": torch.zeros(rows, joint_dim),
        "observation.delta_q": torch.zeros(rows, joint_dim),
        "observation.torque": torch.zeros(rows, joint_dim),
        "action.joint": held_action,
        "timing.state_timestamp_ns": torch.arange(rows) * 10_000_000,
        "timing.action_anchor_timestamp_ns": torch.arange(rows) * 10_000_000,
        "timing.action_index": action_index[:, None],
    }
    _FakeLeRobotDataset.episodes = [
        {"dataset_from_index": 0, "dataset_to_index": rows}
    ]
    config = {
        "dataloader": {
            "backend": "lerobot",
            "repo_id": "fake",
            "root": "fake",
            "high_fps": 100,
            "expert_fps": 25,
            "state_history_horizon": 2,
            "prediction_horizon": 2,
            "action_condition_horizon": 3,
            "action_chunk_horizon": 3,
            "action_rollout_horizon": 3,
            "normalize_mode": None,
            "high_timestamp_key": "timing.state_timestamp_ns",
            "anchor_timestamp_key": "timing.action_anchor_timestamp_ns",
            "high_keys": {
                "q": "observation.joint",
                "dq": "observation.velocity",
                "delta_q": "observation.delta_q",
                "tau": "observation.torque",
                "action": "action.joint",
            },
        },
        "contact_gate": {"enabled": False},
    }

    with patch(
        "data_process.world_model_dataset._load_lerobot_dataset_class",
        return_value=_FakeLeRobotDataset,
    ):
        dataset = TorqueWorldModelDataset(config)

    first = dataset[0]
    second_action = dataset[dataset.valid_indices.index(2)]
    # The anchor row already carries token 0.  The SWM condition starts at
    # the next 25 Hz refresh while anchors continue rolling at 100 Hz.
    assert first["action_chunk_index"].tolist() == [1, 2, 3]
    assert first["action"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert first["action_rollout"].shape == (3, 3, joint_dim)
    # The direct action is held for two 100 Hz rows per 25 Hz token.
    assert first["action_rollout"][:, 0, 0].tolist() == [1.0, 1.0, 2.0]
    assert first["action_rollout"][:, 1, 0].tolist() == [2.0, 2.0, 3.0]
    assert second_action["action_chunk_index"].tolist() == [2, 3, 4]


def test_action_start_offset_zero_remains_available_for_ablation():
    config = {
        "dataloader": {
            "backend": "lerobot",
            "repo_id": "fake",
            "root": "fake",
            "high_fps": 100,
            "expert_fps": 25,
            "state_history_horizon": 2,
            "prediction_horizon": 2,
            "action_condition_horizon": 2,
            "action_chunk_horizon": 2,
            "action_start_offset": 0,
            "normalize_mode": None,
            "high_timestamp_key": "timing.state_timestamp_ns",
            "anchor_timestamp_key": "timing.action_anchor_timestamp_ns",
            "high_keys": {
                "q": "observation.joint",
                "dq": "observation.velocity",
                "delta_q": "observation.delta_q",
                "tau": "observation.torque",
                "action": "action.joint",
            },
        },
        "contact_gate": {"enabled": False},
    }
    with patch(
        "data_process.world_model_dataset._load_lerobot_dataset_class",
        return_value=_FakeLeRobotDataset,
    ):
        dataset = TorqueWorldModelDataset(config)
    assert dataset[0]["action_chunk_index"].tolist() == [0, 1]
