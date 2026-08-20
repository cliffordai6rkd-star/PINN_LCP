from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from model.pinn_model.contact_gate import (
    ContactGateConfig,
    hysteresis_three_phase_mask,
)
from model.pinn_model.torque_world_model import TorqueWorldModel
from train.torque_world_model_loss import TorqueWorldModelLoss
from train.trainer.torque_world_model_opd_train import TorqueWorldModelOPDTrainer


def _config():
    return yaml.safe_load(
        Path("config/train_cfg/contact_world_model.yaml").read_text()
    )


def test_metric_specific_hysteresis_thresholds_are_configurable():
    config = _config()
    config["contact_gate"]["metric"] = "tau_ext_l1"
    gate = ContactGateConfig.from_config(config)
    assert gate.off_threshold == 0.15
    assert gate.on_threshold == 0.75


def test_three_phase_hysteresis_has_precontact_and_confirmation():
    values = torch.tensor([0.0, 0.4, 0.8, 1.6, 1.7, 1.0, 0.2, 0.2])
    labels = hysteresis_three_phase_mask(
        values,
        off_threshold=0.3,
        on_threshold=1.5,
        consecutive_frames=2,
    )
    torch.testing.assert_close(
        labels,
        torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
    )


def test_contact_world_model_does_not_require_dq_and_backpropagates():
    config = _config()
    config["dataloader"]["normalize_mode"] = None
    model = TorqueWorldModel(config)
    loss_fn = TorqueWorldModelLoss(config)
    loss_fn.set_contact_class_weights([1.0, 2.0, 1.0])
    batch = {
        "q": torch.randn(2, 50, 7),
        "tau": torch.randn(2, 50, 7),
        "target_relative_pose": torch.randn(2, 8, 21),
        "target_relative_pose_mask": torch.ones(2, 8, dtype=torch.bool),
        "q_future": torch.randn(2, 32, 7),
        "tau_future": torch.randn(2, 32, 7),
        "contact_future": torch.randint(0, 3, (2, 32, 1)).float(),
    }
    output = model(batch, flow_time=0.5)
    assert output["contact_logits"].shape == (2, 32, 3)
    loss, metrics = loss_fn(output, batch)
    loss.backward()
    assert model.state_encoder.input_size == 14
    assert output["dq_pred_physical"].shape == (2, 32, 7)
    assert output["ddq_pred_physical"].shape == (2, 32, 7)
    assert "dq_loss" in metrics and "ddq_loss" in metrics


def test_q_only_estimator_gives_zero_derivative_residual_for_matching_q():
    config = _config()
    config["dataloader"]["normalize_mode"] = None
    loss_fn = TorqueWorldModelLoss(config)
    batch = {
        "q": torch.randn(1, 50, 7),
        "q_future": torch.randn(1, 32, 7),
        "tau_future": torch.randn(1, 32, 7),
        "contact_future": torch.zeros(1, 32, 1),
    }
    q = batch["q_future"]
    out = {
        "flow_velocity_pred": torch.zeros(1, 32, 17),
        "flow_velocity_target": torch.zeros(1, 32, 17),
        "q_pred": q,
        "tau_pred": batch["tau_future"],
        "contact_logits": torch.zeros(1, 32, 3),
    }
    loss_fn.set_contact_class_weights([1.0, 1.0, 1.0])
    _, metrics = loss_fn(out, batch)
    assert float(metrics["dq_loss"]) == 0.0
    assert float(metrics["ddq_loss"]) == 0.0


def test_contact_opd_distills_contact_logits_without_dq_history():
    config = _config()
    config["dataloader"]["normalize_mode"] = None
    config["model"]["flow_inference_steps"] = 2
    student = TorqueWorldModel(config)
    teacher = TorqueWorldModel(config)
    trainer = TorqueWorldModelOPDTrainer.__new__(TorqueWorldModelOPDTrainer)
    trainer.model = student
    trainer.teacher = teacher
    trainer.teacher_steps = 8
    trainer.student_steps = 2
    trainer.contact_distill_weight = 1.0
    trainer.pose_dynamics = None
    trainer.action_condition_features = ()
    batch = {
        "q": torch.randn(1, 50, 7),
        "tau": torch.randn(1, 50, 7),
        "target_relative_pose": torch.randn(1, 8, 21),
        "target_relative_pose_mask": torch.ones(1, 8, dtype=torch.bool),
        "q_future": torch.randn(1, 32, 7),
        "tau_future": torch.randn(1, 32, 7),
        "contact_future": torch.randint(0, 3, (1, 32, 1)).float(),
    }
    loss, output = trainer._endpoint_distill(batch)
    loss.backward()
    next_batch = trainer._write_back(batch, output)
    teacher_batch = trainer._write_back_real(batch, batch, 0)
    assert output["contact_logits"].shape == (1, 32, 3)
    assert next_batch["q"].shape == batch["q"].shape
    assert teacher_batch["tau"].shape == batch["tau"].shape
    assert "dq" not in next_batch


def test_opd_student_fk_only_evaluates_committed_first_prediction():
    trainer = TorqueWorldModelOPDTrainer.__new__(TorqueWorldModelOPDTrainer)
    calls = []

    class FakePoseDynamics:
        def frame_poses(self, q):
            calls.append(tuple(q.shape))
            return torch.zeros(*q.shape[:-1], 7)

    trainer.model = SimpleNamespace(q_tau_contact_contract=True, wrench_dim=0)
    trainer.pose_dynamics = FakePoseDynamics()
    trainer.action_condition_features = ("relative_pose",)
    trainer.loss_calculator = SimpleNamespace(_physical=lambda key, value: value)
    trainer.dataset = SimpleNamespace(_normalize=lambda key, value: value)
    batch = {
        "q": torch.randn(2, 50, 7),
        "tau": torch.randn(2, 50, 7),
        "target_pose_abs": torch.randn(2, 8, 7),
        "target_relative_pose": torch.randn(2, 8, 7),
        "current_ee_pose": torch.randn(2, 50, 7),
    }
    student_out = {
        "q_pred": torch.randn(2, 32, 7),
        "tau_pred": torch.randn(2, 32, 7),
    }

    next_batch = trainer._write_back(batch, student_out)

    assert calls == [(2, 1, 7)]
    assert next_batch["current_ee_pose"].shape == (2, 50, 7)


def test_opd_real_rollout_reuses_cached_future_pose_without_fk():
    trainer = TorqueWorldModelOPDTrainer.__new__(TorqueWorldModelOPDTrainer)
    calls = []

    class FakePoseDynamics:
        def frame_poses(self, q):
            calls.append(tuple(q.shape))
            return torch.zeros(*q.shape[:-1], 7)

    trainer.model = SimpleNamespace(q_tau_contact_contract=True, wrench_dim=0)
    trainer.pose_dynamics = FakePoseDynamics()
    trainer.action_condition_features = ("relative_pose",)
    trainer.loss_calculator = SimpleNamespace(_physical=lambda key, value: value)
    trainer.dataset = SimpleNamespace(_normalize=lambda key, value: value)
    batch = {
        "q": torch.randn(2, 50, 7),
        "tau": torch.randn(2, 50, 7),
        "target_pose_abs": torch.randn(2, 8, 7),
        "target_relative_pose": torch.randn(2, 8, 7),
        "current_ee_pose": torch.randn(2, 50, 7),
    }
    reference_batch = {
        **batch,
        "q_future": torch.randn(2, 32, 7),
        "tau_future": torch.randn(2, 32, 7),
        "current_ee_pose_future": torch.randn(2, 32, 7),
    }

    next_batch = trainer._write_back_real(batch, reference_batch, 3)

    assert calls == []
    torch.testing.assert_close(
        next_batch["current_ee_pose"][:, -1],
        reference_batch["current_ee_pose_future"][:, 3],
    )
