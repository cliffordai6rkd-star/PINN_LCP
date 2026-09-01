import pytest
import torch

from model.pinn_model.contact_world_model import ContactWorldModel
from train.contact_world_model_loss import ContactWorldModelLoss
from train.trainer.contact_world_model_opd_train import ContactWorldModelOPDTrainer


def cfg():
    return {
        "dataloader": {"state_history_horizon": 4, "prediction_horizon": 3, "action_condition_horizon": 2, "high_fps": 100},
        "model": {"inputs": ["q", "dq", "delta_q", "tau"], "joint_dim": 2, "action_dim": 2, "contact_state_count": 3, "hidden_dim": 8, "state_layers": 1, "action_layers": 1, "attention_heads": 2, "flow_layers": 1, "flow_attention_heads": 2, "flow_ffn_multiplier": 2, "flow_inference_steps": 1, "flow_solver": "euler", "flow_source_mode": "gaussian", "dropout": 0.0},
        "loss": {"dt": 0.01, "kinematic_consistency_weight": 0.0, "ddq_smoothness_weight": 0.0},
    }


def batch():
    value = {key: torch.randn(1, 4, 2) for key in ("q", "dq", "delta_q", "tau")}
    value.update({f"{key}_future": torch.randn(1, 3, 2) for key in ("q", "dq", "delta_q", "tau")})
    value.update(action=torch.randn(1, 2, 2), action_mask=torch.ones(1, 2, dtype=torch.bool), contact_future=torch.zeros(1, 3, 1), contact=torch.zeros(1, 4, 1))
    return value


def test_opd_reuses_source_and_relabels_teacher_on_student_history():
    config = cfg()
    trainer = ContactWorldModelOPDTrainer.__new__(ContactWorldModelOPDTrainer)
    trainer.model = None
    trainer.model = ContactWorldModel(config)
    trainer.teacher = ContactWorldModel(config)
    trainer.teacher_steps = 1
    trainer.student_steps = 1
    trainer.loss_calculator = ContactWorldModelLoss(config)
    trainer.rollout_contact_enabled = False
    trainer._sample_source_noise = lambda value: torch.zeros(1, 3, 11)
    calls = []
    original = trainer.teacher.predict

    def predict(value, **kwargs):
        calls.append(value["q"][:, -1:].clone())
        return original(value, **kwargs)

    trainer.teacher.predict = predict
    value = batch()
    loss, _ = trainer._rollout_distill(value, rollout_steps=2)
    assert torch.isfinite(loss)
    assert len(calls) == 2
    assert not torch.equal(calls[0], calls[1])


def test_opd_write_back_commits_all_streams_first_frame_and_action_chunk():
    trainer = ContactWorldModelOPDTrainer.__new__(ContactWorldModelOPDTrainer)
    trainer.model = None
    value = batch()
    value["action_rollout"] = torch.stack((value["action"], value["action"] + 1), dim=1)
    value["action_rollout_mask"] = torch.ones(1, 2, 2)
    predicted = {f"{key}_pred": torch.ones(1, 3, 2) * 5 for key in ("q", "dq", "delta_q", "tau")}
    next_value = trainer._write_back(value, predicted, rollout_step=0)
    for key in ("q", "dq", "delta_q", "tau"):
        assert torch.equal(next_value[key][:, -1], predicted[f"{key}_pred"][:, 0])
    assert torch.equal(next_value["action"], value["action_rollout"][:, 1])


def test_teacher_action_contract_mismatch_is_rejected():
    config = cfg()
    config["dataloader"]["action_key"] = "action.ee_pose"
    trainer = ContactWorldModelOPDTrainer.__new__(ContactWorldModelOPDTrainer)
    trainer.config = config
    teacher = {"dataloader": {**config["dataloader"], "action_key": "action.other"}, "model": config["model"], "train_data": {"action_alignment": "next"}}
    config["train_data"] = {"action_alignment": "next"}
    with torch.no_grad():
        try:
            trainer._validate_teacher_contract(teacher, {"normalizer": {"stats": {}}})
        except ValueError as error:
            assert "contract mismatch" in str(error)
        else:
            raise AssertionError("action contract mismatch was not rejected")


def test_teacher_action_start_offset_mismatch_is_rejected():
    config = cfg()
    config["dataloader"]["action_start_offset"] = 0
    config["train_data"] = {"action_alignment": "next"}
    trainer = ContactWorldModelOPDTrainer.__new__(ContactWorldModelOPDTrainer)
    trainer.config = config
    teacher = {
        "dataloader": {**config["dataloader"], "action_start_offset": 1},
        "model": config["model"],
        "train_data": {"action_alignment": "next"},
    }
    with pytest.raises(ValueError, match="contract mismatch"):
        trainer._validate_teacher_contract(teacher, {"normalizer": {"stats": {}}})
