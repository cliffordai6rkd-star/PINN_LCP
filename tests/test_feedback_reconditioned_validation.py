from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from train.base_trainer import BaseTrainer
from train.trainer.contact_world_model_train import ContactWorldModelTrainer


STREAMS = ("q", "dq", "delta_q", "tau")


class _RecordingModel:
    inputs = STREAMS
    predicted_state_streams = STREAMS
    future_horizon = 32
    flow_dim = 4

    def __init__(self):
        self.calls = []
        self.training = False

    def eval(self):
        self.training = False
        return self

    def predict(self, batch, *, steps, solver, source_noise):
        del steps, solver, source_noise
        batch_size = batch["q"].shape[0]
        result = {
            f"{key}_pred": torch.zeros(batch_size, self.future_horizon, 1)
            for key in self.predicted_state_streams
        }
        result["flow_state_pred"] = torch.cat(
            [
                result[f"{key}_pred"]
                for key in self.predicted_state_streams
            ],
            dim=-1,
        )
        result["contact_state_pred"] = torch.zeros(
            batch_size, self.future_horizon, 1
        )
        return result

    def sample(
        self,
        batch,
        *,
        num_samples,
        steps,
        solver,
        source_noise,
    ):
        del steps, solver
        assert set(batch) == set(STREAMS) | {"action", "action_mask"}
        self.calls.append(
            {
                "batch": {key: value.clone() for key, value in batch.items()},
                "source_noise": source_noise.clone(),
            }
        )
        batch_size = batch["q"].shape[0]
        result = {}
        for stream_index, key in enumerate(self.predicted_state_streams):
            result[f"{key}_pred"] = torch.full(
                (batch_size, num_samples, self.future_horizon, 1),
                900.0 + stream_index,
            )
        logits = torch.zeros(
            batch_size, num_samples, self.future_horizon, 3
        )
        result["contact_probability"] = logits.softmax(dim=-1)
        return result


def _trainer(num_samples=2):
    trainer = ContactWorldModelTrainer.__new__(ContactWorldModelTrainer)
    trainer.model = _RecordingModel()
    trainer.rollout_source_seed = 2027
    trainer.feedback_num_samples = num_samples
    trainer.rollout_validation_steps = 2
    trainer.rollout_solver = "heun"
    trainer.rollout_divergence_threshold = 1000.0
    trainer.loss_calculator = SimpleNamespace(contact_state_count=3)
    trainer.autocast_context = lambda: nullcontext()
    return trainer


def _batch():
    history = torch.tensor([[[-2.0], [-1.0], [0.0]]])
    future = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32, 1)
    batch = {}
    for stream_index, key in enumerate(STREAMS):
        batch[key] = history + 100.0 * stream_index
        batch[f"{key}_future"] = future + 100.0 * stream_index
    action_rollout = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1, 1)
    batch["action_rollout"] = action_rollout.repeat(1, 1, 2, 1)
    batch["action_rollout_mask"] = torch.ones(1, 32, 2)
    batch["action"] = batch["action_rollout"][:, 0].clone()
    batch["action_mask"] = batch["action_rollout_mask"][:, 0].clone()
    phases = torch.zeros(1, 32, 1)
    phases[:, 4:12] = 1
    phases[:, 12:] = 2
    batch["contact_future"] = phases
    return batch


def _trainer_config(*, feedback_reconditioned=True):
    return {
        "dataloader": {
            "state_history_horizon": 3,
            "prediction_horizon": 4,
            "action_condition_horizon": 2,
            "action_rollout_horizon": 4 if feedback_reconditioned else 0,
            "high_fps": 100,
            "normalize_mode": None,
        },
        "model": {
            "inputs": list(STREAMS),
            "joint_dim": 1,
            "action_dim": 1,
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
        "loss": {
            "dt": 0.01,
            "kinematic_consistency_weight": 0.0,
            "ddq_smoothness_weight": 0.0,
        },
        "train": {
            "device": "cpu",
            "num_workers": 0,
            "rollout_validation": {
                "enabled": True,
                "feedback_reconditioned": feedback_reconditioned,
            },
            "probabilistic_validation": {"enabled": False},
            "ema": {"enabled": False},
            "wandb": {"enabled": False},
        },
    }


def test_feedback_config_defaults_and_teacher_only_disable_switch():
    teacher = ContactWorldModelTrainer(_trainer_config())
    assert teacher.feedback_measurement_update_intervals == (1, 4, 8, 32)
    assert teacher.feedback_num_samples == 8

    disabled = ContactWorldModelTrainer(
        _trainer_config(feedback_reconditioned=False)
    )
    assert disabled.feedback_measurement_update_intervals == ()


def test_feedback_writes_back_measurements_and_moves_action_anchor_without_leakage():
    trainer = _trainer()
    batch = _batch()

    result = trainer._feedback_reconditioned_samples(batch, 3, interval=4)

    assert result["anchors"] == tuple(range(0, 32, 4))
    assert len(trainer.model.calls) == 8
    for call_index, call in enumerate(trainer.model.calls):
        anchor = call_index * 4
        condition = call["batch"]
        torch.testing.assert_close(
            condition["action"], batch["action_rollout"][:, anchor]
        )
        torch.testing.assert_close(
            condition["action_mask"], batch["action_rollout_mask"][:, anchor]
        )
        for stream_index, key in enumerate(STREAMS):
            if anchor == 0:
                expected = batch[key]
            else:
                expected = batch[f"{key}_future"][:, anchor - 3 : anchor]
            torch.testing.assert_close(condition[key], expected)
            assert not torch.any(condition[key] >= 900.0)


def test_feedback_updates_all_input_histories_for_reduced_outputs():
    trainer = _trainer()
    trainer.model.predicted_state_streams = ("q", "tau")
    batch = _batch()

    result = trainer._feedback_reconditioned_samples(
        batch, 0, interval=4
    )

    assert set(result["samples"]) == {"q", "tau"}
    second_condition = trainer.model.calls[1]["batch"]
    for key in STREAMS:
        expected = batch[f"{key}_future"][:, 1:4]
        torch.testing.assert_close(second_condition[key], expected)


def test_feedback_interval_32_has_single_open_loop_time_semantics():
    trainer = _trainer(num_samples=1)
    batch = _batch()

    result = trainer._feedback_reconditioned_samples(batch, 5, interval=32)

    assert result["anchors"] == (0,)
    assert len(trainer.model.calls) == 1
    call = trainer.model.calls[0]
    for key in STREAMS:
        torch.testing.assert_close(call["batch"][key], batch[key])
    torch.testing.assert_close(call["batch"]["action"], batch["action"])
    open_loop_noise = trainer._fixed_source_noise(batch, 5, step=0)
    torch.testing.assert_close(call["source_noise"][:, 0], open_loop_noise)
    assert result["samples"]["q"].shape == (1, 1, 32, 1)


def test_first_feedback_draw_matches_open_loop_noise_for_a_batch():
    trainer = _trainer(num_samples=3)
    batch = _batch()
    batch = {
        key: value.repeat((2,) + (1,) * (value.ndim - 1))
        for key, value in batch.items()
    }

    sampled = trainer._fixed_source_noise(
        batch, 9, step=0, num_samples=3
    )
    open_loop = trainer._fixed_source_noise(batch, 9, step=0)

    torch.testing.assert_close(sampled[:, 0], open_loop)


def test_feedback_metrics_use_feedback_prefixes_and_existing_phase_labels():
    trainer = _trainer()
    accumulator = defaultdict(float)

    trainer._accumulate_feedback_interval(accumulator, _batch(), 0, interval=4)
    metrics = trainer._metric_finalize(accumulator)

    assert "feedback_u4_energy_score" in metrics
    assert "feedback_u4_min_ade" in metrics
    assert "feedback_u4_min_fde" in metrics
    assert "feedback_u4_sample_spread" in metrics
    assert "feedback_u4_coverage_90" in metrics
    assert "feedback_u4_contact_nll" in metrics
    assert "feedback_u4_contact_brier" in metrics
    assert "feedback_u4_contact_macro_f1" in metrics
    assert "feedback_u4_phase_free_energy_score" in metrics
    assert "feedback_u4_phase_transition_energy_score" in metrics
    assert "feedback_u4_phase_contact_energy_score" in metrics


def test_rollout_validation_records_all_three_evaluation_paths():
    trainer = _trainer(num_samples=2)
    trainer.val_loader = [_batch()]
    trainer.ema = None
    trainer.ema_use_for_validation = True
    trainer.rollout_max_batches = 0
    trainer.feedback_max_batches = 0
    trainer.feedback_measurement_update_intervals = (4, 32)
    trainer.free_running_steps = 1
    trainer.free_running_max_batches = 0
    trainer.rollout_horizons = (1, 4, 32)
    trainer.dataset = SimpleNamespace(normalizer=None)
    trainer.batch_to_device = lambda value: value

    metrics = trainer._run_rollout_validation(epoch=7)

    assert "rollout_mse_h32" in metrics
    assert "free_running_mse_h1" in metrics
    assert "feedback_u4_energy_score" in metrics
    assert "feedback_u32_energy_score" in metrics
    assert metrics["feedback_u4_num_samples"] == 2
    assert metrics["feedback_u32_epoch"] == 7


def test_feedback_does_not_replace_validation_loss_unless_enabled(monkeypatch):
    trainer = _trainer()
    trainer.val_loader = object()
    trainer.last_val_epoch_metrics = {}
    trainer.rollout_validation_enabled = True
    trainer.rollout_replace_val_loss_metric = "feedback_u4_energy_score"
    trainer.probabilistic_validation_enabled = False
    trainer._run_rollout_validation = lambda epoch: {
        "feedback_u4_energy_score": 2.5,
        "rollout_epoch": epoch,
    }
    monkeypatch.setattr(
        BaseTrainer, "validate_one_epoch", lambda self, epoch: 7.0
    )

    trainer.rollout_replace_val_loss = False
    assert trainer.validate_one_epoch(3) == 7.0
    trainer.rollout_replace_val_loss = True
    assert trainer.validate_one_epoch(3) == 2.5
