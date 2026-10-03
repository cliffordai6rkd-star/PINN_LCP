import copy

import pytest
import torch

from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
from train.contact_world_model_loss import ContactWorldModelLoss
from train.trainer.contact_world_model_distill_train import ContactWorldModelDistillTrainer


def config():
    return {
        "dataloader": {"state_history_horizon": 3, "prediction_horizon": 3,
                       "action_condition_horizon": 2, "high_fps": 100},
        "model": {"inputs": ["q", "tau"], "outputs": ["q", "tau"],
                  "joint_dim": 2, "action_dim": 2, "hidden_dim": 8,
                  "state_layers": 1, "action_layers": 1, "flow_layers": 1,
                  "flow_attention_heads": 2, "flow_ffn_multiplier": 2,
                  "flow_inference_steps": 8, "flow_solver": "heun", "dropout": 0.1},
        "loss": {"dt": 0.01},
    }


def batch():
    return {"q": torch.randn(1, 3, 2), "tau": torch.randn(1, 3, 2),
            "q_future": torch.randn(1, 3, 2), "tau_future": torch.randn(1, 3, 2),
            "action": torch.randn(1, 2, 2), "action_mask": torch.ones(1, 2, dtype=torch.bool),
            "importance_weight": torch.ones(1)}


@pytest.mark.parametrize("steps", [2, 4])
def test_student_initial_field_calls_gradients_and_reload(steps, tmp_path):
    torch.set_num_threads(1)
    teacher = ContactWorldModel(config()).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=steps).eval()
    assert all(parameter.requires_grad for parameter in student.parameters())
    values = batch()
    encoded_teacher = teacher.encode_conditions(values)
    encoded_student = student.encode_conditions(values)
    noise = torch.randn(1, 3, 4)
    flow_time = torch.full((1, 1), 0.25)
    delta = torch.full((1, 1), 1 / steps)
    with torch.no_grad():
        baseline, _ = teacher.flow_velocity(noise, flow_time, encoded_teacher)
        initialized, _ = student.flow_velocity_student(noise, flow_time, delta, encoded_student)
    torch.testing.assert_close(initialized, baseline, rtol=0, atol=0)

    calls = []
    handle = student.flow_blocks[0].register_forward_hook(lambda *_: calls.append(None))
    output = student.predict_differentiable(values, source_noise=noise)
    handle.remove()
    assert len(calls) == steps
    output["flow_state_pred"].square().mean().backward()
    assert student.flow_input_projection.weight.grad is not None
    assert student.state_encoders["q"].weight_ih_l0.grad is not None
    assert student.flow_delta_embedding.projection[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())

    path = tmp_path / "student.pt"
    torch.save({"model_version": student.MODEL_VERSION,
                "carswm_contract": student.checkpoint_contract(),
                "model": student.state_dict()}, path)
    restored = ContactWorldModelStudent(copy.deepcopy(student._config)).eval()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored.validate_checkpoint(payload)
    restored.load_state_dict(payload["model"], strict=True)
    with torch.no_grad():
        actual = restored.predict(values, source_noise=noise)["flow_state_pred"]
    torch.testing.assert_close(actual, output["flow_state_pred"])


@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_distillation_complete_chain_and_teacher_are_separated(steps, device):
    torch.set_num_threads(1)
    cfg = config()
    if device == "cuda":
        cfg["model"].update(state_layers=2, action_layers=2)
    teacher = ContactWorldModel(cfg).to(device).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=steps).to(device)
    trainer = ContactWorldModelDistillTrainer.__new__(ContactWorldModelDistillTrainer)
    trainer.teacher = teacher
    trainer.model = student
    trainer.device = device
    trainer.device_batch_keys = None
    trainer.non_blocking_transfer = False
    trainer.teacher_steps = 64
    trainer.student_steps = steps
    trainer.global_step = 100
    trainer.loss_calculator = ContactWorldModelLoss(student._config)
    trainer.weights = {"local_weight": 1.0, "terminal_weight": 1.0,
                       "q_d1_weight": 0.05, "q_d2_weight": 0.01,
                       "contact_distill_weight": 0.0, "contact_ce_weight": 0.0}
    trainer.student_start_probability_max = 0.5
    trainer.student_start_probability_warmup_steps = 100
    loss, info = trainer.compute_loss({key: value.to(device) for key, value in batch().items()})
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert student.flow_output[-1].weight.grad is not None
    assert student.state_encoders["q"].weight_ih_l0.grad is not None
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert info["loss_dict"]["student_start_probability"] == 0.5
    assert student.state_encoders["q"].dropout == 0.0


def test_checkpoint_stage_contract_and_full_sampling_validation(tmp_path):
    torch.set_num_threads(1)
    cfg = config()
    cfg["train"] = {"seed": 17, "split_mode": "episode", "val_ratio": 0.2}
    teacher = ContactWorldModel(cfg).eval()
    normalizer = {"stats": {"q": {"mean": torch.zeros(2)}}, "normalize_mode": None}
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"config": cfg, "model": teacher.state_dict(),
                "carswm_contract": teacher.checkpoint_contract(),
                "model_version": teacher.MODEL_VERSION,
                "ema": {"enabled": True}, "normalizer": normalizer}, teacher_path)
    overlay = {"distillation": {"enabled": True, "teacher_checkpoint_path": str(teacher_path),
                                "teacher_steps": 64, "student_steps": 4,
                                "validation_num_samples": 2, "latency_batches": 0},
               "train": {"device": "cpu", "output_dir": str(tmp_path / "s4"),
                         "max_optimizer_steps": 1}}
    stage4 = ContactWorldModelDistillTrainer(overlay)
    stage4.model = stage4.build_model()
    assert stage4.config["train"]["seed"] == 17
    assert stage4.config["train"]["val_ratio"] == 0.2
    stage4.global_step = 0
    stage4.val_loader = [batch()]
    score = stage4.validate_one_epoch(0, force=True)
    assert score >= 0
    assert "student_q_d1_teacher_mse" in stage4.last_val_epoch_metrics
    assert "short_teacher_energy_score" in stage4.last_val_epoch_metrics

    student_path = tmp_path / "s4.pt"
    torch.save({"config": stage4.config, "model": stage4.model.state_dict(),
                "carswm_contract": stage4.model.checkpoint_contract(),
                "model_version": stage4.model.MODEL_VERSION,
                "global_step": 1,
                "normalizer": normalizer}, student_path)
    overlay2 = copy.deepcopy(overlay)
    overlay2["distillation"].update(student_steps=2, student_init_checkpoint_path=str(student_path))
    stage2 = ContactWorldModelDistillTrainer(overlay2)
    initialized = stage2.build_model()
    for name, value in stage4.model.state_dict().items():
        torch.testing.assert_close(initialized.state_dict()[name], value)
    assert initialized.student_steps == 2


def test_validation_uses_internal_future_horizon_after_temporal_stride():
    torch.set_num_threads(1)
    cfg = config()
    cfg["dataloader"].update(state_history_horizon=4, prediction_horizon=4)
    cfg["train"] = {"downsample": 2}
    teacher = ContactWorldModel(cfg).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=2).eval()
    trainer = ContactWorldModelDistillTrainer.__new__(ContactWorldModelDistillTrainer)
    trainer.teacher = teacher
    trainer.model = student
    trainer.device = "cpu"
    trainer.ema = None
    trainer.device_batch_keys = None
    trainer.non_blocking_transfer = False
    trainer.ema_use_for_validation = True
    trainer.student_steps = 2
    trainer.global_step = 0
    trainer.step_based_training = False
    trainer.validation_num_samples = 2
    trainer.validation_max_batches = 0
    trainer.validation_seed = 9
    trainer.latency_batches = 0
    trainer.config = cfg
    values = {"q": torch.randn(1, 4, 2), "tau": torch.randn(1, 4, 2),
              "q_future": torch.randn(1, 4, 2), "tau_future": torch.randn(1, 4, 2),
              "action": torch.randn(1, 2, 2), "action_mask": torch.ones(1, 2, dtype=torch.bool)}
    trainer.val_loader = [values]
    assert torch.isfinite(torch.tensor(trainer.validate_one_epoch(0)))


@pytest.mark.parametrize('steps', [2, 4])
def test_student_cache_equivalence_and_each_flow_state_receives_terminal_gradient(steps):
    torch.set_num_threads(1)
    teacher = ContactWorldModel(config()).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=steps).eval()
    values, noise = batch(), torch.randn(1, 3, 4)
    with torch.no_grad():
        uncached = student.predict(values, source_noise=noise, cache_condition_kv=False)
        cached = student.predict(values, source_noise=noise, cache_condition_kv=True)
        for key in ('flow_state_pred', 'contact_probability'):
            torch.testing.assert_close(uncached[key], cached[key], atol=2e-6, rtol=2e-5)
    encoded = student.encode_conditions(values)
    states = student.integrate_flow(noise, encoded, return_states=True)
    for state in states[1:]:
        state.retain_grad()
    states[-1].square().mean().backward()
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in states[1:])
    assert student.action_encoder.weight_ih_l0.grad.abs().sum() > 0
    assert student.state_encoders['q'].weight_ih_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError, match='separately distilled'):
        student.predict(values, steps=8, source_noise=noise)
    with pytest.raises(ValueError, match='Euler'):
        student.predict(values, solver='heun', source_noise=noise)


def test_checkpoint_dispatch_metadata_and_strict_state():
    from model.pinn_model.checkpoint import model_from_checkpoint
    teacher = ContactWorldModel(config())
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=4)
    for model in (teacher, student):
        payload = {'config': model._config, 'model_version': model.MODEL_VERSION,
                   'carswm_contract': model.checkpoint_contract(), 'model': model.state_dict()}
        restored = model_from_checkpoint(payload)
        assert type(restored) is type(model)
        restored.load_state_dict(payload['model'], strict=True)
    payload['model'] = teacher.state_dict()
    with pytest.raises(RuntimeError, match='Missing key'):
        model_from_checkpoint(payload).load_state_dict(payload['model'], strict=True)
    payload['carswm_contract']['student']['steps'] = 2
    with pytest.raises(ValueError, match='contract mismatch'):
        model_from_checkpoint(payload)


def test_low_frequency_gradient_diagnostics_do_not_overwrite_accumulated_gradients():
    torch.set_num_threads(1)
    teacher = ContactWorldModel(config()).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=4)
    trainer = ContactWorldModelDistillTrainer.__new__(ContactWorldModelDistillTrainer)
    trainer.teacher, trainer.model, trainer.device = teacher, student, 'cpu'
    trainer.teacher_steps, trainer.student_steps, trainer.global_step = 64, 4, 10
    trainer.loss_calculator = ContactWorldModelLoss(student._config)
    trainer.weights = {'local_weight': 1., 'terminal_weight': 1., 'q_d1_weight': 0., 'q_d2_weight': 0.,
                       'contact_distill_weight': .1, 'contact_ce_weight': 0.}
    trainer.student_start_probability_max, trainer.student_start_probability_warmup_steps = .5, 10
    trainer.diagnostics_every, trainer.local_loss_mode = 10, 'endpoint'
    shared = student.flow_input_projection.weight
    shared.grad = torch.ones_like(shared)
    previous = shared.grad.clone()
    loss, info = trainer.compute_loss(batch())
    torch.testing.assert_close(shared.grad, previous)
    assert 'local_shared_grad_norm' in info['loss_dict']
    assert 'terminal_shared_grad_norm' in info['loss_dict']
    assert 'flow_state_mse_s1' in info['loss_dict']
    (loss / 2).backward()
    accumulated = shared.grad.clone()
    loss2, info2 = trainer.compute_loss(batch())
    torch.testing.assert_close(shared.grad, accumulated)
    assert 'local_shared_grad_norm' not in info2['loss_dict']
    (loss2 / 2).backward()
    assert torch.isfinite(shared.grad).all()
    assert all(p.grad is None for p in teacher.parameters())


def test_teacher_request_cache_matches_uncached_and_is_built_once_for_draws(monkeypatch):
    torch.set_num_threads(1)
    teacher = ContactWorldModel(config()).eval().requires_grad_(False)
    values, noise = batch(), torch.randn(1, 3, 3, 4)
    calls = []
    original = teacher.flow_blocks[0].prepare_condition_kv
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(teacher.flow_blocks[0], 'prepare_condition_kv', count)
    eager = teacher.sample(values, num_samples=3, steps=4, solver='euler', source_noise=noise)
    cached = teacher.sample(values, num_samples=3, steps=4, solver='euler', source_noise=noise, cache_condition_kv=True)
    assert len(calls) == 1
    torch.testing.assert_close(cached['flow_state_pred'], eager['flow_state_pred'], atol=2e-6, rtol=2e-5)
    assert all(p.grad is None for p in teacher.parameters())


def test_endpoint_execution_and_optional_physical_join_with_partially_normalized_streams():
    torch.set_num_threads(1)
    cfg = config()
    cfg['model']['inputs'] = ['q', 'dq', 'tau']
    cfg['dataloader'].update(normalize_mode='gaussian', normalize_lowdim_keys=['tau', 'action'])
    teacher = ContactWorldModel(cfg).eval().requires_grad_(False)
    student = ContactWorldModelStudent.from_teacher(teacher, student_steps=4)
    trainer = ContactWorldModelDistillTrainer.__new__(ContactWorldModelDistillTrainer)
    trainer.teacher, trainer.model, trainer.device, trainer.config = teacher, student, 'cpu', cfg
    trainer.teacher_steps, trainer.student_steps, trainer.global_step = 64, 4, 0
    trainer.loss_calculator = ContactWorldModelLoss(cfg)
    # q and dq are physical already: no statistics should be required for them.
    from train.nomalizer import Normalizer
    trainer.loss_calculator.normalizer = Normalizer({})
    trainer.weights = {'local_weight': 1., 'terminal_weight': 1., 'q_d1_weight': 0., 'q_d2_weight': 0.,
                       'contact_distill_weight': 0., 'contact_ce_weight': 0., 'execution_weight': .25, 'join_weight': .01}
    trainer.student_start_probability_max, trainer.student_start_probability_warmup_steps = .5, 5000
    trainer.local_loss_mode = 'endpoint'
    trainer.execution_window = {'delay_steps': 1, 'execute_steps': 2, 'mode': 'prefetch'}
    values = batch()
    values['dq'] = torch.randn_like(values['q'])
    loss, info = trainer.compute_loss(values)
    assert torch.isfinite(loss) and torch.isfinite(info['loss_dict']['join'])
    assert info['loss_dict']['execution'] > 0
    loss.backward()
    assert student.state_encoders['dq'].weight_ih_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
