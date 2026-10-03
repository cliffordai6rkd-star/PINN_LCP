"""Focused source/inversion checks without LeRobot data or robot hardware."""
import copy

import torch

from model.pinn_model.inverse_gaussian_source import ConditionalGaussianLatentSource, invert_heun
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from scripts.posttrain_inverse_gaussian_latent import heun_rollout, make_student
from test_latent_contact_world_model import batch, config, ready_model


def test_conditional_source_starts_as_standard_normal():
    source = ConditionalGaussianLatentSource(16, 4, 5, hidden_dim=12)
    condition = torch.randn(2, 16)
    noise = torch.randn(2, 3, 4, 5)
    torch.testing.assert_close(source.transform(condition, noise), noise)
    torch.testing.assert_close(source.kl_per_sample(condition), torch.zeros(2))
    assert source.nll_per_sample(condition, noise[:, 0]).shape == (2,)


def test_fixed_point_heun_inverts_deployed_discrete_map():
    condition = {"scale": torch.tensor(0.4)}

    def velocity(state, time, encoded):
        return encoded["scale"] * state + 0.1 * time

    source = torch.randn(3, 4, 5)
    state = source
    steps = 4
    for i in range(steps):
        first = velocity(state, i/steps, condition)
        second = velocity(state + first/steps, (i+1)/steps, condition)
        state = state + (first+second)/(2*steps)
    recovered, cycle = invert_heun(velocity, state, condition, steps=steps, iterations=8)
    torch.testing.assert_close(recovered, source, atol=1e-5, rtol=1e-5)
    assert cycle.max() < 1e-5


def test_new_source_preserves_baseline_initially_and_reduces_nfe():
    cfg = config()
    cfg["model"].update(flow_inference_steps=16, flow_solver="heun")
    base = ready_model(cfg).eval()
    payload = {"config": copy.deepcopy(cfg)}
    student, new_cfg = make_student(base, payload, 4, source_hidden_dim=12, temperature=1.0)
    student.eval()
    values = batch(cfg, b=2)
    epsilon = torch.randn(2, 2, 4, 5)
    baseline = base.sample(values, steps=4, num_samples=2, source_noise=epsilon)
    candidate = student.sample(values, num_samples=2, source_noise=epsilon)
    torch.testing.assert_close(candidate["latent"], baseline["latent"])
    assert candidate["nfe"] == 8 and base.sample(values, num_samples=2, source_noise=epsilon)["nfe"] == 32
    assert new_cfg["model"]["flow_source_mode"] == "conditional_gaussian"
    assert "source" in student.checkpoint_contract() and "source" not in base.checkpoint_contract()


def test_student_checkpoint_strict_reload(tmp_path):
    cfg = config()
    base = ready_model(cfg)
    student, new_cfg = make_student(base, {"config": cfg}, 4, source_hidden_dim=12, temperature=1.0)
    path = tmp_path / "student.pt"
    torch.save({"model_version": student.MODEL_VERSION, "carswm_contract": student.checkpoint_contract(),
                "config": new_cfg, "model": student.state_dict()}, path)
    restored, _ = load_latent_checkpoint(path)
    assert restored.source_mode == "conditional_gaussian"
    torch.testing.assert_close(restored.source_model.network[-1].weight,
                               student.source_model.network[-1].weight)


def test_few_step_distillation_updates_source_and_velocity():
    cfg = config()
    base = ready_model(cfg).eval()
    student, _ = make_student(base, {"config": cfg}, 2, source_hidden_dim=12, temperature=1.0)
    values = batch(cfg, b=2)
    encoded = student.encode_conditions(values)
    source = student.source_from_noise(torch.randn(2, 4, 5), encoded)
    endpoint = heun_rollout(student.velocity, source, encoded, 2)
    with torch.no_grad():
        reference = heun_rollout(base.velocity, source.detach(), base.encode_conditions(values), 4)
    loss = (endpoint-reference).square().mean() + student.source_model.kl_per_sample(
        encoded["source_condition"]).mean()
    loss.backward()
    assert student.source_model.network[-1].weight.grad is not None
    assert student.velocity_head[-1].weight.grad is not None
