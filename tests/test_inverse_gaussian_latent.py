"""Focused source/inversion checks without LeRobot data or robot hardware."""
import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from model.pinn_model.inverse_gaussian_source import ConditionalGaussianLatentSource, invert_heun
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from scripts.posttrain_inverse_gaussian_latent import (
    configure_training_stage, flow_distillation_losses, heun_rollout, make_student, resolve_training_mode,
)
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
    student, _ = make_student(base, {"config": cfg}, 2, source_hidden_dim=12, temperature=1.0, mode="joint")
    configure_training_stage(student, "joint")
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


def test_shallower_student_transfers_only_prefix_blocks_and_reloads(tmp_path):
    cfg = config()
    cfg["model"]["flow_layers"] = 4
    base = ready_model(cfg)
    student, new_cfg = make_student(base, {"config": cfg}, 4, source_hidden_dim=12,
                                    temperature=1.0, flow_layers=2, mode="distill-only")
    assert len(student.flow_blocks) == 2
    for index in range(2):
        for key, value in student.flow_blocks[index].state_dict().items():
            torch.testing.assert_close(value, base.flow_blocks[index].state_dict()[key])
    with pytest.raises(ValueError, match="teacher depth"):
        make_student(base, {"config": cfg}, 4, source_hidden_dim=12, temperature=1.0, flow_layers=5)
    with pytest.raises(ValueError, match="preserve.*depth"):
        make_student(base, {"config": cfg}, 4, source_hidden_dim=12, temperature=1.0, flow_layers=2)
    path = tmp_path / "shallow.pt"
    torch.save({"model_version": student.MODEL_VERSION, "carswm_contract": student.checkpoint_contract(),
                "config": new_cfg, "model": student.state_dict()}, path)
    restored, _ = load_latent_checkpoint(path)
    assert len(restored.flow_blocks) == 2
    assert restored.source_mode == "gaussian"


@pytest.mark.parametrize("mode,counts", [("source-only", (1000, 0)), ("distill-only", (0, 1000)), ("joint", (1000, 1000))])
def test_training_mode_defaults_do_not_mix_independent_experiments(mode, counts):
    args = SimpleNamespace(mode=mode, source_updates=None, joint_updates=None, teacher_steps=None, temperature=1.0)
    resolve_training_mode(args)
    assert (args.source_updates, args.joint_updates) == counts


@pytest.mark.parametrize("mode,source,flow", [("source-only", 1, 1), ("distill-only", 1, 1), ("joint", 1, 0)])
def test_training_mode_rejects_crossed_update_budgets(mode, source, flow):
    args = SimpleNamespace(mode=mode, source_updates=source, joint_updates=flow, teacher_steps=None, temperature=1.0)
    with pytest.raises(ValueError, match="updates"):
        resolve_training_mode(args)


def test_source_only_optimizer_cannot_modify_flow_conditions_or_codec():
    cfg = config()
    cfg["model"]["dropout"] = 0.5
    base = ready_model(cfg).eval()
    student, _ = make_student(base, {"config": cfg}, 4, source_hidden_dim=12, temperature=1.0)
    before = {key: value.detach().clone() for key, value in student.state_dict().items() if torch.is_tensor(value)}
    with torch.no_grad():
        encoded = base.encode_conditions(batch(cfg, b=2))
    loss = student.source_model.nll_per_sample(encoded["source_condition"], torch.ones(2, 4, 5)).mean()
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=0.01)
    loss.backward()
    optimizer.step()
    changed = []
    for key, value in student.state_dict().items():
        if key in before:
            if key.startswith("source_model."):
                changed.append(not torch.equal(value, before[key]))
            else:
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(changed)
    assert all(p.grad is None for name, p in student.named_parameters() if not name.startswith("source_model."))
    assert not any(module.training for module in student.modules())


def test_independent_distillation_updates_only_flow_with_shared_standard_noise(tmp_path):
    cfg = config()
    cfg["model"]["dropout"] = 0.5
    base = ready_model(cfg).eval().requires_grad_(False)
    student, new_cfg = make_student(base, {"config": cfg}, 2, source_hidden_dim=12, temperature=1.0,
                                    mode="distill-only")
    assert student.source_mode == "gaussian"
    assert not any(key.startswith("source_model.") for key in student.state_dict())
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    with torch.no_grad():
        teacher_encoded = base.encode_conditions(batch(cfg, b=2), cache_condition_kv=True)
    encoded = {key: value for key, value in teacher_encoded.items() if key != "condition_kv_cache"}
    epsilon = torch.randn(2, 4, 5)
    source = student.source_from_noise(epsilon, encoded)
    torch.testing.assert_close(source, epsilon, rtol=0, atol=0)
    endpoint, velocity = flow_distillation_losses(student, base, encoded, teacher_encoded, source,
                                                  steps=2, teacher_steps=4)
    # The target map and repeated student queries have no dropout randomness.
    again = flow_distillation_losses(student, base, encoded, teacher_encoded, source, steps=2, teacher_steps=4)
    torch.testing.assert_close(endpoint, again[0], rtol=0, atol=0)
    torch.testing.assert_close(velocity, again[1], rtol=0, atol=0)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=0.01)
    (endpoint+velocity).backward()
    optimizer.step()
    changes = []
    for name, p in student.named_parameters():
        if name.startswith(student.FLOW_MODULES):
            changes.append(not torch.equal(p, before[name]))
        else:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p.grad is None
    assert any(changes) and all(p.grad is None for p in base.parameters())
    path = tmp_path / "distill_only.pt"
    torch.save({"model_version": student.MODEL_VERSION, "carswm_contract": student.checkpoint_contract(),
                "config": new_cfg, "model": student.state_dict()}, path)
    restored, _ = load_latent_checkpoint(path)
    assert restored.source_mode == "gaussian"


@pytest.mark.parametrize("mode", ["source-only", "distill-only"])
def test_cli_exports_independent_experiments_without_running_other_route(tmp_path, monkeypatch, mode):
    from scripts import posttrain_inverse_gaussian_latent as script

    cfg = config()
    base = ready_model(cfg).eval()
    before = {key: value.clone() for key, value in base.state_dict().items() if torch.is_tensor(value)}
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_version": base.MODEL_VERSION, "carswm_contract": base.checkpoint_contract(),
                "config": cfg, "model": base.state_dict(),
                "normalizer": {"stats": {"q": {"mean": [0., 0.], "std": [1., 1.]}}},
                "latent_training": {"data_contract": {"train_indices_sha256": "test_split"}}}, checkpoint)

    class RecordedConditions(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            self.values = batch(cfg, b=2)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {key: value[index] for key, value in self.values.items()}

        def set_normalizer(self, normalizer):
            pass

        batch_collate = staticmethod(torch.utils.data.default_collate)

    monkeypatch.setattr(script, "LatentContactWorldModelDataset", RecordedConditions)
    monkeypatch.setattr(script, "episode_train_indices", lambda dataset, config: ([0, 1], "test_split"))

    def forbidden(*args, **kwargs):
        raise AssertionError("independent mode executed the other experiment")

    if mode == "distill-only":
        monkeypatch.setattr(script, "invert_heun", forbidden)
        monkeypatch.setattr(ConditionalGaussianLatentSource, "nll_per_sample", forbidden)
        monkeypatch.setattr(ConditionalGaussianLatentSource, "kl_per_sample", forbidden)
        budget = ["--flow-updates", "1", "--teacher-steps", "4"]
    else:
        monkeypatch.setattr(script, "flow_distillation_losses", forbidden)
        monkeypatch.setattr(script, "invert_heun", lambda velocity, target, encoded, **kwargs:
                            (target+1, torch.zeros(len(target))))
        budget = ["--source-updates", "1"]
    output = tmp_path / "candidate.pt"
    monkeypatch.setattr(sys, "argv", ["posttrain", "--mode", mode, "--base-checkpoint", str(checkpoint),
                                     "--output", str(output), "--steps", "2", "--batch-size", "2",
                                     "--source-hidden-dim", "12", "--device", "cpu", *budget])
    script.main()
    restored, payload = load_latent_checkpoint(output)
    metadata = payload["inverse_gaussian_posttrain"]
    assert metadata["mode"] == mode and metadata["nfe"] == 4
    if mode == "source-only":
        assert metadata["distillation_updates"] == 0 and metadata["teacher_source_policy"] is None
        assert metadata["flow_transfer"] == "unchanged_teacher_weights"
        for key, value in restored.state_dict().items():
            if key in before:
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    else:
        assert restored.source_mode == "gaussian" and metadata["source_updates"] == 0
        assert metadata["inverse_solver"] is None and metadata["metrics"]["maximum_cycle_rmse"] is None
        assert metadata["teacher_source_policy"] == "shared_standard_gaussian_epsilon"
        for name in (*base.CONDITION_MODULES, *base.CODEC_MODULES):
            for key, value in restored.state_dict().items():
                if key in before and key.startswith(name+"."):
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
