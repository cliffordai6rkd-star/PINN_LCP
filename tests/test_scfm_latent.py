"""SCFM math, isolation and resumable CLI tests without a robot or LeRobot data."""
import math
import json
import sys

import pytest
import torch
import yaml

from model.pinn_model.latent_contact_world_model import load_latent_checkpoint
from model.pinn_model.scfm_distillation import (
    SCFMSettings, ShortcutSchedule, dual_target, frozen_snapshot, sample_schedule, scfm_loss, update_flow_ema,
)
from scripts.posttrain_inverse_gaussian_latent import make_student
from scripts import posttrain_scfm_latent as script
from test_latent_contact_world_model import batch, config, ready_model


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_dual_target_branches_nonuniform_intervals_and_stop_gradient():
    state = torch.zeros(2, 1, 1, requires_grad=True)
    times = torch.tensor([[0., .2, .7], [.1, .5, .7]])
    mask = torch.tensor([True, False])
    parameter = torch.tensor(2., requires_grad=True)
    seen = []

    def teacher(z, t, encoded):
        assert len(z) == 1 and encoded["history"].item() == 10
        return z + parameter

    def fast(z, t, encoded):
        assert len(z) == 1 and encoded["history"].item() == 20
        return z + 4

    def slow(z, t, encoded):
        seen.append(z.clone())
        return z + t[:, None, None]

    result = dual_target(state, times, mask, {"history": torch.tensor([10, 20])}, teacher, fast, slow)
    torch.testing.assert_close(seen[0], torch.tensor([[[.4]], [[1.6]]]))
    expected = torch.tensor([[[(.2*2+.5*.6)/.7]], [[(.4*4+.2*2.1)/.6]]])
    torch.testing.assert_close(result, expected)
    # The average predicts the endpoint of the two Euler steps exactly.
    torch.testing.assert_close(state.detach() + (times[:, 2]-times[:, 0])[:, None, None]*result,
                               torch.tensor([[[.7]], [[2.02]]]))
    assert not result.requires_grad and parameter.grad is None and state.grad is None


def test_target_fixed_point_and_curved_field_is_not_instantaneous_matching():
    state = torch.ones(2, 1, 1)
    times = torch.tensor([[0., .25, .5], [.25, .5, 1.]])
    mask = torch.tensor([True, False])
    constant = lambda z, t, e: torch.full_like(z, 3.)
    target = dual_target(state, times, mask, {}, constant, constant, constant)
    torch.testing.assert_close(target, torch.full_like(state, 3.))
    curve = lambda z, t, e: z
    target = dual_target(state, times, mask, {}, curve, curve, curve)
    assert (target > curve(state, times[:, 0], {})).all()


@pytest.mark.parametrize("anchor_ratio", [0., .2, 1.])
@pytest.mark.parametrize("size", [1, 17])
def test_schedule_repeatable_and_valid_with_anchors(anchor_ratio, size):
    settings = SCFMSettings(anchor_ratio=anchor_ratio, shift_min=2.5, shift_max=4.5)
    first = sample_schedule(size, settings, torch.Generator().manual_seed(8))
    second = sample_schedule(size, settings, torch.Generator().manual_seed(8))
    torch.testing.assert_close(first.times, second.times)
    assert torch.equal(first.teacher, second.teacher)
    assert (first.times >= 0).all() and (first.times <= 1).all()
    assert (first.times[~first.anchor, 1:] > first.times[~first.anchor, :-1]).all()
    assert first.anchor.sum() == math.ceil(size*anchor_ratio)


def test_shift_is_applied_to_noise_sigma_before_reversing_time():
    settings = SCFMSettings(teacher_ratio=1., teacher_min_steps=4, teacher_max_steps=4,
                            shift_min=3., shift_max=3.)
    schedule = sample_schedule(50, settings, torch.Generator().manual_seed(2))
    first_pair = schedule.times[schedule.times[:, 0] < .01]
    assert len(first_pair) > 0
    # Noise sigma .75 -> 3*.75/(1+2*.75)=.9 -> WM time .1
    torch.testing.assert_close(first_pair[0], torch.tensor([1e-5, .1, .25]), atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("values", [dict(teacher_min_steps=3), dict(teacher_max_steps=33),
                                   dict(fast_ema_decay=1.), dict(shift_min=0.), dict(teacher_ratio=1.1),
                                   dict(learning_rate=float('nan')), dict(steps=0)])
def test_invalid_settings_rejected(values):
    with pytest.raises(ValueError):
        SCFMSettings(**values).validate()


@pytest.mark.parametrize("anchor_ratio", [0., .25, 1.])
def test_training_only_changes_flow_and_ema_formula(anchor_ratio):
    cfg = config()
    cfg["model"]["dropout"] = .5
    teacher = ready_model(cfg).eval().requires_grad_(False)
    student, _ = make_student(teacher, {"config": cfg}, 4, source_hidden_dim=12,
                              temperature=1., mode="distill-only")
    fast, slow = frozen_snapshot(student), frozen_snapshot(student)
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    values = batch(cfg, b=4)
    with torch.no_grad():
        encoded = teacher.encode_conditions(values)
        target = teacher.target_latent(encoded["_prepared_batch"])
    schedule = sample_schedule(4, SCFMSettings(anchor_ratio=anchor_ratio), torch.Generator().manual_seed(9))
    loss, stats = scfm_loss(student, teacher, fast, slow, target, torch.randn_like(target), encoded, schedule)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=.01)
    loss.backward()
    optimizer.step()
    assert sum(stats[key] for key in ("teacher_windows", "self_windows", "anchor_windows")) == 4
    changed = []
    prefixes = tuple(name+"." for name in student.FLOW_MODULES if name != "source_model")
    for name, p in student.named_parameters():
        if name.startswith(prefixes):
            changed.append(not torch.equal(p, before[name]))
        else:
            torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            assert p.grad is None
    assert any(changed)
    assert all(p.grad is None for model in (teacher, fast, slow) for p in model.parameters())
    update_flow_ema(fast, student, 0.)
    update_flow_ema(slow, student, .9)
    for name, p in student.named_parameters():
        torch.testing.assert_close(dict(fast.named_parameters())[name], p)
        expected = before[name]*.9+p*.1 if name.startswith(prefixes) else before[name]
        torch.testing.assert_close(dict(slow.named_parameters())[name], expected)
    assert not any(module.training for model in (student, fast, slow) for module in model.modules())


def test_bfloat_target_velocity_is_converted_to_state_dtype():
    state = torch.ones(2, 1, 1)
    velocity = lambda z, t, e: z.to(torch.bfloat16)
    target = dual_target(state, torch.tensor([[0., .5, 1.]]).repeat(2, 1),
                         torch.tensor([True, False]), {}, velocity, velocity, velocity)
    assert target.dtype == torch.float32 and torch.isfinite(target).all()


def cli_fixture(tmp_path, monkeypatch):
    cfg = config()
    cfg["model"].update(flow_inference_steps=16, flow_solver="heun")
    teacher = ready_model(cfg).eval()
    normalizer = {"stats": {key: {"mean": torch.zeros(2), "std": torch.ones(2)} for key in ("q", "tau")}}
    checkpoint = tmp_path / "base.pt"
    torch.save({"model_version": teacher.MODEL_VERSION, "carswm_contract": teacher.checkpoint_contract(),
                "config": cfg, "model": teacher.state_dict(), "normalizer": normalizer,
                "latent_training": {"data_contract": {"train_indices_sha256": script.index_hash([0, 1, 2, 3]),
                                                        "val_indices_sha256": script.index_hash([4, 5])}}}, checkpoint)
    recorded = batch(cfg, b=6)
    accessed = []

    class RecordedConditions(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 6

        def __getitem__(self, index):
            accessed.append(index)
            return {key: value[index] for key, value in recorded.items()}

        def set_normalizer(self, normalizer):
            pass

        batch_collate = staticmethod(torch.utils.data.default_collate)

    monkeypatch.setattr(script, "LatentContactWorldModelDataset", RecordedConditions)
    monkeypatch.setattr(script, "episode_train_indices", lambda d, c: ([0, 1, 2, 3], script.index_hash([0, 1, 2, 3])))
    settings = SCFMSettings(updates=2, batch_size=2, teacher_min_steps=4, teacher_max_steps=4,
                            reference_steps=4, validation_samples=3, validation_batches=1,
                            save_every=1, validate_every=1, few_shot_windows=3)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"scfm": settings.as_dict()}))
    return cfg, teacher, checkpoint, path, accessed


def test_cli_exports_gaussian_euler_and_resumes_identically(tmp_path, monkeypatch):
    cfg, teacher, checkpoint, configuration, accessed = cli_fixture(tmp_path, monkeypatch)
    original_hash = script.file_hash(checkpoint)
    full = tmp_path/"full.pt"
    resumed = tmp_path/"resumed.pt"
    common = ["scfm", "--base-checkpoint", str(checkpoint), "--config", str(configuration), "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", [*common, "--output", str(full)])
    script.main()
    monkeypatch.setattr(sys, "argv", [*common, "--output", str(resumed), "--updates", "1"])
    script.main()
    monkeypatch.setattr(sys, "argv", [*common, "--output", str(resumed), "--resume", str(resumed)])
    script.main()
    restored, payload = load_latent_checkpoint(resumed)
    complete, complete_payload = load_latent_checkpoint(full)
    assert script.file_hash(checkpoint) == original_hash
    assert restored.source_mode == "gaussian" and restored.flow_solver == "euler"
    assert restored.sample(batch(cfg, b=1))["nfe"] == 4
    for name, p in restored.named_parameters():
        torch.testing.assert_close(p, dict(complete.named_parameters())[name], rtol=0, atol=0)
        if name.startswith(tuple(n+"." for n in (*teacher.CONDITION_MODULES, *teacher.CODEC_MODULES))):
            torch.testing.assert_close(p, dict(teacher.named_parameters())[name], rtol=0, atol=0)
    metadata = payload["scfm_posttrain"]
    assert metadata["completed_updates"] == 2
    assert "inverse_gaussian_posttrain" not in payload
    assert set(metadata["selected_train_indices"]) <= {0, 1, 2, 3}
    assert metadata["initial_validation"]["variants"]["teacher_fine"]["nfe"] == 8
    for name, metrics in metadata["last_validation"]["variants"].items():
        for key, value in metrics.items():
            expected = complete_payload["scfm_posttrain"]["last_validation"]["variants"][name][key]
            if isinstance(value, float):
                assert value == pytest.approx(expected, abs=1e-6)
            else:
                assert value == expected
    assert set(accessed) <= set(range(6)) and {4, 5} <= set(accessed)
    report = tmp_path/"eval.json"
    monkeypatch.setattr(sys, "argv", [*common, "--output", str(report), "--resume", str(resumed), "--evaluate-only"])
    script.main()
    assert report.exists() and script.file_hash(checkpoint) == original_hash
    # Exercise the existing runtime/benchmark on the actual exported artifact.
    # It must inherit Euler and must not label the student's reference as teacher.
    from scripts import benchmark_latent_cwm as benchmark
    condition = tmp_path/"condition.pt"
    torch.save(batch(cfg, b=1), condition)
    latency = tmp_path/"latency.json"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--checkpoint", str(resumed), "--condition", str(condition),
                                     "--steps", "4", "--warmup", "0", "--repeats", "1", "--device", "cpu",
                                     "--output", str(latency)])
    benchmark.main()
    runtime = json.loads(latency.read_text())
    assert runtime["solver"] == "euler" and runtime["step_sweep"][0]["nfe"] == 4
    assert "NOT the original teacher" in runtime["reference"]


def test_cli_rejects_changed_validation_split(tmp_path, monkeypatch):
    _, _, checkpoint, configuration, _ = cli_fixture(tmp_path, monkeypatch)
    payload = torch.load(checkpoint, weights_only=False)
    payload["latent_training"]["data_contract"]["val_indices_sha256"] = "different"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(sys, "argv", ["scfm", "--base-checkpoint", str(checkpoint), "--config", str(configuration),
                                     "--output", str(tmp_path/"bad.pt"), "--device", "cpu"])
    with pytest.raises(ValueError, match="episode split"):
        script.main()
