"""Inference caches preserve sampled futures and never survive a request."""
import pytest
import torch

from test_latent_contact_world_model import batch, config, ready_model
from model.pinn_model.latent_pretrained import normalizer_envelope
from train.nomalizer import Normalizer


@pytest.mark.parametrize("source_mode", ["gaussian", "conditional_gaussian"])
@pytest.mark.parametrize("solver", ["heun", "euler"])
def test_cached_sampling_matches_original_with_masked_actions(source_mode, solver):
    cfg = config()
    cfg["model"].update(flow_source_mode=source_mode, flow_layers=2)
    model = ready_model(cfg).eval()
    if source_mode == "conditional_gaussian":
        # Exercise trained mean/std, not just the N(0,I) initialization.
        torch.nn.init.normal_(model.source_model.network[-1].weight, std=0.01)
    values = batch(cfg, b=2)
    values["action_mask"][0, -1] = False
    noise = torch.randn(2, 3, 4, 5)
    reference = model.sample(values, num_samples=3, steps=4, solver=solver, source_noise=noise,
                             cache_condition_kv=False, cache_time_embeddings=False)
    cached = model.sample(values, num_samples=3, steps=4, solver=solver, source_noise=noise)
    for key in ("latent", "q_pred", "tau_pred", "contact_probability"):
        torch.testing.assert_close(cached[key], reference[key], atol=3e-6, rtol=1e-5)
    assert reference["nfe"] == cached["nfe"]


def test_cache_built_once_per_block_and_refreshed_on_next_request(monkeypatch):
    cfg = config()
    cfg["model"]["flow_layers"] = 2
    model = ready_model(cfg).eval()
    calls = []
    for block in model.flow_blocks:
        original = block.prepare_condition_kv
        def capture(history, action, original=original):
            calls.append(history.shape[0])
            return original(history, action)
        monkeypatch.setattr(block, "prepare_condition_kv", capture)
    values = batch(cfg, b=2)
    noise = torch.randn(2, 3, 4, 5)
    model.sample(values, num_samples=3, steps=4, source_noise=noise)
    assert calls == [2, 2]  # Project B conditions, not B * samples, once per block.
    values["tau"] += 0.3
    fresh = model.sample(values, num_samples=3, steps=4, source_noise=noise)
    assert calls == [2, 2, 2, 2]
    uncached = model.sample(values, num_samples=3, steps=4, source_noise=noise,
                            cache_condition_kv=False, cache_time_embeddings=False)
    torch.testing.assert_close(fresh["latent"], uncached["latent"], atol=3e-6, rtol=1e-5)


def test_sampling_cache_is_eval_only_and_keeps_training_path():
    cfg = config()
    model = ready_model(cfg).train()
    values = batch(cfg)
    with pytest.raises(ValueError, match="eval"):
        model.sample(values, cache_condition_kv=True)
    with pytest.raises(ValueError, match="eval"):
        model.sample(values, cache_time_embeddings=True)
    model.sample(values, steps=1)  # Defaults fall back while training.
    model(values, flow_time=0.5)["flow_velocity_pred"].square().mean().backward()
    assert model.velocity_head[-1].weight.grad is not None


def test_compile_core_has_no_graph_break_and_matches_eager():
    cfg = config()
    model = ready_model(cfg).eval()
    values = batch(cfg, b=2)
    noise = torch.randn(2, 1, 4, 5)
    # Eager backend checks graph capture/interface; CUDA Inductor speed is a
    # target-machine benchmark, not established by this CPU semantic test.
    compiled = torch.compile(model.integrate_latent, backend="eager", fullgraph=True)
    eager = model.sample(values, steps=2, source_noise=noise)
    result = model.sample(values, steps=2, source_noise=noise, integration_fn=compiled)
    torch.testing.assert_close(result["latent"], eager["latent"])


def test_prepared_motion_normalizers_preserve_features_and_clear_on_setup_change():
    cfg = config(pretrained=True)
    model = ready_model(cfg).eval()
    stats = {key: {"mean": torch.randn(7), "std": torch.rand(7)+0.5}
             for key in ("q", "dq", "delta_q", "tau", "action")}
    model.wm_normalizer = normalizer_envelope(Normalizer(stats), cfg)
    pretrained_stats = {key: {"mean": torch.randn(7), "std": torch.rand(7)+0.5}
                        for key in ("q", "dq", "delta_q")}
    model.pretrained_normalizer = normalizer_envelope(Normalizer(pretrained_stats), cfg)
    values = batch(cfg, b=2)
    with torch.no_grad():
        reference = model.motion_features(values)
        model.prepare_runtime_normalizers()
        torch.testing.assert_close(model.motion_features(values), reference, rtol=0, atol=0)
    assert not any("runtime_normalizer" in key for key in model.state_dict())
    model.to("cpu")
    assert model.runtime_normalizer is None and model._runtime_pretrained_normalizer is None
    model.prepare_runtime_normalizers()
    model.train()
    assert model.runtime_normalizer is None and model._runtime_pretrained_normalizer is None
