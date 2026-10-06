"""Conditional codec semantics and real frozen/adapter SFT updates and resume."""
import copy
import json

import pytest
import torch

from test_contact_world_model_dataset import make_dataset
from test_latent_contact_world_model import batch, config, threads
from test_latent_contact_world_model_training import training_case
from model.pinn_model.latent_contact_world_model import (
    CONDITIONAL_MODEL_VERSION, LatentContactWorldModel, load_latent_checkpoint,
)
from model.pinn_model.latent_pretrained import convert_scale
from train.latent_contact_world_model_loss import LatentContactWorldModelLoss
from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer


def conditional_config(mode="frozen"):
    cfg = config()
    cfg["model"].update(family=CONDITIONAL_MODEL_VERSION, local_state_dim=4, flow_adapter_rank=2)
    cfg["sft"] = {"flow_mode": mode, "lambda_fm": 0.7, "lambda_reconstruction": 1.3}
    return cfg


def envelope(cfg, offset=0., scale=1.):
    return {"normalize_mode": "gaussian", "normalize_lowdim_keys": ["q", "dq", "delta_q", "tau", "action"],
            "eps": 1e-6, "stats": {
                key: {"mean": torch.full((dim,), offset), "std": torch.full((dim,), scale)}
                for key, dim in [(k, cfg["model"]["joint_dim"]) for k in ("q", "dq", "delta_q", "tau")]
                + [("action", cfg["model"]["action_dim"])]}}


def finalized(cfg):
    model = LatentContactWorldModel(cfg)
    model.wm_normalizer = envelope(cfg)
    model.snapshot_target_encoder()
    model.codec_ready.fill_(True)
    model.codec_snapshot = "synthetic_frozen_v2"
    model.set_stage("flow")
    return model


def save_model(model, path):
    torch.save({"config": model.config, "model_version": model.MODEL_VERSION,
                "carswm_contract": model.checkpoint_contract(), "model": model.state_dict()}, path)


def test_conditional_codec_learns_both_paths_and_freezes_target_snapshot():
    cfg = conditional_config()
    model = LatentContactWorldModel(cfg)
    values = batch(cfg)
    out = model(values)
    loss, _ = LatentContactWorldModelLoss(cfg)(out, values)
    loss.backward()
    for name in ("future_encoder", "local_state_encoder", "q_head", "tau_head", "contact_head"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for name in (*model.CONDITION_MODULES, *model.FLOW_MODULES, "target_state_encoder")
               for p in getattr(model, name).parameters())
    model.wm_normalizer = envelope(cfg)
    model.snapshot_target_encoder()
    for a, b in zip(model.local_state_encoder.parameters(), model.target_state_encoder.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.codec_ready.fill_(True)
    model.set_stage("flow")
    model.train()
    assert all(not getattr(model, name).training for name in model.CODEC_MODULES)
    z = model.target_latent(values)
    assert not z.requires_grad


@pytest.mark.parametrize("stride", [False, True])
def test_relative_future_and_residual_anchor_on_normalized_scale(stride):
    cfg = conditional_config()
    cfg["train"]["downsample"] = stride
    model = finalized(cfg)
    values = batch(cfg)
    prepared = model.prepare_batch(values)
    torch.testing.assert_close(model.future_input(values)[..., :model.joint_dim],
                               prepared["q_future"]-values["q"][:, -1:])
    for parameter in model.q_head.parameters():
        parameter.data.zero_()
    raw = torch.randn(3, 2, model.future_horizon, model.latent_dim)
    out = model.decode(raw, values)
    torch.testing.assert_close(out["q_pred"], values["q"][:, -1, None, None].expand_as(out["q_pred"]))
    physical = convert_scale("q", out["q_pred"], model.wm_normalizer, inverse=True)
    anchor = convert_scale("q", values["q"][:, -1], model.wm_normalizer, inverse=True)
    torch.testing.assert_close(physical, anchor[:, None, None].expand_as(physical))
    with pytest.raises(ValueError, match="observed history"):
        model.decode(raw)


@pytest.mark.parametrize("mode", ["frozen", "adapter"])
def test_sft_gradients_updates_and_fixed_target_coordinates(mode):
    cfg = conditional_config(mode)
    model = finalized(cfg)
    model.set_stage("sft")
    model.train()
    values = batch(cfg)
    target_before = model.target_latent(values).clone()
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    stats = (model.latent_mean.clone(), model.latent_std.clone())
    out = model(values, flow_time=0.4, source_noise=torch.zeros_like(target_before))
    assert not out["flow_target_latent"].requires_grad
    # The source of reconstruction is the true frozen target, separate from sampled futures.
    expected_rec = model.decode(target_before*model.latent_std+model.latent_mean, values)
    for key in ("q_pred", "tau_pred", "contact_logits"):
        torch.testing.assert_close(out["reconstruction"][key], expected_rec[key])
    calculator = LatentContactWorldModelLoss(cfg)
    fm, _ = calculator.flow_loss(out, values)
    rec, _ = calculator.codec_loss(out["reconstruction"], out["_prepared_batch"])
    total, metrics = calculator(out, values)
    torch.testing.assert_close(total, 0.7*fm+1.3*rec)
    assert "sft_q_mse" in metrics and "latent_fm_loss" in metrics
    optimizer = torch.optim.AdamW(model.parameter_groups(), lr=0.01)
    total.backward()
    for name in (*model.CONDITION_MODULES, "local_state_encoder", "q_head", "tau_head", "contact_head"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for name in (*model.FLOW_MODULES, "future_encoder", "target_state_encoder")
               for p in getattr(model, name).parameters())
    assert all(not getattr(model, name).training for name in model.FLOW_MODULES)
    included = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in included for name in model.FLOW_MODULES for p in getattr(model, name).parameters())
    if mode == "adapter":
        assert model.flow_adapter.up.weight.grad.abs().sum() > 0
    else:
        assert all(p.grad is None and id(p) not in included for p in model.flow_adapter.parameters())
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    torch.testing.assert_close(model.target_latent(values), target_before, rtol=0, atol=0)
    torch.testing.assert_close(model.latent_mean, stats[0], rtol=0, atol=0)
    torch.testing.assert_close(model.latent_std, stats[1], rtol=0, atol=0)


def test_sft_branch_loss_isolation():
    cfg = conditional_config()
    model = finalized(cfg)
    model.set_stage("sft")
    values = batch(cfg)
    out = model(values)
    fm, _ = LatentContactWorldModelLoss(cfg).flow_loss(out, values)
    fm.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.motion_encoder.parameters())
    assert all(p.grad is None for name in ("local_state_encoder", "q_head", "tau_head", "contact_head")
               for p in getattr(model, name).parameters())
    model.zero_grad(set_to_none=True)
    out = model(values)
    rec, _ = LatentContactWorldModelLoss(cfg).codec_loss(out["reconstruction"], out["_prepared_batch"])
    rec.backward()
    assert all(p.grad is None for name in (*model.CONDITION_MODULES, *model.FLOW_MODULES, "future_encoder", "target_state_encoder")
               for p in getattr(model, name).parameters())


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("train_motion", [False, True])
def test_sft_motion_freezing_policy_applies_to_scratch_and_next(pretrained, train_motion):
    cfg = config(pretrained=pretrained)
    cfg["model"].update(family=CONDITIONAL_MODEL_VERSION, local_state_dim=4, flow_adapter_rank=2)
    cfg["sft"] = {"train_motion_encoder": train_motion}
    model = finalized(cfg)
    assert all(p.requires_grad == (not pretrained) for p in model.motion_encoder.parameters())
    model.set_stage("sft")
    model.train()
    assert model.motion_encoder.training == train_motion
    assert all(p.requires_grad == train_motion for p in model.motion_encoder.parameters())


def test_transfer_preserves_source_scale_for_targets_and_target_scale_for_decoding(tmp_path):
    cfg = conditional_config()
    source = finalized(cfg)
    source.wm_normalizer = envelope(cfg, 7., 3.)
    source.snapshot_target_encoder()
    source.latent_mean.fill_(2.)
    source.latent_std.fill_(4.)
    path = tmp_path/"source.pt"
    save_model(source, path)
    target = LatentContactWorldModel(cfg)
    target.wm_normalizer = envelope(cfg, -2., 0.5)
    target.initialize_sft(path)
    physical = batch(cfg)
    def normalize(values, normalizer):
        out = dict(values)
        for key in ("q", "dq", "delta_q", "tau", "action", "q_future", "tau_future"):
            out[key] = convert_scale(key.removesuffix("_future"), values[key], normalizer)
        return out
    source_batch = normalize(physical, source.wm_normalizer)
    target_batch = normalize(physical, target.wm_normalizer)
    torch.testing.assert_close(target.target_latent(target_batch), source.target_latent(source_batch))
    for parameter in target.q_head.parameters():
        parameter.data.zero_()
    torch.testing.assert_close(target.codec_forward(target_batch)["q_pred"],
                               target_batch["q"][:, -1:].expand_as(target_batch["q_future"]))
    assert target.transfer_provenance["checkpoint"] == str(path)
    saved = tmp_path/"sft.pt"
    save_model(target, saved)
    path.unlink()
    restored, _ = load_latent_checkpoint(saved)
    torch.testing.assert_close(restored.target_latent(target_batch), target.target_latent(target_batch), rtol=0, atol=0)


def test_initial_adapter_is_zero_and_sampling_uses_local_history_without_future():
    cfg = conditional_config("adapter")
    model = finalized(cfg).eval()
    features = torch.randn(3, 4, model.hidden_dim)
    torch.testing.assert_close(model.flow_adapter(features), torch.zeros(3, 4, model.latent_dim))
    assert sum(p.numel() for p in model.flow_adapter.parameters()) == (model.hidden_dim+model.latent_dim)*2
    values = batch(cfg)
    condition = {key: values[key] for key in model.CONDITION_KEYS}
    noise = torch.randn(3, 2, 4, 5)
    out = model.sample(condition, num_samples=2, source_noise=noise, steps=2)
    decoded = model.decode(out["raw_latent"], condition)
    for key in ("q_pred", "tau_pred", "contact_logits"):
        torch.testing.assert_close(out[key], decoded[key])
    changed = dict(condition, action=condition["action"]+10.)
    action_out = model.sample(changed, num_samples=2, source_noise=noise, steps=2)
    assert not torch.equal(out["latent"], action_out["latent"])
    assert not torch.equal(out["q_pred"], action_out["q_pred"])
    zero_decoded = model.decode(torch.zeros_like(out["raw_latent"]), condition)
    assert not torch.equal(out["q_pred"], zero_decoded["q_pred"])


def test_transfer_rejects_v1_and_noninvertible_normalization(tmp_path):
    cfg = config()
    legacy = LatentContactWorldModel(cfg)
    legacy.codec_ready.fill_(True)
    legacy.set_stage("flow")
    path = tmp_path/"v1.pt"
    save_model(legacy, path)
    restored, _ = load_latent_checkpoint(path)
    assert not restored.conditional_codec
    v2cfg = conditional_config()
    target = LatentContactWorldModel(v2cfg)
    target.wm_normalizer = envelope(v2cfg)
    with pytest.raises(ValueError, match="finalized v2"):
        target.initialize_sft(path)
    source = finalized(v2cfg)
    save_model(source, path)
    target.wm_normalizer["normalize_mode"] = "quantile"
    with pytest.raises(ValueError, match="invertible"):
        target.initialize_sft(path)


def pretraining_config(make_dataset, output):
    cfg = training_case(make_dataset, output)
    cfg["model"].update(family=CONDITIONAL_MODEL_VERSION, local_state_dim=4, flow_adapter_rank=2)
    cfg["train"].update(max_optimizer_steps=2, probabilistic_validation={"enabled": False})
    return cfg


@pytest.mark.parametrize("mode", ["frozen", "adapter"])
def test_real_sft_resume_exact_and_self_contained(make_dataset, tmp_path, mode):
    pre_cfg = pretraining_config(make_dataset, tmp_path/"pre")
    pretrainer = LatentContactWorldModelTrainer(pre_cfg)
    pretrainer.train()
    codec = torch.load(tmp_path/"pre/codec.pt", weights_only=False)
    assert codec["model_version"] == "latent_codec_v2"
    assert "target_state_encoder" in codec["modules"]
    source = pretrainer.ckpt_dir/"latest.pt"
    cfg = copy.deepcopy(pre_cfg)
    cfg["train"].update(stage="sft", max_optimizer_steps=4, output_dir=str(tmp_path/"full"))
    cfg["sft"] = {"pretrained_checkpoint": str(source), "flow_mode": mode}
    full = LatentContactWorldModelTrainer(cfg)
    full.train()
    assert full.stage == "sft" and full.global_step == 4 and full.codec_step == 2
    assert (full.output_dir/"sft_initial_reconstruction.json").exists()
    assert full.final_validation["sft_reconstruction_loss"] >= 0
    for name in (*full.model.FLOW_MODULES, "future_encoder", "target_state_encoder"):
        for key, value in getattr(full.model, name).state_dict().items():
            torch.testing.assert_close(value, getattr(pretrainer.ema.model, name).state_dict()[key], rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(full.model.q_head.parameters(), pretrainer.model.q_head.parameters()))
    cfg["train"]["output_dir"] = str(tmp_path/"partial")
    partial = LatentContactWorldModelTrainer(cfg)
    partial.setup()
    partial.run_stage(stop_after_updates=1)
    source.unlink()
    cfg["train"]["resume_from"] = str(partial.output_dir)
    resumed = LatentContactWorldModelTrainer(cfg)
    resumed.train()
    for model_name in ("model", "ema"):
        a, b = getattr(full, model_name), getattr(resumed, model_name)
        if model_name == "ema":
            a, b = a.model, b.model
        for key, value in a.state_dict().items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    status = json.loads((resumed.output_dir/"status.json").read_text())
    assert status["stage"] == "sft" and status["sft_step"] == 4 and status["status"] == "complete"
    assert sorted(p.name for p in resumed.ckpt_dir.glob("step_*.pt")) == ["step_00000002.pt", "step_00000004.pt"]


def test_conditional_codec_reuse_and_transfer_with_different_horizons(make_dataset, tmp_path):
    cfg = pretraining_config(make_dataset, tmp_path/"pre")
    pretrainer = LatentContactWorldModelTrainer(cfg)
    pretrainer.train()
    cfg["train"].update(stage="flow", output_dir=str(tmp_path/"reuse"))
    cfg["codec"]["checkpoint_path"] = str(pretrainer.output_dir/"codec.pt")
    reuse = LatentContactWorldModelTrainer(cfg)
    reuse.train()
    assert reuse.model.codec_snapshot == pretrainer.model.codec_snapshot
    cfg["train"].update(stage="sft", output_dir=str(tmp_path/"target"))
    cfg["codec"]["checkpoint_path"] = None
    cfg["dataloader"].update(state_history_horizon=40, prediction_horizon=6, action_condition_horizon=6)
    cfg["sft"] = {"pretrained_checkpoint": str(pretrainer.ckpt_dir/"latest.pt")}
    target = LatentContactWorldModelTrainer(cfg)
    target.train()
    assert target.model.future_horizon == 6 and target.global_step == 2


@pytest.mark.parametrize("key", ["latent_dim", "flow_attention_heads"])
def test_transfer_rejects_semantically_incompatible_architecture(tmp_path, key):
    cfg = conditional_config()
    path = tmp_path/"pre.pt"
    save_model(finalized(cfg), path)
    bad = copy.deepcopy(cfg)
    bad["model"][key] *= 2
    target = LatentContactWorldModel(bad)
    target.wm_normalizer = envelope(bad)
    with pytest.raises(ValueError, match=key.removeprefix("flow_")):
        target.initialize_sft(path)


def test_validation_samples_each_task_and_reports_latent_ablations(make_dataset, tmp_path):
    cfg = pretraining_config(make_dataset, tmp_path/"ablation")
    cfg["model"]["flow_inference_steps"] = 2
    cfg["train"]["probabilistic_validation"] = {
        "enabled": True, "num_samples": 2, "max_batches": 1,
        "per_task_max_batches": 1, "latent_ablation": True,
    }
    trainer = LatentContactWorldModelTrainer(cfg)
    trainer.setup()
    trainer.run_stage()
    trainer.finalize_codec()
    values = next(iter(trainer.val_loader))
    b = values["q"].shape[0]
    # Ordered source batches must not spend the entire sampling budget on task 0.
    trainer.val_loader = [dict(values, task_index=torch.full((b,), task)) for task in (0, 0, 1, 1)]
    for parameter in trainer.model.q_head.parameters():
        parameter.data.zero_()
    trainer.ema_use_for_validation = False
    result = trainer.evaluate()
    assert result["task_0_evaluated_windows"] == b and result["task_1_evaluated_windows"] == b
    assert result["task_0_contact_0_support"] == b*trainer.model.future_horizon
    assert result["task_1_contact_0_support"] == b*trainer.model.future_horizon
    assert "task_0_latent_fm_mse" in result and "task_1_latent_fm_mse" in result
    for name in ("zero", "shuffled"):
        assert result[f"latent_{name}_tau_physical_rmse"] >= 0
        assert result[f"latent_{name}_contact_ce"] >= 0
        # An interface deliberately ignoring latent has identical q ablations.
        assert result[f"latent_{name}_q_physical_rmse"] == pytest.approx(result["q_physical_rmse"])


def test_sft_configs_match_source_architecture_and_current_dataset_locations():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]/"config/train_cfg/latent_wm"
    pre = yaml.safe_load((root/"nero_pretrain_all.yaml").read_text())
    assert len(pre["train_data"]["sources"]) == 4
    assert {source["name"] for source in pre["train_data"]["sources"]} == {
        "insert_usb", "push_button", "cucumber_peeling", "wipe_board",
    }
    for task in ("peel_cucumber", "erase_board"):
        target = yaml.safe_load((root/f"xarm_{task}_sft.yaml").read_text())
        for key in ("family", "joint_dim", "action_dim", "hidden_dim", "latent_dim", "local_state_dim", "flow_adapter_rank",
                    "flow_layers", "flow_attention_heads", "flow_ffn_multiplier"):
            assert target["model"][key] == pre["model"][key]
        assert target["train"]["stage"] == "sft" and target["sft"]["flow_mode"] == "frozen"
        assert target["codec"]["checkpoint_path"] is None
        assert target["train_data"]["sources"][0]["root"] == f"data/xarm_co_v3/{task}_v3"
        assert target["train"]["probabilistic_validation"]["latent_ablation"]
        assert target["dataloader"]["filters"] == pre["dataloader"]["filters"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 smoke requires a GPU")
def test_cuda_bfloat16_pretraining_sft_and_sampling(tmp_path):
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]/"config/train_cfg/latent_wm"
    cfg = yaml.safe_load((root/"nero_pretrain_all.yaml").read_text())
    source = LatentContactWorldModel(cfg).cuda()
    source.wm_normalizer = envelope(cfg)

    def values_for(model):
        return {key: value.cuda() for key, value in batch(model.config, b=2).items()}

    def update(model):
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameter_groups(), lr=1e-4)
        values = values_for(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = LatentContactWorldModelLoss(model.config)(model(values), values)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        return values

    update(source)  # Conditional codec reconstruction, including the local branch.
    source.snapshot_target_encoder()
    source.codec_ready.fill_(True)
    source.codec_snapshot = "synthetic_gpu_smoke"
    source.set_stage("flow")
    update(source)
    path = tmp_path/"gpu_source.pt"
    save_model(source, path)
    for mode in ("frozen", "adapter"):
        target_cfg = yaml.safe_load((root/"xarm_peel_cucumber_sft.yaml").read_text())
        target_cfg["sft"]["flow_mode"] = mode
        target = LatentContactWorldModel(target_cfg).cuda()
        target.wm_normalizer = envelope(target_cfg, offset=2., scale=0.5)
        target.initialize_sft(path)
        frozen = {name: p.clone() for name, p in target.named_parameters() if not p.requires_grad}
        values = update(target)
        for name, parameter in target.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        condition = {key: values[key] for key in target.CONDITION_KEYS}
        target.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = target.sample(condition, num_samples=2, steps=2)
        assert sampled["q_pred"].shape == (2, 2, 32, 7)
        assert all(torch.isfinite(sampled[key]).all() for key in ("q_pred", "tau_pred", "contact_probability"))
