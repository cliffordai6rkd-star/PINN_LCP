"""Deterministic prediction, temporal contracts, and complete trainer integration."""

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from data_process import contact_world_model_dataset as dataset_module
from model.pinn_model.deterministic_world_model import DeterministicRobotStateWorldModel
from train.trainer.deterministic_world_model_train import (
    DeterministicWorldModelLoss,
    DeterministicWorldModelTrainer,
    run_smoke_test,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return {
        "dataloader": {"state_history_horizon": 6, "prediction_horizon": 6,
                       "action_condition_horizon": 3, "high_fps": 100, "expert_fps": 25,
                       "action_start_offset": 1, "normalize_mode": "gaussian",
                       "normalize_lowdim_keys": ["q", "dq", "delta_q", "tau", "action"]},
        "model": {"inputs": ["q", "dq", "delta_q", "tau"], "outputs": ["q", "tau"],
                  "joint_dim": 2, "action_dim": 3, "hidden_dim": 8, "state_layers": 2,
                  "action_layers": 2, "decoder_layers": 2, "attention_heads": 2,
                  "ffn_multiplier": 2, "dropout": 0.1, "contact_state_count": 3},
        "contact_gate": {"class_weights": [1, 1, 1]},
        "loss": {"stream_weights": {"q": 1.0, "tau": 2.0}, "contact_weight": 0.1,
                 "use_importance_weight": True},
        "train": {"device": "cpu", "downsample": False, "num_workers": 0,
                  "batch_size": 2, "max_optimizer_steps": 1, "checkpoint_every_steps": 1,
                  "val_ratio": 0.5, "val_every": 1, "split_mode": "episode",
                  "ema": {"enabled": True, "decay": 0.9}, "wandb": {"enabled": False}},
    }


def batch(cfg, count=2):
    data, model = cfg["dataloader"], cfg["model"]
    result = {key: torch.randn(count, data["state_history_horizon"], model["joint_dim"])
              for key in model["inputs"]}
    result.update({f"{key}_future": torch.randn(count, data["prediction_horizon"], model["joint_dim"])
                   for key in model["outputs"]})
    result.update(action=torch.randn(count, data["action_condition_horizon"], model["action_dim"]),
                  action_mask=torch.ones(count, data["action_condition_horizon"], dtype=torch.bool),
                  contact_future=torch.randint(0, 3, (count, data["prediction_horizon"], 1)).float(),
                  importance_weight=torch.tensor([0.5, 2.0])[:count])
    return result


@pytest.mark.parametrize("downsample", [False, True, 3])
@pytest.mark.parametrize("outputs", [["q", "tau"], ["dq"], ["tau", "delta_q", "q"]])
def test_shapes_striding_and_backward(downsample, outputs):
    cfg = config()
    cfg["train"]["downsample"] = downsample
    cfg["model"]["outputs"] = outputs
    model = DeterministicRobotStateWorldModel(cfg)
    values = batch(cfg)
    values["history_indices"] = torch.arange(6)[None].expand(2, -1)
    values["future_indices"] = torch.arange(1, 7)[None].expand(2, -1)
    out = model(values)
    stride = 2 if downsample is True else int(downsample or 1)
    horizon = 6 // stride
    for key in outputs:
        assert out[f"{key}_pred"].shape == (2, horizon, 2)
    assert out["contact_logits"].shape == (2, horizon, 3)
    assert out["state_tokens"].shape == (2, 4 * horizon, 8)
    assert out["action_tokens"].shape == (2, 3, 8)
    prepared = out["_prepared_batch"]
    assert torch.equal(prepared["q"][:, -1], values["q"][:, -1])
    assert prepared["history_indices"][0].tolist() == list(range(stride - 1, 6, stride))
    assert prepared["future_indices"][0].tolist() == list(range(1, 7, stride))
    assert torch.equal(prepared["action"], values["action"])
    for key, value in model.prepare_batch(prepared).items():
        torch.testing.assert_close(value, prepared[key])
    loss, _ = DeterministicWorldModelLoss(cfg)(out, values)
    loss.backward()
    for module in (model.state_head, model.contact_head, model.action_encoder,
                   model.decoder_blocks[0].history_cross_attn, model.decoder_blocks[0].action_cross_attn,
                   *model.state_encoders.values()):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert model.future_query.grad.abs().sum() > 0
    model.eval()
    public = model.predict(values)
    for key in outputs:
        assert public[f"{key}_pred"].shape == (2, 6, 2)
        torch.testing.assert_close(public[f"{key}_pred"], model(values)[f"{key}_pred"].repeat_interleave(stride, 1))
    assert public["contact_logits"].shape == (2, 6, 3)


def test_eval_is_deterministic_target_independent_and_has_no_flow(monkeypatch):
    from model.pinn_model.contact_world_model import ContactWorldModel
    def forbidden(*args, **kwargs):
        raise AssertionError("baseline must not execute a flow model")
    monkeypatch.setattr(ContactWorldModel, "integrate_flow", forbidden)
    monkeypatch.setattr(ContactWorldModel, "forward", forbidden)
    cfg = config()
    model = DeterministicRobotStateWorldModel(cfg).eval()
    assert not isinstance(model, ContactWorldModel)
    assert not hasattr(model, "integrate_flow")
    assert "source_noise" not in inspect.signature(model.forward).parameters
    values = batch(cfg)
    first = model(values)
    conditions = {key: values[key] for key in model.CONDITION_KEYS}
    torch.manual_seed(12345)
    second = model(conditions)
    for key in ("q_pred", "tau_pred", "contact_logits"):
        assert torch.equal(first[key], second[key])
    for key in model.TARGET_KEYS:
        values[key].fill_(float("nan"))
    torch.testing.assert_close(model(values)["q_pred"], first["q_pred"])


def test_masked_actions_cannot_change_predictions():
    cfg = config()
    model = DeterministicRobotStateWorldModel(cfg).eval()
    values = batch(cfg)
    values["action_mask"][:, 1] = False
    first = model(values)
    values["action"][:, 1] = 10000
    second = model(values)
    for key in ("q_pred", "tau_pred", "contact_logits"):
        torch.testing.assert_close(first[key], second[key])
    values["action_mask"].zero_()
    with pytest.raises(ValueError, match="at least one valid action"):
        model(values)
    cfg["model"]["use_action_padding_mask"] = False
    assert torch.isfinite(DeterministicRobotStateWorldModel(cfg)(values)["q_pred"]).all()


@pytest.mark.parametrize("downsample", [False, True, 3])
def test_condition_encoder_and_preparation_match_wm(downsample):
    from model.pinn_model.contact_world_model import ContactWorldModel
    cfg = config()
    cfg["train"]["downsample"] = downsample
    cfg["model"].update(flow_layers=2, flow_attention_heads=2, flow_ffn_multiplier=2)
    baseline = DeterministicRobotStateWorldModel(cfg).eval()
    wm = ContactWorldModel(cfg).eval()
    for key in ("state_encoders", "modality_embeddings", "action_encoder", "history_pos_embedding",
                "action_pos_embedding", "state_token_norm", "action_token_norm"):
        getattr(baseline, key).load_state_dict(getattr(wm, key).state_dict())
    values = batch(cfg)
    expected, actual = wm.encode_conditions(values), baseline.encode_conditions(values)
    for key in ("state_tokens", "action_tokens", "condition_summary"):
        torch.testing.assert_close(actual[key], expected[key])
    for key in values:
        torch.testing.assert_close(actual["_prepared_batch"][key], expected["_prepared_batch"][key])


def test_weighted_regression_and_contact_ce_have_correct_batch_reduction():
    cfg = config()
    cfg["contact_gate"]["class_weights"] = [1, 2, 3]
    loss_fn = DeterministicWorldModelLoss(cfg)
    logits = torch.tensor([[[1., 0., -1.]], [[-1., 0., 1.]]], requires_grad=True)
    out = {"q_pred": torch.tensor([[[1., 1.]], [[3., 3.]]], requires_grad=True),
           "tau_pred": torch.tensor([[[2., 2.]], [[4., 4.]]], requires_grad=True),
           "contact_logits": logits}
    values = {"q_future": torch.zeros(2, 1, 2), "tau_future": torch.zeros(2, 1, 2),
              "contact_future": torch.tensor([[[0.]], [[2.]]]), "importance_weight": torch.tensor([2., 0.5])}
    state_ps = torch.tensor([1. + 2 * 4, 9. + 2 * 16])
    ce = F.cross_entropy(logits[:, 0], torch.tensor([0, 2]), weight=torch.tensor([1., 2., 3.]), reduction="none")
    actual, metrics = loss_fn(out, values)
    expected = ((state_ps + 0.1 * ce) * values["importance_weight"]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad.abs().sum() > 0
    assert metrics["q_mse"] == 5
    loss_fn.use_importance_weight = False
    torch.testing.assert_close(loss_fn(out, values)[0], (state_ps + 0.1 * ce).mean())
    loss_fn.use_importance_weight = True
    values["importance_weight"].zero_()
    assert loss_fn(out, values)[0] == 0


@pytest.mark.parametrize("bad_weight", [torch.tensor([-1., 1.]), torch.tensor([float("nan"), 1.]), torch.ones(2, 1)])
def test_invalid_importance_weights_rejected(bad_weight):
    cfg = config()
    values = batch(cfg)
    values["importance_weight"] = bad_weight
    with pytest.raises(ValueError, match="importance_weight"):
        DeterministicWorldModelLoss(cfg)(DeterministicRobotStateWorldModel(cfg)(values), values)


@pytest.mark.parametrize("downsample", [0, -1, 1.5, 4])
def test_invalid_temporal_stride_rejected(downsample):
    cfg = config()
    cfg["train"]["downsample"] = downsample
    with pytest.raises(ValueError):
        DeterministicRobotStateWorldModel(cfg)


@pytest.fixture
def lerobot_fixture(monkeypatch, tmp_path):
    # Two episodes with different state means expose normalizer/split leakage.
    size = 96
    rows = torch.arange(size).repeat(2)
    episode_shift = torch.arange(2).repeat_interleave(size).float() * 100
    values = (rows.float() + episode_shift)[:, None].repeat(1, 2)
    columns = {field: values.clone() for key, field in dataset_module.V3_STATE_FIELDS.items()
               if key != "action"}
    signal = torch.where(rows < 32, 0.0, torch.where(rows < 64, 0.5, 2.0))
    columns["observation.tau_ext"] = signal[:, None].repeat(1, 2)
    indices = rows // 4
    columns.update({"action.ee_pose": (indices.float() + episode_shift)[:, None].repeat(1, 3),
                    "timing.state_timestamp_ns": (rows + 1) * 10_000_000,
                    "timing.action_index": indices,
                    "timing.action_anchor_timestamp_ns": indices * 40_000_000 + 8_000_000})
    class Table:
        def with_format(self, *args, **kwargs):
            return self
        def __getitem__(self, key):
            return columns
    source = SimpleNamespace(hf_dataset=Table(), meta=SimpleNamespace(episodes=[
        {"episode_index": i, "dataset_from_index": i * size, "dataset_to_index": (i + 1) * size}
        for i in range(2)
    ]))
    monkeypatch.setattr(dataset_module, "_load_lerobot_dataset_class", lambda: lambda **kwargs: source)
    cfg = config()
    cfg["wm_v3_only"] = True
    cfg["dataloader"].update(root=str(tmp_path), repo_id="synthetic_v3", action_key="action.ee_pose",
                             high_timestamp_key="timing.state_timestamp_ns",
                             anchor_timestamp_key="timing.action_anchor_timestamp_ns")
    cfg["contact_gate"].update(enabled=True, label_mode="three_phase", metric="tau_ext_l1",
                               phase_label_mode="transition_band", consecutive_frames=1,
                               thresholds={"tau_ext_l1": {"off": 0.5, "on": 2.0}}, class_weights="auto")
    cfg["train"].update(output_dir=str(tmp_path / "run"), val_episode_indices=[1],
                        contact_sampling={"enabled": True, "phase_weights": [1., 5., 5.],
                                          "future_phase_reduction": "max", "replacement": True},
                        scheduler={"name": "cosine", "T_max": 4}, defer_metric_sync=True)
    return cfg


@pytest.mark.parametrize("downsample", [False, True])
@pytest.mark.parametrize("ema", [False, True])
def test_real_dataset_trainer_step_validation_checkpoint_and_resume(lerobot_fixture, monkeypatch, downsample, ema):
    cfg = lerobot_fixture
    cfg["train"]["downsample"] = downsample
    cfg["train"]["ema"]["enabled"] = ema
    # Keep this an integration test of the actual loop, excluding only plotting.
    monkeypatch.setattr(DeterministicWorldModelTrainer, "save_loss_plot", lambda self: None)
    trainer = DeterministicWorldModelTrainer(cfg)
    summary = trainer.train()
    assert summary["global_step"] == 1
    assert isinstance(trainer.dataset, dataset_module.ContactWorldModelDataset)
    train_indices = trainer.loader.dataset.indices
    val_indices = trainer.val_loader.dataset.indices
    assert set(train_indices).isdisjoint(val_indices)
    assert all(trainer.dataset.valid_indices[i] < 96 for i in train_indices)
    assert all(trainer.dataset.valid_indices[i] >= 96 for i in val_indices)
    covered = trainer.dataset.covered_raw_indices(train_indices)
    expected_mean = trainer.dataset.high_tensors["q"][covered].mean(0)
    torch.testing.assert_close(trainer.dataset.normalizer.stats["q"]["mean"], expected_mean)
    sampler = trainer.loader.sampler
    corrections = torch.tensor([trainer.dataset.importance_weight_by_sample_index[trainer.dataset.valid_indices[i]]
                                for i in train_indices], dtype=torch.double)
    torch.testing.assert_close((sampler.weights / sampler.weights.sum()) * corrections,
                               torch.full_like(corrections, 1 / len(corrections)))
    for index in val_indices:
        assert trainer.dataset[index]["importance_weight"] == 1
    metrics = trainer.last_val_epoch_metrics
    assert {"q_mse", "q_mae", "tau_mse", "tau_mae", "contact_accuracy", "contact_macro_f1", "contact_ce", "total_loss"} <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert not any("flow" in key or "energy" in key for key in metrics)
    checkpoint_path = Path(cfg["train"]["output_dir"]) / "checkpoints/latest.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_version"] == "deterministic_wm_v1"
    assert checkpoint["deterministic_wm_contract"] == checkpoint["carswm_contract"]
    assert checkpoint["ema"]["enabled"] is ema
    if downsample:
        # Already-trained v1 checkpoints did not contain the shared WM key.
        del checkpoint["carswm_contract"]
        torch.save(checkpoint, checkpoint_path)
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg["train"].update(resume_from=str(checkpoint_path), max_optimizer_steps=2)
    resumed = DeterministicWorldModelTrainer(resumed_cfg)
    resumed.setup()
    assert resumed.global_step == 1
    assert resumed.optimizer.state
    expected = checkpoint["model_raw"] if ema else checkpoint["model"]
    for key, tensor in resumed.model.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
    if ema:
        for key, tensor in resumed.ema.model.state_dict().items():
            torch.testing.assert_close(tensor, checkpoint["model"][key], rtol=0, atol=0)
    resumed.train_one_epoch(resumed.resume_epoch)
    assert resumed.global_step == 2


def test_confusion_matrix_macro_f1_includes_all_classes():
    confusion = torch.tensor([[8, 2, 0], [1, 1, 0], [0, 0, 0]])
    metrics = DeterministicWorldModelTrainer._contact_metrics(confusion)
    assert metrics["contact_accuracy"] == pytest.approx(9 / 12)
    assert metrics["contact_macro_f1"] == pytest.approx((16 / 19 + 2 / 5 + 0) / 3)


def test_checkpoint_contract_rejects_changed_stream_order_and_flow_version():
    cfg = config()
    model = DeterministicRobotStateWorldModel(cfg)
    checkpoint = {"model_version": model.MODEL_VERSION, "deterministic_wm_contract": model.baseline_contract()}
    model.validate_checkpoint(checkpoint)
    cfg["model"]["outputs"] = ["tau", "q"]
    with pytest.raises(ValueError, match="contract"):
        DeterministicRobotStateWorldModel(cfg).validate_checkpoint(checkpoint)
    checkpoint["model_version"] = "carswm_v9"
    with pytest.raises(ValueError, match="model_version"):
        model.validate_checkpoint(checkpoint)


def test_baseline_config_preserves_wm_experiment_settings():
    root = Path(__file__).resolve().parents[1] / "config/train_cfg"
    wm = yaml.safe_load((root / "cwm_all_50hz.yaml").read_text())
    baseline = yaml.safe_load((root / "deterministic_wm_baseline.yaml").read_text())
    for key in ("train_data", "dataloader", "action_contract", "contact_gate"):
        assert baseline[key] == wm[key]
    excluded = {"output_dir", "wandb", "validation_flow_time", "rollout_validation",
                "probabilistic_validation", "checkpoint_visualization"}
    assert {k: v for k, v in baseline["train"].items() if k not in excluded} == {
        k: v for k, v in wm["train"].items() if k not in excluded}
    assert not any(key.startswith("flow_") for key in baseline["model"])
    assert baseline["model"]["dropout"] == wm["model"]["dropout"]
    assert baseline["train"]["output_dir"] != wm["train"]["output_dir"]


def test_cpu_smoke_saves_ema_checkpoint(tmp_path):
    cfg = config()
    cfg["train"]["output_dir"] = str(tmp_path)
    report = run_smoke_test(cfg)
    assert report["optimizer_steps"] == 1
    assert report["external_shapes"]["contact_logits"] == [2, 6, 3]
    checkpoint = torch.load(report["checkpoint"], weights_only=False)
    assert checkpoint["model_raw"] and checkpoint["model"]
    assert checkpoint["ema"]["enabled"]


@pytest.mark.parametrize('downsample', [False, True, 3])
def test_wm_sampling_interface_decodes_once_and_ignores_flow_noise(downsample, monkeypatch):
    cfg = config()
    cfg['train']['downsample'] = downsample
    model = DeterministicRobotStateWorldModel(cfg).eval()
    values = batch(cfg)
    expected = model.predict(values)
    calls = []
    decode = model.predict_from_conditions

    def counted(encoded):
        calls.append(True)
        return decode(encoded)

    monkeypatch.setattr(model, 'predict_from_conditions', counted)
    sampled = model.sample(values, num_samples=3, steps=8, solver='heun',
                           source_noise=torch.randn(2, 3, 6, 4))
    assert len(calls) == 1
    for key, value in sampled.items():
        assert value.shape == (2, 3, 6, expected[key].shape[-1])
        for index in range(3):
            torch.testing.assert_close(value[:, index], expected[key], rtol=0, atol=0)
    assert not sampled['q_pred'].requires_grad
    with pytest.raises(ValueError, match='num_samples'):
        model.sample(values, num_samples=0)


def test_shared_and_legacy_checkpoint_contract_validation():
    model = DeterministicRobotStateWorldModel(config())
    payload = {'model_version': model.MODEL_VERSION,
               'carswm_contract': model.checkpoint_contract()}
    model.validate_checkpoint(payload)
    payload['deterministic_wm_contract'] = model.baseline_contract()
    model.validate_checkpoint(payload)
    payload['deterministic_wm_contract']['inputs'] = ('tau', 'q')
    with pytest.raises(ValueError, match='contract'):
        model.validate_checkpoint(payload)
    with pytest.raises(ValueError, match='contract'):
        model.validate_checkpoint({'model_version': model.MODEL_VERSION})
