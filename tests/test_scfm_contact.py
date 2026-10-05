"""Legacy GRU SCFM: frozen encoders, native export and exact resumed training."""
import sys

import pytest
import torch
import yaml

from model.pinn_model.contact_scfm import SCFMContactWorldModel, load_contact_checkpoint, make_contact_student
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.scfm_distillation import (
    SCFMSettings, frozen_snapshot, sample_schedule, scfm_loss, update_flow_ema,
)
from scripts import posttrain_scfm_latent as script
from test_contact_world_model import batch, config


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_legacy_scfm_freezes_condition_contact_and_shared_positions(tmp_path):
    cfg = config()
    cfg["model"].update(flow_inference_steps=32, dropout=.5, use_action_padding_mask=False)
    base = ContactWorldModel(cfg).eval()
    checkpoint = tmp_path/"teacher.pt"
    torch.save({"model_version": base.MODEL_VERSION, "config": cfg,
                "carswm_contract": base.checkpoint_contract(), "model": base.state_dict()}, checkpoint)
    teacher, payload = load_contact_checkpoint(checkpoint)
    student, _ = make_contact_student(teacher, payload, 4)
    fast, slow = frozen_snapshot(student), frozen_snapshot(student)
    before = {name: p.detach().clone() for name, p in student.named_parameters()}
    values = batch(cfg)
    with torch.no_grad():
        encoded = teacher.encode_conditions(values)
        target = teacher._target_flow_state(encoded["_prepared_batch"], values["q"])
    assert encoded["action_padding_mask"] is None
    schedule = sample_schedule(2, SCFMSettings(), torch.Generator().manual_seed(3))
    loss, _ = scfm_loss(student, teacher, fast, slow, target, torch.randn_like(target), encoded, schedule)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=.01)
    loss.backward()
    optimizer.step()
    update_flow_ema(fast, student, 0.)
    prefixes = tuple(name+"." for name in student.FLOW_MODULES)
    changed = []
    for name, parameter in student.named_parameters():
        if name.startswith(prefixes):
            changed.append(not torch.equal(parameter, before[name]))
        else:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            assert parameter.grad is None
    assert any(changed)
    assert all(p.grad is None for model in (teacher, fast, slow) for p in model.parameters())
    native = ContactWorldModel(student._config).eval()
    native.load_state_dict(student.state_dict(), strict=True)
    native.validate_checkpoint({"model_version": student.MODEL_VERSION,
                                "carswm_contract": student.checkpoint_contract()})
    noise = torch.randn(2, student.future_horizon, student.flow_dim)
    torch.testing.assert_close(native.predict(values, source_noise=noise)["flow_state_pred"],
                               student.predict(values, source_noise=noise)["flow_state_pred"])


def test_contact_cli_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    cfg = config()
    cfg["train"] = {"seed": 42}
    teacher = ContactWorldModel(cfg).eval()
    original = tmp_path/"teacher.pt"
    norm = {"stats": {k: {"mean": torch.zeros(2), "std": torch.ones(2)} for k in ("q", "tau")}}
    torch.save({"config": cfg, "model": teacher.state_dict(), "normalizer": norm,
                "carswm_contract": teacher.checkpoint_contract(), "model_version": teacher.MODEL_VERSION}, original)
    records = batch(cfg)

    class Records(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 4
        def __getitem__(self, index): return {k: v[index % 2] for k, v in records.items()}
        def set_normalizer(self, normalizer): pass
        batch_collate = staticmethod(torch.utils.data.default_collate)

    monkeypatch.setattr(script, "ContactWorldModelDataset", Records)
    monkeypatch.setattr(script, "episode_train_indices", lambda d, c: ([0, 1], script.index_hash([0, 1])))
    settings = SCFMSettings(updates=2, batch_size=2, teacher_min_steps=4, teacher_max_steps=4,
                            reference_steps=2, validation_samples=2, validation_batches=1,
                            save_every=1, validate_every=1)
    configuration = tmp_path/"settings.yaml"
    configuration.write_text(yaml.safe_dump({"scfm": settings.as_dict()}))
    common = ["scfm", "--base-checkpoint", str(original), "--config", str(configuration), "--device", "cpu"]
    full, resumed = tmp_path/"full.pt", tmp_path/"resumed.pt"
    for extra in (["--output", str(full)], ["--output", str(resumed), "--updates", "1"],
                  ["--output", str(resumed), "--resume", str(resumed)]):
        monkeypatch.setattr(sys, "argv", [*common, *extra])
        script.main(architecture="contact")
    a, payload = load_contact_checkpoint(full)
    b, _ = load_contact_checkpoint(resumed)
    for name, p in a.named_parameters():
        torch.testing.assert_close(p, dict(b.named_parameters())[name], rtol=0, atol=0)
    assert payload["scfm_posttrain"]["architecture"] == "contact"
    assert not payload["scfm_posttrain"]["original_split_hash_verified"]
    assert a.flow_inference_steps == 4 and a.flow_solver == "euler"
    assert payload["scfm_posttrain"]["last_validation"]["variants"]["scfm_coarse"]["nfe"] == 4


def test_relocation_updates_source_without_changing_original(tmp_path):
    cfg = config()
    cfg["train_data"] = {"format": "lerobot_v3", "sources": [{"root": "old", "repo_id": "old"}]}
    relocated = script.relocate_data(cfg, tmp_path, "new")
    assert relocated["train_data"]["sources"][0] == {"root": str(tmp_path), "repo_id": "new"}
    assert cfg["train_data"]["sources"][0]["root"] == "old"


def test_schema11_preserves_new_contact_semantics_and_rejects_mismatch(tmp_path):
    cfg = config()
    cfg["contact_gate"] = {"enabled": True, "label_mode": "three_phase", "metric": "tau_ext_l1",
                           "contact_threshold": 11.7, "precontact_duration_s": 1.0}
    model = SCFMContactWorldModel(cfg)
    contract = model.checkpoint_contract()
    assert contract["schema_version"] == 11
    assert contract["contact"]["classes"] == ["free", "alignment", "contact"]
    assert contract["contact"]["comparison"] == ">"
    payload = {"model_version": model.MODEL_VERSION, "config": cfg,
               "carswm_contract": contract, "model": model.state_dict()}
    path = tmp_path/"schema11.pt";torch.save(payload, path)
    restored, cp = load_contact_checkpoint(path)
    student, _ = make_contact_student(restored, cp, 4)
    assert student.checkpoint_contract()["contact"] == contract["contact"]
    assert student.checkpoint_contract()["schema_version"] == 11
    payload["carswm_contract"]["contact"]["contact_threshold"] = 10.
    torch.save(payload, path)
    with pytest.raises(ValueError, match="contract mismatch"):
        load_contact_checkpoint(path)
