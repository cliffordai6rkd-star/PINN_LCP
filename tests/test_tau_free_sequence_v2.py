from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from train.trainer.tau_free_sequence_train_v2 import TauFreeSequenceTrainerV2
from train.trainer.tau_f_sequence_train import TauFTrainer


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "train_cfg"
    / "tau_free_sequence_v2.yaml"
)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_v2_config_matches_q_history_to_measured_torque_contract():
    config = load_config()

    assert config["dataloader"]["horizon"] == 50
    assert config["dataloader"]["pad_history"] is False
    assert config["dataloader"]["contact_free"] is True
    assert config["dataloader"]["lowdim_keys"]["q"] == "observation.joint"
    assert config["dataloader"]["lowdim_keys"]["dq"] == "observation.velocity"
    assert config["dataloader"]["lowdim_keys"]["delta_q"] == (
        "observation.delta_q"
    )
    assert config["dataloader"]["lowdim_keys"]["tau"] == "observation.torque"
    assert config["model"]["inputs"] == ["q", "dq", "delta_q"]
    assert config["model"]["target_key"] == "tau"
    assert config["model"]["architecture"] == "lstm"
    assert config["train"]["val_ratio"] == pytest.approx(0.1)
    assert config["train"]["split_mode"] == "sample"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "inputs"), ["q", "tau"], "inputs"),
        (("model", "target_key"), "tau_f", "target_key"),
        (("model", "architecture"), "gru", "architecture"),
        (("dataloader", "horizon"), 25, "horizon"),
        (("dataloader", "pad_history"), True, "pad_history"),
        (("dataloader", "contact_free"), False, "contact_free"),
    ],
)
def test_v2_rejects_training_inference_contract_drift(path, value, message):
    config = load_config()
    config["train"]["device"] = "cpu"
    config[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        TauFreeSequenceTrainerV2(config)


def test_v2_model_uses_configured_inputs_and_last_measured_torque():
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"].update(
        hidden_dim=8,
        num_layers=1,
        head_hidden_dim=8,
        dropout=0.0,
    )
    trainer = TauFreeSequenceTrainerV2(config)
    trainer.model = trainer.build_model()
    tau = torch.randn(3, 50, 7)
    batch = {
        "q": torch.randn(3, 50, 7),
        "dq": torch.randn(3, 50, 7),
        "delta_q": torch.randn(3, 50, 7),
        "tau": tau,
    }

    loss, out = trainer.compute_loss(batch)

    assert trainer.model.active_inputs == ["q", "dq", "delta_q"]
    assert trainer.model.recurrent.input_size == 21
    assert loss.ndim == 0
    torch.testing.assert_close(out["tau_f_target"], tau[:, -1])


@pytest.mark.parametrize(
    "inputs",
    [
        ["q"],
        ["q", "dq", "delta_q"],
    ],
)
def test_v2_allows_next_input_ablations_from_config(inputs):
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"]["inputs"] = inputs

    trainer = TauFreeSequenceTrainerV2(config)

    assert trainer.build_model().active_inputs == inputs


def test_v2_reports_missing_optional_dataset_column_before_cache(monkeypatch):
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"]["inputs"] = ["q", "dq", "delta_q"]
    dataset = SimpleNamespace(
        stats_dataset=SimpleNamespace(
            column_names=[
                "observation.joint",
                "observation.velocity",
                "observation.torque",
            ]
        )
    )
    monkeypatch.setattr(TauFTrainer, "build_dataset", lambda _self: dataset)
    trainer = TauFreeSequenceTrainerV2(config)

    with pytest.raises(ValueError, match="meaningful q_cmd"):
        trainer.build_dataset()


def test_v2_windows_do_not_carry_hidden_state_between_samples():
    config = load_config()
    config["train"]["device"] = "cpu"
    config["model"].update(
        hidden_dim=8,
        num_layers=1,
        head_hidden_dim=8,
        dropout=0.0,
    )
    model = TauFreeSequenceTrainerV2(config).build_model().eval()
    first = {
        "q": torch.randn(2, 50, 7),
        "dq": torch.randn(2, 50, 7),
        "delta_q": torch.randn(2, 50, 7),
    }
    second = {
        "q": torch.randn(2, 50, 7),
        "dq": torch.randn(2, 50, 7),
        "delta_q": torch.randn(2, 50, 7),
    }

    with torch.no_grad():
        model(first)
        after_first = model(second)["tau_f_pred"]
        repeated = model(second)["tau_f_pred"]

    torch.testing.assert_close(after_first, repeated)
