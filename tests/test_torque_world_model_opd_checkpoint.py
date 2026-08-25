from pathlib import Path

import pytest

from train.trainer.torque_world_model_opd_train import TorqueWorldModelOPDTrainer


def test_resolve_teacher_checkpoint_selects_latest_step(tmp_path: Path):
    checkpoint_dir = tmp_path / "teacher"
    checkpoints = checkpoint_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_00000008.pt").touch()
    (checkpoints / "step_00000010.pt").touch()
    (checkpoints / "latest.pt").touch()

    resolved = TorqueWorldModelOPDTrainer._resolve_teacher_checkpoint(checkpoint_dir)

    assert resolved.name == "step_00000010.pt"


def test_resolve_teacher_checkpoint_selects_latest_scheduled_epoch(tmp_path: Path):
    checkpoint_dir = tmp_path / "teacher"
    checkpoints = checkpoint_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "epoch_0000500.pt").touch()
    (checkpoints / "epoch_0001500.pt").touch()
    (checkpoints / "epoch_0001000.pt").touch()

    resolved = TorqueWorldModelOPDTrainer._resolve_teacher_checkpoint(checkpoint_dir)

    assert resolved.name == "epoch_0001500.pt"


def test_resolve_teacher_checkpoint_prefers_epoch_over_legacy_step(tmp_path: Path):
    checkpoint_dir = tmp_path / "teacher"
    checkpoints = checkpoint_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_00099999.pt").touch()
    (checkpoints / "epoch_0000100.pt").touch()

    resolved = TorqueWorldModelOPDTrainer._resolve_teacher_checkpoint(checkpoint_dir)

    assert resolved.name == "epoch_0000100.pt"


def test_resolve_teacher_checkpoint_reports_missing_teacher(tmp_path: Path):
    missing = tmp_path / "teacher" / "checkpoints"

    with pytest.raises(FileNotFoundError, match="Train the Teacher first"):
        TorqueWorldModelOPDTrainer._resolve_teacher_checkpoint(missing)


def test_resolve_teacher_checkpoint_reports_empty_directory(tmp_path: Path):
    checkpoint_dir = tmp_path / "teacher"
    (checkpoint_dir / "checkpoints").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"contains no \.pt files"):
        TorqueWorldModelOPDTrainer._resolve_teacher_checkpoint(checkpoint_dir)
