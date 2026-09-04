import pytest
import torch

from train.trainer.contact_world_model_train import ContactWorldModelTrainer
from train.carswm_metrics import (
    contact_confusion_matrix,
    contact_macro_f1_from_confusion,
    contact_metrics,
)


def test_contact_nll_is_negative_log_probability():
    probabilities = torch.tensor(
        [[[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]]], dtype=torch.float32
    )
    target = torch.tensor([[[0.0], [1.0]]])

    metrics = contact_metrics(probabilities, target)

    expected = (-torch.log(torch.tensor([0.8, 0.7]))).mean()
    assert metrics["contact_nll"].item() == pytest.approx(expected.item())
    assert metrics["contact_nll"].item() >= 0.0


def test_contact_macro_f1_uses_one_confusion_matrix_for_all_frames():
    probabilities = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            [
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
        ]
    )
    target = torch.tensor(
        [
            [[0.0], [0.0]],
            [[1.0], [2.0]],
        ]
    )

    confusion = contact_confusion_matrix(probabilities, target)
    metrics = contact_metrics(probabilities, target)

    assert confusion.tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["contact_macro_f1"].item() == pytest.approx(1.0)
    assert contact_macro_f1_from_confusion(confusion) == pytest.approx(1.0)


def test_absent_class_is_not_penalized_when_global_confusion_has_no_error():
    probabilities = torch.tensor(
        [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]
    )
    target = torch.tensor([[[0.0], [0.0]]])

    confusion = contact_confusion_matrix(probabilities, target)
    assert contact_macro_f1_from_confusion(confusion).item() == pytest.approx(1.0 / 3.0)


def test_task_macro_metrics_weight_tasks_equally():
    metrics = {
        "probabilistic_task_small__energy_score": 1.0,
        "probabilistic_task_small__min_ade": 2.0,
        "probabilistic_task_large__energy_score": 3.0,
        "probabilistic_task_large__min_ade": 4.0,
    }

    ContactWorldModelTrainer._finalize_task_macro_metrics(
        metrics, prefix="probabilistic"
    )

    assert metrics["probabilistic_task_macro_energy_score"] == pytest.approx(2.0)
    assert metrics["probabilistic_task_macro_min_ade"] == pytest.approx(3.0)
