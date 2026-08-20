from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_process.tool.torque_world_model_rollout_visualizer import (
    RunningSquaredError,
    plot_signal_comparison,
    resolve_checkpoint_path,
    select_rollout_indices,
)


def test_checkpoint_directory_selects_lowest_validation_loss(tmp_path):
    worse = tmp_path / "epoch_010_val_loss_0.9.pt"
    best = tmp_path / "epoch_011_val_loss_0.12.pt"
    worse.touch()
    best.touch()

    assert resolve_checkpoint_path(tmp_path) == best.resolve()


def test_checkpoint_directory_selects_newest_optimizer_step(tmp_path):
    (tmp_path / "step_00045000.pt").touch()
    (tmp_path / "step_00050000.pt").touch()
    (tmp_path / "latest.pt").touch()

    assert resolve_checkpoint_path(tmp_path).name == "step_00050000.pt"


def test_rollout_selection_is_episode_local_and_strided():
    episodes = [
        {"episode_index": 3, "dataset_from_index": 0, "dataset_to_index": 10},
        {"episode_index": 7, "dataset_from_index": 10, "dataset_to_index": 20},
    ]
    dataset = SimpleNamespace(
        future_horizon=4,
        valid_indices=[1, 2, 3, 4, 5, 11, 12, 13, 14, 15],
        episodes=episodes,
    )

    selected = select_rollout_indices(
        dataset,
        {
            "episode_indices": [7],
            "sample_stride": 2,
            "start_offset": 1,
            "max_rollouts_per_episode": 2,
        },
    )

    assert selected == [6, 8]


def test_rollout_selection_rejects_unknown_episode():
    dataset = SimpleNamespace(
        future_horizon=4,
        valid_indices=[1],
        episodes=[
            {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 3}
        ],
    )
    with pytest.raises(ValueError, match="do not exist"):
        select_rollout_indices(dataset, {"episode_indices": [2]})


def test_running_squared_error_reports_per_dimension_rmse():
    metric = RunningSquaredError()
    data = torch.zeros(1, 2, 2)
    prediction = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    metric.update("q", data, prediction)

    result = metric.result()["q"]
    assert result["rmse_per_dimension"] == pytest.approx([1.0, 2.0])
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))


def test_plot_contains_blue_data_and_yellow_prediction(tmp_path, monkeypatch):
    captured = {}

    def capture_savefig(self, path, *args, **kwargs):
        lines = [line for axis in self.axes for line in axis.lines]
        captured["colors"] = [line.get_color() for line in lines]

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_savefig)
    plot_signal_comparison(
        np.array([0.01, 0.02]),
        np.zeros((2, 2)),
        np.ones((2, 2)),
        signal="q",
        title="test",
        output_path=tmp_path / "q.png",
    )

    assert captured["colors"] == ["tab:blue", "#E6A700"] * 2
