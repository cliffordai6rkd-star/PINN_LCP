from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_process.tool.sequence_torque_inference_visualizer import (
    CheckpointNormalizer,
    _plot_indices,
    _select_episodes,
    add_external_wrench_rollout,
    causal_trailing_moving_average,
    infer_sequence_columns,
    resolve_checkpoint_path,
    rollout_metrics,
    torque_error_metrics,
)


def test_episode_selection_defaults_to_entire_dataset():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            episodes=[
                {"episode_index": 2, "length": 10},
                {"episode_index": 0, "length": 20},
            ]
        )
    )

    selected = _select_episodes(dataset, requested=None, all_episodes=False)

    assert [index for index, _ in selected] == [0, 2]


def test_episode_selection_honors_explicit_indices():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            episodes=[{"episode_index": 0}, {"episode_index": 1}]
        )
    )

    selected = _select_episodes(dataset, requested=[1], all_episodes=False)

    assert [index for index, _ in selected] == [1]


def test_plot_indices_use_every_point_by_default():
    assert _plot_indices(10_000, None).tolist() == list(range(10_000))


def test_checkpoint_directory_selects_lowest_numeric_score(tmp_path):
    worse = tmp_path / "epoch_002_val_loss_0.20.pt"
    best = tmp_path / "epoch_001_train_eval_loss_0.08.pt"
    checkpoint = {
        "config": {
            "model": {
                "target_key": "tau_f",
                "architecture": "gru",
                "inputs": ["q"],
            },
            "dataloader": {"horizon": 3},
            "train": {"split_mode": "purged_temporal"},
        }
    }
    torch.save(checkpoint, worse)
    torch.save(checkpoint, best)

    assert resolve_checkpoint_path(tmp_path) == best.resolve()


def test_checkpoint_directory_rejects_mixed_split_contracts(tmp_path):
    for split_mode in ("sample", "purged_temporal"):
        torch.save(
            {
                "config": {
                    "model": {"target_key": "tau", "architecture": "lstm"},
                    "dataloader": {"horizon": 50},
                    "train": {"split_mode": split_mode},
                }
            },
            tmp_path / f"{split_mode}_val_loss_0.01.pt",
        )

    with pytest.raises(ValueError, match="mixes different model/split contracts"):
        resolve_checkpoint_path(tmp_path)


def test_quantile_normalizer_round_trip():
    checkpoint = {
        "normalizer": {
            "normalize_mode": "quantile",
            "normalize_lowdim_keys": ["q"],
            "stats": {
                "q": {
                    "q01": torch.tensor([-1.0]),
                    "q99": torch.tensor([3.0]),
                }
            },
        }
    }
    normalizer = CheckpointNormalizer(checkpoint, {"dataloader": {}})
    value = torch.tensor([[1.0], [2.0]])

    normalized = normalizer.normalize("q", value)
    restored = normalizer.denormalize("q", normalized)

    torch.testing.assert_close(normalized, torch.tensor([[0.0], [0.5]]))
    torch.testing.assert_close(restored, value)


def test_inference_uses_only_complete_history_and_returns_physical_arrays():
    class LastValueModel(torch.nn.Module):
        active_inputs = ("q",)

        def forward(self, batch):
            return {"tau_f_pred": batch["q"][:, -1]}

    columns = {
        "q": torch.arange(6, dtype=torch.float32).reshape(6, 1),
        "tau_f": torch.arange(6, dtype=torch.float32).reshape(6, 1),
        "timestamp": torch.arange(6, dtype=torch.float64) * 0.01,
    }
    normalizer = CheckpointNormalizer(
        {"normalizer": {"normalize_mode": None}},
        {"dataloader": {}},
    )

    result = infer_sequence_columns(
        LastValueModel(),
        normalizer,
        columns,
        target_key="tau_f",
        horizon=3,
        start_frame=0,
        end_frame=None,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert result["target_frame"].tolist() == [2, 3, 4, 5]
    assert result["timestamp_s"].tolist() == [0.02, 0.03, 0.04, 0.05]
    torch.testing.assert_close(
        torch.from_numpy(result["prediction_nm"]),
        torch.tensor([[2.0], [3.0], [4.0], [5.0]]),
    )
    torch.testing.assert_close(torch.from_numpy(result["error_nm"]), torch.zeros(4, 1))


def test_torque_error_metrics_reports_joint_physical_metrics():
    result = {
        "error_nm": torch.tensor([[1.0, -2.0], [3.0, 0.0]]).numpy(),
    }

    metrics = torque_error_metrics(result)

    assert metrics["sample_count"] == 2
    assert metrics["overall_mae_nm"] == 1.5
    assert metrics["joint_metrics"][0]["mae_nm"] == 2.0
    assert metrics["joint_metrics"][1]["bias_nm"] == -1.0


class FakeRolloutDynamics:
    def inverse_dynamics(self, q, dq, ddq, **kwargs):
        return torch.ones_like(q)

    def frame_jacobians(self, q, **kwargs):
        return torch.eye(6, 7, dtype=q.dtype).expand(len(q), 6, 7).clone()


def test_tau_free_rollout_maps_prediction_residual_directly_to_wrench():
    result = {
        "target_frame": torch.tensor([2, 3]).numpy(),
        "timestamp_s": torch.tensor([0.02, 0.03]).numpy(),
        "error_nm": torch.full((2, 7), 0.2).numpy(),
    }
    columns = {"q": torch.zeros(5, 7)}

    output = add_external_wrench_rollout(
        "tau_free",
        result,
        columns,
        {"physics": {"wrench_damping": 0.02}},
        dynamics=FakeRolloutDynamics(),
    )

    torch.testing.assert_close(
        torch.from_numpy(output["tau_ext_nm"]),
        torch.full((2, 7), -0.2, dtype=torch.float64),
    )
    expected = -0.2 / (1.0 + 0.02**2)
    torch.testing.assert_close(
        torch.from_numpy(output["wrench_ext"]),
        torch.full((2, 6), expected, dtype=torch.float64),
    )
    contribution = torch.from_numpy(output["wrench_joint_contribution"])
    assert contribution.shape == (2, 7, 6)
    torch.testing.assert_close(contribution.sum(dim=1), torch.from_numpy(output["wrench_ext"]))
    metrics = rollout_metrics(output)
    assert metrics["force_norm_mean_n"] > 0.0
    influence = metrics["joint_wrench_influence"]
    assert len(influence) == 7
    assert influence[0]["force_contribution_rms_n"] > 0.0
    assert influence[6]["force_contribution_rms_n"] == 0.0


def test_tau_ext_filter_removes_an_isolated_residual_before_wrench_mapping():
    error = torch.zeros(5, 7, dtype=torch.float64)
    error[2] = 1.0
    result = {
        "target_frame": torch.arange(5).numpy(),
        "timestamp_s": (torch.arange(5, dtype=torch.float64) * 0.01).numpy(),
        "error_nm": error.numpy(),
    }
    columns = {"q": torch.zeros(5, 7)}
    config = {
        "physics": {"wrench_damping": 0.02},
        "rollout": {
            "tau_ext_filter": {
                "enabled": True,
                "mode": "median",
                "window": 3,
                "cutoff_hz": 20.0,
            }
        },
    }

    output = add_external_wrench_rollout(
        "tau_free",
        result,
        columns,
        config,
        dynamics=FakeRolloutDynamics(),
    )

    assert np.max(np.abs(output["tau_ext_raw_nm"])) == 1.0
    assert np.max(np.abs(output["wrench_ext_raw"])) > 0.0
    torch.testing.assert_close(
        torch.from_numpy(output["tau_ext_nm"]),
        torch.zeros(5, 7, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.from_numpy(output["wrench_ext"]),
        torch.zeros(5, 6, dtype=torch.float64),
    )
    assert rollout_metrics(output)["unfiltered"]["force_norm_max_n"] > 0.0


def test_causal_moving_average_spreads_an_impulse_over_its_trailing_window():
    values = np.asarray([[0.0], [0.0], [3.0], [0.0], [0.0], [0.0]])

    filtered = causal_trailing_moving_average(values, window=3)

    np.testing.assert_allclose(
        filtered[:, 0],
        np.asarray([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
    )


def test_tau_f_rollout_replays_causal_rnea_filter_before_wrench_mapping():
    result = {
        "target_frame": torch.tensor([2, 3]).numpy(),
        "timestamp_s": torch.tensor([0.02, 0.03]).numpy(),
        "prediction_nm": torch.full((2, 7), 0.5).numpy(),
        "error_nm": torch.zeros(2, 7).numpy(),
    }
    columns = {
        "q": torch.zeros(5, 7),
        "dq": torch.zeros(5, 7),
        "tau": torch.full((5, 7), 2.0),
        "timestamp": torch.arange(5, dtype=torch.float64) * 0.01,
    }
    config = {
        "model": {"target_filter": {"cutoff_hz": 10.0, "median_window": 1}},
        "rollout": {"measured_tau_already_filtered": True},
        "physics": {"wrench_damping": 0.02},
    }

    output = add_external_wrench_rollout(
        "tau_f",
        result,
        columns,
        config,
        dynamics=FakeRolloutDynamics(),
    )

    torch.testing.assert_close(
        torch.from_numpy(output["tau_id_filtered_nm"]),
        torch.ones(2, 7, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.from_numpy(output["tau_ext_nm"]),
        torch.full((2, 7), 0.5, dtype=torch.float64),
    )
    expected_wrench = 0.5 / (1.0 + 0.02**2)
    torch.testing.assert_close(
        torch.from_numpy(output["wrench_ext"]),
        torch.full((2, 6), expected_wrench, dtype=torch.float64),
    )
