import copy

import pytest
import torch

from train.trajectory_jitter_metrics import (SCHEMA, aligned_revision, evaluate_jitter_log,
                                             physical_difference)


def forecast(episode, request, anchor, q, *, sample=0, takeover=None):
    return {"episode_id": episode, "request_id": request, "request_anchor_ns": anchor,
            "q": q, "selected_sample": sample,
            "future_time_ns": torch.arange(1, q.shape[1] + 1, dtype=torch.int64) * 10_000_000 + anchor,
            "takeover_forecast_time_ns": takeover}


def log(forecasts=None, execution=None):
    return {"schema": SCHEMA, "q_unit": "rad", "time_unit": "ns",
            "metadata": {"noise_policy": "absolute_time_common_noise", "execute_steps": 20, "control_hz": 100},
            "forecasts": forecasts or [], "execution": execution or []}


def test_match_physical_time_not_array_index_and_preserve_logged_sample():
    # Two equally smooth predictions have different first array values because
    # their request anchors differ. Their overlap at equal time is identical.
    old = torch.arange(1., 11.).reshape(1, 10, 1)
    new = torch.arange(4., 14.).reshape(1, 10, 1)
    # Sample 0 is deliberately wrong; tool must use logged sample 1.
    payload = log([forecast("a", 0, 0, old),
                   forecast("a", 1, 30_000_000, torch.cat((new + 999, new)), sample=1,
                            takeover=60_000_000)])
    result = evaluate_jitter_log(payload)
    pair = result["adjacent_replans"][0]
    assert pair["overlap_frames"] == 7
    assert pair["overlap_q_revision_rad"]["max_abs"] == 0
    assert pair["takeover_q_revision_rad"]["max_abs"] == 0
    assert pair["new_selected_sample"] == 1


def test_smooth_chunks_can_have_large_replan_jump():
    old = torch.zeros(1, 10, 1)
    new = torch.full((1, 10, 1), .25)
    result = evaluate_jitter_log(log([forecast("a", 0, 0, old),
        forecast("a", 1, 30_000_000, new, takeover=60_000_000)]))
    assert result["forecasts"][1]["chunk_internal"]["jerk_rad_per_s3"]["max_abs"] == 0
    assert result["summary"]["overlap_q_revision_rad"]["max_abs"] == .25
    assert result["summary"]["takeover_q_revision_rad"]["max_abs"] == .25


def test_missing_overlap_or_takeover_is_null_not_zero_and_episode_isolation():
    q = torch.zeros(1, 4, 1)
    result = evaluate_jitter_log(log([forecast("a", 0, 0, q), forecast("b", 0, 0, q + 100),
                                    forecast("a", 1, 100_000_000, q)]))
    assert result["summary"]["adjacent_pair_count"] == 1
    assert result["summary"]["pairs_without_overlap"] == 1
    assert result["summary"]["overlap_q_revision_rad"]["rmse"] is None
    assert result["summary"]["takeover_q_revision_rad"]["rmse"] is None
    assert result["adjacent_replans"][0]["takeover_unavailable_reason"] == "not_logged"
    assert evaluate_jitter_log(log([forecast("a", 0, 0, q)]))["summary"]["adjacent_pair_count"] == 0


def test_takeover_beyond_old_horizon_is_explicitly_unavailable():
    q = torch.zeros(1, 4, 1)
    result = evaluate_jitter_log(log([forecast("a", 0, 0, q),
        forecast("a", 1, 20_000_000, q, takeover=60_000_000)]))
    pair = result["adjacent_replans"][0]
    assert pair["overlap_frames"] == 2
    assert pair["takeover_unavailable_reason"] == "outside_old_forecast_overlap"
    assert pair["takeover_q_revision_rad"]["max_abs"] is None


def test_actual_nonuniform_dt_and_large_integer_epoch_in_rad_units():
    epoch = 1_790_000_000_000_000_000
    offsets = torch.tensor([0, 10_000_000, 25_000_000, 50_000_000, 90_000_000], dtype=torch.int64)
    t = offsets.double() * 1e-9
    q = t.pow(3)[:, None]
    torch.testing.assert_close(physical_difference(q, offsets + epoch, 3), torch.full((2, 1), 6., dtype=torch.float64))
    torch.testing.assert_close(physical_difference(t.square()[:, None], offsets + epoch, 2),
                               torch.full((3, 1), 2., dtype=torch.float64))


def test_actual_issued_commands_boundary_increment_and_measured_are_separate():
    q = torch.tensor([[0.], [.01], [.02], [.07], [.08], [.09]], dtype=torch.float64)
    execution = {"episode_id": "a", "q_command": q,
        "command_time_ns": torch.arange(6, dtype=torch.int64) * 10_000_000,
        "request_id": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
        "q_measured": torch.zeros(4, 1), "measured_time_ns": torch.arange(4, dtype=torch.int64) * 12_000_000}
    result = evaluate_jitter_log(log(execution=[execution]))["execution"][0]
    assert result["takeover_boundaries"]["count"] == 1
    assert result["takeover_boundaries"]["command_increment_rad"]["max_abs"] == pytest.approx(.05)
    assert result["takeover_boundaries"]["velocity_rad_per_s"]["max_abs"] == pytest.approx(5.)
    assert result["takeover_boundaries"]["jerk_rad_per_s3"]["max_abs"] > 0
    assert result["measured"]["jerk_rad_per_s3"]["max_abs"] == 0


def test_missing_execution_boundaries_and_short_derivative_are_null():
    trace = {"episode_id": "a", "q_command": torch.zeros(2, 1),
             "command_time_ns": torch.tensor([0, 10_000_000], dtype=torch.int64)}
    row = evaluate_jitter_log(log(execution=[trace]))["execution"][0]
    assert row["takeover_boundaries"] is None
    assert row["boundary_unavailable_reason"]
    assert row["command"]["jerk_rad_per_s3"]["rmse"] is None
    trace["request_id"] = torch.tensor([0, 0], dtype=torch.int64)
    row = evaluate_jitter_log(log(execution=[trace]))["execution"][0]
    assert row["takeover_boundaries"]["count"] == 0
    assert row["takeover_boundaries"]["command_increment_rad"]["rmse"] is None


@pytest.mark.parametrize("bad_time", [torch.tensor([1, 1, 3], dtype=torch.int64),
                                     torch.tensor([3, 2, 1], dtype=torch.int64),
                                     torch.tensor([1., 2., 3.])])
def test_duplicate_decreasing_or_float_timestamps_rejected(bad_time):
    with pytest.raises(ValueError, match="time"):
        aligned_revision(torch.zeros(3, 1), bad_time, torch.zeros(3, 1), bad_time)


def test_contract_invalid_selection_units_anchor_and_duplicates_rejected():
    payload = log([forecast("a", 0, 0, torch.zeros(1, 4, 1))])
    variants = []
    item = copy.deepcopy(payload); item["q_unit"] = "normalized"; variants.append(item)
    item = copy.deepcopy(payload); item["forecasts"][0]["selected_sample"] = 1; variants.append(item)
    item = copy.deepcopy(payload); item["forecasts"] *= 2; variants.append(item)
    item = copy.deepcopy(payload); item["forecasts"].append(forecast("a", 1, 0, torch.zeros(1, 4, 1))); variants.append(item)
    item = copy.deepcopy(payload); item["forecasts"][0]["takeover_forecast_time_ns"] = 123; variants.append(item)
    for item in variants:
        with pytest.raises(ValueError):
            evaluate_jitter_log(item)
