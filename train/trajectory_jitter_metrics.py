"""Offline diagnostics for physical joint trajectories and replan revisions.

No model loading, sample selection, smoothing, or robot commands happen here.
All timestamps are unique increasing int64 nanoseconds within each trajectory.
"""
from __future__ import annotations

import torch

SCHEMA = "trajectory_jitter_v1"


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _times(value, length, name):
    if not torch.is_tensor(value) or value.dtype != torch.int64 or value.shape != (length,):
        raise ValueError(f"{name} must be int64 [{length}] nanoseconds")
    value = value.detach().cpu()
    if length == 0 or (torch.diff(value) <= 0).any():
        raise ValueError(f"{name} must be nonempty, unique and strictly increasing")
    return value


def _q(value, ndim, name):
    if not torch.is_tensor(value) or value.ndim != ndim or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor of rank {ndim} in radians")
    if any(size == 0 for size in value.shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value.detach().cpu().double()


def magnitude_stats(values):
    """Signed values summarized by magnitude; unavailable statistics are null."""
    values = values.reshape(-1).double()
    if not values.numel():
        return {"scalar_count": 0, "rmse": None, "mean_abs": None,
                "p95_abs": None, "p99_abs": None, "max_abs": None}
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite metric values")
    absolute = values.abs()
    return {"scalar_count": values.numel(), "rmse": float(values.square().mean().sqrt()),
            "mean_abs": float(absolute.mean()), "p95_abs": float(torch.quantile(absolute, .95)),
            "p99_abs": float(torch.quantile(absolute, .99)), "max_abs": float(absolute.max())}


def physical_difference(q, time_ns, order):
    """k! divided differences, in rad/s**k, with actual nonuniform time.

    Uniform sampling gives ordinary successive differences divided by dt**k.
    On irregular grids this reproduces derivatives of polynomials of degree k.
    It is a discrete diagnostic, not a noise-free measured physical derivative.
    """
    q = _q(q, 2, "q")
    time_ns = _times(time_ns, len(q), "time_ns")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3")
    if len(q) <= order:
        return q[:0]
    # Subtract integer epochs before conversion; large epoch ns lose precision
    # when converted to floating point first.
    seconds = (time_ns - time_ns[0]).double() * 1e-9
    result = q
    for level in range(1, order + 1):
        result = level * torch.diff(result, dim=0) / (seconds[level:] - seconds[:-level])[:, None]
    return result


def _trajectory_metrics(q, time_ns):
    names = ("velocity_rad_per_s", "acceleration_rad_per_s2", "jerk_rad_per_s3")
    return {name: magnitude_stats(physical_difference(q, time_ns, order))
            for order, name in enumerate(names, 1)}


def aligned_revision(old_q, old_time_ns, new_q, new_time_ns):
    """Match exactly equal physical timestamps, never equal array indices."""
    old_q, new_q = _q(old_q, 2, "old_q"), _q(new_q, 2, "new_q")
    old_time_ns = _times(old_time_ns, len(old_q), "old_time_ns")
    new_time_ns = _times(new_time_ns, len(new_q), "new_time_ns")
    if old_q.shape[1] != new_q.shape[1]:
        raise ValueError("old/new joint dimensions differ")
    indices = torch.searchsorted(old_time_ns, new_time_ns)
    valid = indices < len(old_time_ns)
    valid &= old_time_ns[indices.clamp_max(len(old_time_ns) - 1)] == new_time_ns
    return new_q[valid] - old_q[indices[valid]], new_time_ns[valid]


def _episode(value):
    if not isinstance(value, str) or not value:
        raise ValueError("episode_id must be a nonempty string")
    return value


def evaluate_jitter_log(payload):
    """Evaluate the documented trajectory_jitter_v1 log; return JSON-safe data."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"log schema must be {SCHEMA}")
    if payload.get("q_unit") != "rad" or payload.get("time_unit") != "ns":
        raise ValueError("explicit physical q_unit=rad and time_unit=ns are required")
    forecasts, execution = payload.get("forecasts", []), payload.get("execution", [])
    if not isinstance(forecasts, list) or not isinstance(execution, list) or not (forecasts or execution):
        raise ValueError("forecasts/execution must be lists with at least one record")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    result = {"schema": SCHEMA, "q_unit": "rad", "time_unit": "ns", "metadata": metadata,
              "note": "diagnostics only; prediction revisions include valid feedback corrections; success is evaluated separately",
              "forecasts": [], "adjacent_replans": [], "execution": []}
    previous, seen_requests, overlap_values, takeover_values = {}, set(), [], []
    for record in forecasts:
        if not isinstance(record, dict):
            raise ValueError("forecast records must be mappings")
        episode = _episode(record.get("episode_id"))
        request = _integer(record.get("request_id"), "request_id")
        identity = (episode, request)
        if identity in seen_requests:
            raise ValueError("duplicate request_id within episode")
        seen_requests.add(identity)
        samples = _q(record.get("q"), 3, "forecast q [K,H,D]")
        selected = _integer(record.get("selected_sample"), "selected_sample")
        if not 0 <= selected < len(samples):
            raise ValueError("selected_sample is outside supplied forecast samples")
        q = samples[selected]
        times = _times(record.get("future_time_ns"), len(q), "future_time_ns")
        anchor = _integer(record.get("request_anchor_ns"), "request_anchor_ns")
        if times[0] <= anchor:
            raise ValueError("forecast timestamps must follow request_anchor_ns")
        takeover = record.get("takeover_forecast_time_ns")
        if takeover is not None:
            _integer(takeover, "takeover_forecast_time_ns")
            if not (times == takeover).any():
                raise ValueError("takeover_forecast_time_ns must identify a supplied forecast frame")
        result["forecasts"].append({"episode_id": episode, "request_id": request,
            "selected_sample": selected, "frames": len(q), "chunk_internal": _trajectory_metrics(q, times)})
        if episode in previous:
            old = previous[episode]
            if anchor <= old["anchor"]:
                raise ValueError("request_anchor_ns must strictly increase within each episode")
            delta, matched_times = aligned_revision(old["q"], old["times"], q, times)
            overlap_values.append(delta)
            takeover_delta = delta[:0] if takeover is None else delta[matched_times == takeover]
            takeover_values.append(takeover_delta)
            reason = ("not_logged" if takeover is None else
                      "outside_old_forecast_overlap" if not len(takeover_delta) else None)
            result["adjacent_replans"].append({"episode_id": episode,
                "old_request_id": old["request"], "new_request_id": request,
                "old_selected_sample": old["selected"], "new_selected_sample": selected,
                "overlap_frames": len(delta), "overlap_q_revision_rad": magnitude_stats(delta),
                "takeover_forecast_time_ns": takeover,
                "takeover_unavailable_reason": reason,
                "takeover_q_revision_rad": magnitude_stats(takeover_delta)})
        previous[episode] = {"q": q, "times": times, "anchor": anchor, "request": request, "selected": selected}

    seen_execution = set()
    for record in execution:
        if not isinstance(record, dict):
            raise ValueError("execution records must be mappings")
        episode = _episode(record.get("episode_id"))
        if episode in seen_execution:
            raise ValueError("provide one complete execution trace per episode")
        seen_execution.add(episode)
        q = _q(record.get("q_command"), 2, "q_command [T,D]")
        times = _times(record.get("command_time_ns"), len(q), "command_time_ns")
        row = {"episode_id": episode, "frames": len(q), "command": _trajectory_metrics(q, times),
               "boundary_unavailable_reason": None, "takeover_boundaries": None, "measured": None}
        request_ids = record.get("request_id")
        if request_ids is None:
            row["boundary_unavailable_reason"] = "execution request_id trace not logged"
        else:
            if not torch.is_tensor(request_ids) or request_ids.dtype != torch.int64 or request_ids.shape != (len(q),):
                raise ValueError("execution request_id must be int64 [T]; use -1 before any forecast")
            request_ids = request_ids.detach().cpu()
            # A change to -1 means hold/recovery, not a model takeover.
            boundaries = torch.zeros(len(q), dtype=torch.bool)
            boundaries[1:] = (request_ids[1:] != request_ids[:-1]) & (request_ids[1:] >= 0)
            cumulative = boundaries.long().cumsum(0)
            metrics = {"count": int(boundaries.sum()),
                       "command_increment_rad": magnitude_stats(torch.diff(q, dim=0)[boundaries[1:]])}
            for order, name in enumerate(("velocity_rad_per_s", "acceleration_rad_per_s2", "jerk_rad_per_s3"), 1):
                derivative = physical_difference(q, times, order)
                crosses = cumulative[order:] - cumulative[:-order] > 0 if len(q) > order else boundaries[:0]
                metrics[name] = magnitude_stats(derivative[crosses])
            row["takeover_boundaries"] = metrics
        if "q_measured" in record or "measured_time_ns" in record:
            measured = _q(record.get("q_measured"), 2, "q_measured [U,D]")
            measured_times = _times(record.get("measured_time_ns"), len(measured), "measured_time_ns")
            if measured.shape[1] != q.shape[1]:
                raise ValueError("command/measured joint dimensions differ")
            row["measured"] = _trajectory_metrics(measured, measured_times)
        result["execution"].append(row)

    def pooled(values):
        # Pool scalar differences, allowing differing horizons and joint counts.
        return magnitude_stats(torch.cat([value.reshape(-1) for value in values]) if values else torch.empty(0))

    result["summary"] = {"forecast_count": len(forecasts), "execution_episode_count": len(execution),
        "adjacent_pair_count": len(result["adjacent_replans"]),
        "pairs_without_overlap": sum(row["overlap_frames"] == 0 for row in result["adjacent_replans"]),
        "pairs_without_takeover_revision": sum(row["takeover_unavailable_reason"] is not None for row in result["adjacent_replans"]),
        "overlap_q_revision_rad": pooled(overlap_values), "takeover_q_revision_rad": pooled(takeover_values)}
    return result
