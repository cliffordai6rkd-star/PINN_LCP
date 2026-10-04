# Replan jump diagnostics in physical joint coordinates

The primary question is whether a new prediction replaces the previous prediction with a different joint target at the **same physical future time**. A smooth individual chunk can still jump at every replan. This tool measures that revision separately from chunk derivatives and the actual issued command. It loads logs only: no model loading, sample selection, filtering or hardware commands.

## Run

```bash
python scripts/evaluate_trajectory_jitter.py --log trajectory_log.pt --output jitter.json
```

The input is a tensor/primitives `.pt` mapping, loaded with `weights_only=True`. `q_unit=rad` is required. Denormalize using each checkpoint's own saved statistics before logging. Degrees and normalized values cannot be compared by changing a label. All timestamps must be int64 nanoseconds, unique and strictly increasing within their trajectory. Use one consistent physical clock; subtract integer epochs before converting to seconds. Missing statistics are reported as `null` with a count/reason, never as zero.

## Minimal log schema

```python
torch.save({
    "schema": "trajectory_jitter_v1",
    "q_unit": "rad",
    "time_unit": "ns",
    "metadata": {
        "variant": "original_4_steps",  # record checkpoint hash and solver/NFE too
        "control_hz": 100,
        "execute_steps": 20,
        "noise_policy": "independent_per_request",
        "schedule_policy": "fixed_matched_takeover",
    },
    "forecasts": [
        {
            "episode_id": "trial_000",
            "request_id": 0,
            "request_anchor_ns": 1_000_000_000,
            "q": physical_q,                 # floating [K,H,D], rad
            "selected_sample": 0,             # REQUIRED: actually chosen by deployment
            "future_time_ns": future_times,   # int64 [H], physical prediction times
            # Optional: physical time of the new forecast frame used at takeover.
            # This identifies a future_time_ns entry, NOT arrival/completion time.
            "takeover_forecast_time_ns": 1_050_000_000,
        },
        # Requests must have increasing anchors within each episode.
    ],
    "execution": [
        {
            "episode_id": "trial_000",
            "q_command": issued_q,            # floating [T,D], rad; after clipping
            "command_time_ns": actual_sends,  # int64 [T], actual successful sends
            "request_id": issued_requests,   # optional int64 [T], -1 before WM/hold
            # Optional: raw measured q at its OWN feedback timestamps.
            "q_measured": measured_q,         # floating [U,D], rad
            "measured_time_ns": feedback_ns,  # int64 [U]
        },
    ],
}, "trajectory_log.pt")
```

Either forecasts or execution can be absent, but the log must contain a record. A forecast contains every candidate supplied in the log; the evaluator uses only the explicit `selected_sample`. Equal sample indices across requests do **not** guarantee the same physical mode. It does not choose a smoother candidate or average samples. Adjacent requests are paired within each `episode_id`; episode resets are never compared. Supply one complete command trace per episode. Log repeated stale feedback only once at its source timestamp; repeated issued holds remain legitimate command samples at distinct send times.

## Priority metrics

1. `overlap_q_revision_rad`: new minus old selected prediction at exactly matching absolute future timestamps. It includes valid corrections caused by new feedback, so it is a consistency diagnostic rather than automatically an error. RMSE, p95/p99 absolute joint revision and max absolute revision are reported. No overlap produces null metrics.
2. `takeover_q_revision_rad`: same-time forecast revision at the logged `takeover_forecast_time_ns`. If this metadata is absent or the old forecast does not cover that frame, the reason is reported. Never compare the first index of both chunks unless they really represent the same time.
3. `takeover_boundaries.command_increment_rad`: actual issued `q[i]-q[i-1]` when execution changes to a new nonnegative request ID. Unlike the same-time forecast revision, this includes normal motion over one command interval. A transition from `-1` also counts as a takeover; inspect initial/recovery events separately if needed. Changes to `-1` are holds, not takeovers. Without request IDs boundary metrics are unavailable.
4. `command` and `measured`: physical finite-difference velocity, acceleration and jerk using the actual timestamps. Boundary derivatives use only stencils that cross a takeover. `chunk_internal` reports these inside each selected prediction. Insufficient frames produce null. For uneven intervals the implementation uses `k!` divided differences; on uniform grids it is ordinary successive difference divided by `dt**k`. Measured jerk is sensitive to sensor noise and repeated feedback; report preprocessing and compare the same measurement pipeline.

The tool does not estimate success rate or select a winner from these metrics. Lower jitter alone can mean reduced responsiveness. Report task success, contact timing and force/torque excursions alongside these results.

## Minimal independent experiment

Hold history preprocessing, physical action stream, checkpoint output units, control rate, selected-sample rule and **execute_steps** constant. Count both ODE steps and NFE: four Heun steps are eight velocity evaluations. A smaller solver step count must not silently cause more frequent replanning in the first comparison.

- Compare the existing checkpoint at 32 and 4 steps on the same recorded request sequence, same epsilon for each paired request, same solver, identical forecast physical grids and takeover schedule. This isolates integration from different observations/timing.
- Evaluate condition-Gaussian training and distillation as separate factors, as described in their training docs; retain the original weights for the solver-only row. Use each model's correct source transform on the shared base epsilon.
- Repeat adjacent replans with independent per-request noise and a documented coherent-noise policy. The present xArm `WMAdapter.infer()` does not supply `source_noise`, so each call samples fresh noise; `selected_sample=0` fixes an array index, not the sampled mode. Reusing a seed per chunk gives the same relative-index epsilon, not necessarily the same epsilon at the same physical timestamp. A stronger diagnostic stores base epsilon by absolute forecast time for overlapping frames. It is still an ablation, not guaranteed physical mode tracking.
- Finally measure natural latency with the same control rate and execution length. Record actual request anchors, completion times, takeover frames, request IDs and successful clipped sends. The xArm runtime calibrates prefetch from measured WM latency, consumes `q[selected_sample, step-request.anchor]`, and takes over after its configured execution length; a faster model can therefore change the consumed horizon/action phase even at unchanged 100 Hz control. Log target timestamps explicitly rather than inferring them from inference completion.

The existing `benchmark_latent_cwm.py` evaluates one fixed input/noise for latency and paired numerical differences. It does not produce replan or execution logs. This evaluator needs the logger/inference replay to supply the schema above; it does not add a latent loader to xArm or claim deployment compatibility.
