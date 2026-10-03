"""Measure latent CARS-WM inference latency and paired numerical differences.

Example (run from repository root):
  PYTHONPATH=. python scripts/benchmark_latent_cwm.py \
      --checkpoint outputs/model.pt --device cuda:0 --steps 16,4,2 \
      --condition normalized_condition.pt --output benchmark.json

A configuration instead of a checkpoint creates random weights, a synthetic
codec, and identity latent statistics. It can measure cost, never task quality.
Without --condition, the input window is synthetic even for a real checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Allow both `python -m scripts...` and a direct script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from train.nomalizer import Normalizer


def step_list(value):
    try:
        values = list(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated positive integers") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("steps must be comma-separated positive integers")
    return values


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def measure(function, *, device, warmup, repeats):
    sync(device)
    warmup_start = time.perf_counter()
    first_warmup_ms = None
    for index in range(warmup):
        function()
        if index == 0:
            sync(device)
            first_warmup_ms = (time.perf_counter() - warmup_start) * 1000
    sync(device)
    warmup_total_ms = (time.perf_counter() - warmup_start) * 1000 if warmup else 0.0
    elapsed = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        function()
        sync(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    p99 = quantile(elapsed, 0.99)
    return {
        "p50_ms": quantile(elapsed, 0.50),
        "p95_ms": quantile(elapsed, 0.95),
        "p99_ms": p99,
        "mean_ms": statistics.mean(elapsed),
        "min_ms": min(elapsed),
        "max_ms": max(elapsed),
        "p99_below_10ms": p99 < 10.0,
        "all_measured_calls_below_10ms": max(elapsed) < 10.0,
        "over_10ms_fraction": sum(value > 10.0 for value in elapsed) / len(elapsed),
        "warmup": warmup,
        "first_warmup_ms": first_warmup_ms,
        "warmup_total_ms": warmup_total_ms,
        "repeats": repeats,
    }


def precision_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])


def check_device_precision(device, precision):
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("supported benchmark devices are cpu, cuda and mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this PyTorch runtime")
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support bf16")
    elif device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable in this PyTorch runtime")
        if precision == "bf16":
            raise ValueError("this benchmark supports fp32/fp16 on MPS; use CUDA or CPU for bf16")
    elif precision == "fp16":
        raise ValueError("CPU fp16 LSTM autocast is unsupported here; use fp32 or bf16")


def synthetic_model(config_path, device):
    import yaml

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    model = LatentContactWorldModel(copy.deepcopy(config))
    # Preserve the configured architecture without opening a pretrained file.
    # An identity transfer normalizer is sufficient for this cost-only run.
    if model.frozen_motion:
        envelope = {
            "normalize_mode": "gaussian", "normalize_lowdim_keys": ["q", "dq", "delta_q"],
            "eps": 1e-6,
            "stats": {key: {"mean": torch.zeros(model.joint_dim), "std": torch.ones(model.joint_dim)}
                      for key in ("q", "dq", "delta_q")},
        }
        model.pretrained_normalizer = copy.deepcopy(envelope)
        model.wm_normalizer = copy.deepcopy(envelope)
    model.codec_ready.fill_(True)
    model.codec_snapshot = "synthetic_benchmark_only"
    model.set_stage("flow")
    return model.to(device).eval().prepare_runtime_normalizers(physical=False), {}


def condition_batch(model, condition_path, device):
    if condition_path:
        values = torch.load(condition_path, map_location="cpu", weights_only=True)
        if not isinstance(values, dict):
            raise ValueError("--condition must contain a mapping of already-normalized tensors")
        required = ("q", "dq", "delta_q", "tau", "action", "history_grid_positions",
                    "action_grid_positions", "future_grid_positions")
        missing = set(required) - values.keys()
        if missing:
            raise ValueError(f"condition is missing {sorted(missing)}")
        values = {key: values[key] for key in required + ("action_mask",) if key in values}
        if any(not torch.is_tensor(value) for value in values.values()):
            raise ValueError("condition values must be tensors")
        if any(value.ndim < 2 or value.shape[0] != 1 for value in values.values()):
            raise ValueError("this latency benchmark requires batch size 1; condition tensors need a batch axis")
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise ValueError("condition tensors must be finite")
        # Model weights remain fp32. Autocast controls operation precision.
        return {key: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                for key, value in values.items()}
    history, future, actions = model.external_history_horizon, model.external_future_horizon, model.action_condition_horizon
    values = {key: torch.randn(1, history, model.joint_dim, device=device)
              for key in ("q", "dq", "delta_q", "tau")}
    values.update(
        action=torch.randn(1, actions, model.action_dim, device=device),
        action_mask=torch.ones(1, actions, device=device, dtype=torch.bool),
        history_grid_positions=torch.arange(1-history, 1, device=device).unsqueeze(0),
        action_grid_positions=(1 + model.grid.ratio * torch.arange(actions, device=device)).unsqueeze(0),
        future_grid_positions=torch.arange(1, future+1, device=device).unsqueeze(0),
    )
    return values


def repeat_encoded(encoded, num_samples):
    result = dict(encoded)
    if num_samples > 1:
        for key in ("history", "action", "action_padding_mask", "future_pe"):
            result[key] = result[key].repeat_interleave(num_samples, 0)
        if "condition_kv_cache" in result:
            result["condition_kv_cache"] = tuple(
                tuple(value.repeat_interleave(num_samples, 0) for value in cache)
                for cache in result["condition_kv_cache"])
    return result


def error_stats(actual, reference):
    difference = actual.detach().float() - reference.detach().float()
    if not torch.isfinite(difference).all():
        raise RuntimeError("nonfinite prediction or paired difference; inference is not numerically valid")
    return {"rmse": difference.square().mean().sqrt().item(),
            "mean_abs": difference.abs().mean().item(), "max_abs": difference.abs().max().item()}


def output_difference(actual, reference, normalizer):
    result = {name: error_stats(actual[key], reference[key]) for name, key in
              (("latent", "latent"), ("q", "q_pred"), ("tau", "tau_pred"),
               ("contact_probability", "contact_probability"))}
    result["contact_argmax_disagreement_fraction"] = (
        actual["contact_state_pred"] != reference["contact_state_pred"]).float().mean().item()
    if normalizer is not None:
        instance, mode = normalizer
        for stream in ("q", "tau"):
            key = stream + "_pred"
            denormalize = getattr(instance, mode + "_denormalize")
            result[stream + "_denormalized_native_units"] = error_stats(
                denormalize(stream, actual[key].float()), denormalize(stream, reference[key].float()))
    return result


def output_normalizer(model, payload):
    envelope = model.wm_normalizer
    if envelope is None and payload.get("normalizer"):
        envelope = {**payload["normalizer"], "normalize_mode": (model.config.get("dataloader") or {}).get("normalize_mode"),
                    "normalize_lowdim_keys": (model.config.get("dataloader") or {}).get("normalize_lowdim_keys", [])}
    if not envelope or envelope.get("normalize_mode") not in {"gaussian", "limit", "quantile"}:
        return None, "q/tau errors are in model coordinates; no output normalizer is available"
    keys = envelope.get("normalize_lowdim_keys", [])
    if not all(key in keys for key in ("q", "tau")):
        return None, "q/tau errors are in model coordinates; both output normalization contracts are required"
    normalizer = Normalizer(envelope.get("stats", {}), eps=float(envelope.get("eps", 1e-6)))
    normalizer.validate(envelope["normalize_mode"], ["q", "tau"], {"q": model.joint_dim, "tau": model.joint_dim})
    note = "denormalized q/tau errors use the saved checkpoint statistics and native data units"
    if envelope["normalize_mode"] == "quantile":
        note += "; quantile clipping cannot be inverted, so this is a mapping into the saved quantile range"
    return (normalizer, envelope["normalize_mode"]), note


def hardware_info(device):
    result = {"device": str(device), "platform": platform.platform(), "machine": platform.machine(),
              "processor": platform.processor(), "torch_version": torch.__version__,
              "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(name=properties.name, total_memory_bytes=properties.total_memory,
                      compute_capability=f"{properties.major}.{properties.minor}", cuda_runtime=torch.version.cuda)
    elif device.type == "mps":
        result["name"] = "Apple MPS (see platform/machine for host details)"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--config", type=Path, help="random initialization; synthetic cost-only benchmark")
    parser.add_argument("--condition", type=Path, help=".pt mapping of already-normalized batch-1 inputs and true relative grids")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--steps", type=step_list, default=step_list("16,4,2"))
    parser.add_argument("--solver", choices=("euler", "heun"), help="default: checkpoint/config solver")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--raw-weights", action="store_true", help="use model_raw when present, rather than saved EMA/model weights")
    parser.add_argument("--compile", action="store_true", help="compile the pure ODE core; Inductor requires CUDA")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")
    parser.add_argument("--compile-backend", choices=("inductor", "eager"), default="inductor",
                        help="eager only checks graph capture; it is not a speed optimization")
    args = parser.parse_args()
    if args.num_samples < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("samples/repeats must be positive; warmup must be nonnegative")
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    check_device_precision(device, args.precision)
    if args.compile and args.compile_backend == "inductor" and device.type != "cuda":
        parser.error("--compile with Inductor is enabled only on CUDA; --compile-backend eager can check graph capture on CPU")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        # fp32 means full float32, without implicit TF32 changes to comparisons.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.checkpoint:
        model, payload = load_latent_checkpoint(args.checkpoint, device=device, use_ema=not args.raw_weights)
    else:
        model, payload = synthetic_model(args.config, device)
    batch = condition_batch(model, args.condition, device)
    noise = torch.randn(1, args.num_samples, model.future_horizon, model.latent_dim, device=device)
    solver = args.solver or model.flow_solver
    normalizer, units_note = output_normalizer(model, payload) if args.checkpoint else (None, "synthetic model has no physical output calibration")
    report = {
        "weights": "loaded_checkpoint" if args.checkpoint else "synthetic_random_initialization",
        "checkpoint_or_config": str((args.checkpoint or args.config).resolve()),
        "condition": str(args.condition.resolve()) if args.condition else "synthetic_random_normalized_inputs",
        "hardware": hardware_info(device), "batch_size": 1, "num_samples": args.num_samples,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": args.precision, "weight_dtype": "float32", "fixed_source_noise_dtype": "float32",
        "solver": solver, "source_mode": model.source_mode,
        "history_tokens": model.history_horizon, "future_tokens": model.future_horizon,
        "action_tokens": model.action_condition_horizon, "hidden_dim": model.hidden_dim, "latent_dim": model.latent_dim,
        "flow_layers": len(model.flow_blocks), "seed": args.seed,
        "units_note": units_note,
        "scope": "full_sample covers 3 LSTMs (including frozen NEXT scale conversion), condition/time caches, source transform, ODE and output codec; excludes robot I/O, external physical preprocessing/normalization, input H2D, action scoring and control",
        "latency_threshold_note": "10 ms is the model budget comparison only; this report cannot certify a complete 100 Hz control loop",
        "quality_note": "paired output differences are numerical comparisons, not task success validation; fewer ODE steps and lower precision require held-out task evaluation",
        "reference": "same checkpoint, same base noise, cached 16-step output at selected precision",
        "timing_note": "synchronized wall-clock latency; input/H2D/noise creation and report comparisons excluded; caches rebuilt within every full_sample call",
        "compile": {"enabled": args.compile, "backend": args.compile_backend if args.compile else None,
                    "mode": args.compile_mode if args.compile and args.compile_backend == "inductor" else None,
                    "note": "compilation covers the pure ODE core; eager backend checks graph capture and cannot certify an Inductor speedup"},
    }
    if payload.get("inverse_gaussian_posttrain") or model.source_mode == "conditional_gaussian":
        report["reference"] += "; this post-trained checkpoint's 16-step output is NOT the original teacher"

    def invoke(function, precision=args.precision):
        with torch.inference_mode(), precision_context(device, precision):
            return function()

    compiled_integrator = None
    if args.compile:
        compile_kwargs = {"backend": args.compile_backend, "fullgraph": True}
        if args.compile_backend == "inductor":
            compile_kwargs["mode"] = args.compile_mode
        compiled_integrator = torch.compile(model.integrate_latent, **compile_kwargs)

    def sample(steps, cached, precision=args.precision, *, compiled=False):
        return invoke(lambda: model.sample(batch, num_samples=args.num_samples, steps=steps, solver=solver,
                                          source_noise=noise, cache_condition_kv=cached,
                                          cache_time_embeddings=cached,
                                          integration_fn=compiled_integrator if compiled else None), precision)

    reference = sample(16, True)
    report["stages"] = {}
    for cached in (False, True):
        label = "cached" if cached else "uncached"
        encoding = lambda: invoke(lambda: model.encode_conditions(batch, cache_condition_kv=cached))
        report["stages"]["condition_encoding_" + label] = measure(
            encoding, device=device, warmup=args.warmup, repeats=args.repeats)
        encoded = repeat_encoded(encoding(), args.num_samples)
        latent = invoke(lambda: model.source_from_noise(noise, encoded)).reshape(
            args.num_samples, model.future_horizon, model.latent_dim)
        embedding = invoke(lambda: model.flow_time_embedding(latent.new_full((args.num_samples,), 0.5))) if cached else None
        velocity = lambda: invoke(lambda: model.velocity(latent, 0.5, encoded, time_embedding=embedding))
        report["stages"]["single_velocity_" + label] = measure(
            velocity, device=device, warmup=args.warmup, repeats=args.repeats)
    raw_latent = reference["raw_latent"].reshape(args.num_samples, model.future_horizon, model.latent_dim)
    report["stages"]["output_codec"] = measure(
        lambda: invoke(lambda: model.decode(raw_latent)), device=device, warmup=args.warmup, repeats=args.repeats)
    report["step_sweep"] = []
    for steps in args.steps:
        cached_output = sample(steps, True)
        uncached_output = sample(steps, False)
        row = {
            "steps": steps, "nfe": cached_output["nfe"],
            "full_sample_cached": measure(lambda: sample(steps, True), device=device, warmup=args.warmup, repeats=args.repeats),
            "full_sample_uncached": measure(lambda: sample(steps, False), device=device, warmup=args.warmup, repeats=args.repeats),
            "cache_difference_same_noise": output_difference(cached_output, uncached_output, normalizer),
            "difference_from_same_checkpoint_16_steps": output_difference(cached_output, reference, normalizer),
        }
        row["cache_p50_speedup"] = row["full_sample_uncached"]["p50_ms"] / row["full_sample_cached"]["p50_ms"]
        if args.compile:
            sync(device)
            cold_start = time.perf_counter()
            compiled_output = sample(steps, True, compiled=True)
            sync(device)
            row["compile_first_call_wall_ms"] = (time.perf_counter() - cold_start) * 1000
            row["compile_first_call_note"] = "first full_sample for this step specialization; includes graph compilation/capture and model execution, not a steady-state latency"
            row["full_sample_compiled"] = measure(
                lambda: sample(steps, True, compiled=True), device=device, warmup=args.warmup, repeats=args.repeats)
            row["compile_difference_from_eager_same_steps_noise_precision"] = output_difference(compiled_output, cached_output, normalizer)
            row["compile_p50_speedup_over_cached_eager"] = row["full_sample_cached"]["p50_ms"] / row["full_sample_compiled"]["p50_ms"]
            fp32_output = cached_output if args.precision == "fp32" else sample(steps, True, "fp32")
            row["compile_difference_from_fp32_eager_same_steps_noise"] = output_difference(compiled_output, fp32_output, normalizer)
        if args.precision != "fp32":
            fp32_output = sample(steps, True, "fp32")
            row["precision_difference_from_fp32_same_steps_noise"] = output_difference(cached_output, fp32_output, normalizer)
        report["step_sweep"].append(row)
    encoded_report = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded_report + "\n", encoding="utf-8")
    print(encoded_report)


if __name__ == "__main__":
    main()
