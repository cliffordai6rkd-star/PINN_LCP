"""Measure old GRU WM checkpoint latency; synthetic inputs imply no quality claim."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pinn_model.contact_scfm import load_contact_checkpoint
from scripts.benchmark_carswm import synthetic_batch
from scripts.posttrain_scfm_latent import file_hash, precision_context, to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--condition", type=Path, help="already normalized tensor batch")
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("output must not overwrite checkpoint")
    device = torch.device(args.device)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("BF16 requires supported CUDA")
    torch.manual_seed(42)
    model, payload = load_contact_checkpoint(args.checkpoint, device=device)
    batch = (synthetic_batch(payload["config"], 1) if args.synthetic else
             torch.load(args.condition, map_location="cpu", weights_only=True))
    batch = to_device(batch, device)
    noise = torch.randn(len(batch["q"]), model.future_horizon, model.flow_dim, device=device)

    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)

    rows = []
    with torch.inference_mode(), precision_context(device, args.precision):
        for steps, solver, cached in [(32, "heun", False), (4, "heun", False),
                                      (4, "euler", False), (4, "euler", True)]:
            def request():
                return model.predict(batch, steps=steps, solver=solver, source_noise=noise,
                                     cache_condition_kv=cached)
            for _ in range(args.warmup): request()
            sync()
            times = []
            for _ in range(args.repeats):
                start = perf_counter()
                output = request()
                sync()
                times.append((perf_counter()-start)*1000)
            assert torch.isfinite(output["flow_state_pred"]).all()
            rows.append({"steps": steps, "solver": solver, "nfe": steps*(2 if solver == "heun" else 1),
                         "condition_kv_cache": cached, "mean_ms": float(np.mean(times)),
                         "p50_ms": float(np.median(times)), "p95_ms": float(np.quantile(times, .95))})
    result = {"checkpoint": str(args.checkpoint), "sha256": file_hash(args.checkpoint),
              "model_version": model.MODEL_VERSION, "parameters": sum(p.numel() for p in model.parameters()),
              "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch_version": torch.__version__, "precision": args.precision,
              "synthetic_inputs": args.synthetic, "batch_size": len(batch["q"]), "samples": 1,
              "quality_evaluated": False,
              "scope": "normalized GPU tensors -> encoder + flow + contact head, synchronized wall time; excludes IO/normalization/control",
              "warmup": args.warmup, "repeats": args.repeats, "variants": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
