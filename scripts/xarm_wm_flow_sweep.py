"""Configuration snapshots, resumable training, fixed-noise evaluation and timings."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAMILIES = ("contact_wm", "latent_wm")
STEPS = (32, 16, 8)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report(root):
    training, evaluation = [], []
    for family in FAMILIES:
        path = root / family / "timing.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        attempts = record.get("training_attempts", [])
        training.append(dict(
            model=family, status=record["status"], optimizer_steps=record["optimizer_steps"],
            codec_steps=record["codec_steps"],
            started_at=attempts[0]["started_at"] if attempts else "",
            completed_at=record.get("training_completed_at", ""),
            measured_training_seconds=round(sum(a.get("seconds", 0) for a in attempts), 3),
            timing_complete=bool(attempts) and all("seconds" in a for a in attempts),
        ))
        for entry in record.get("evaluations", []):
            evaluation.append({"model": family, **{k: v for k, v in entry.items() if k != "metrics"},
                               **entry["metrics"]})
    for name, rows in (("training_times.csv", training), ("evaluation_results.csv", evaluation)):
        path = root / name
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    return training


def prepare(args):
    for family, source in zip(FAMILIES, (args.contact_config, args.latent_config)):
        out = args.root / family
        out.mkdir(parents=True, exist_ok=True)
        path = out / "config.yaml"
        if path.exists():
            if not (out / "timing.json").exists():
                raise RuntimeError(f"Missing timing.json beside existing config: {path}")
            continue  # Resumes use the original snapshot, even if the source YAML changes.
        config = yaml.safe_load(source.read_text())
        data, train = config["dataloader"], config["train"]
        expected = {"high_fps": 100, "expert_fps": 25, "action_condition_horizon": 8,
                    "prediction_horizon": 32, "action_condition_mode": "direct"}
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"{source}: expected {key}={value}, got {data.get(key)}")
        train["output_dir"] = str(out)
        for key in ("resume_from", "resume_checkpoint", "resume"):
            train.pop(key, None)
        inference_steps = int(config["model"]["flow_inference_steps"])
        # Contact probabilistic validation reads rollout.steps even when rollout is disabled.
        train.setdefault("rollout_validation", {})["steps"] = inference_steps
        train.setdefault("checkpoint_visualization", {})["flow_steps"] = inference_steps
        wandb = train.setdefault("wandb", {})
        wandb["name"] = f"{family}_{args.root.name}"
        if os.environ.get("WANDB_MODE"):
            wandb["mode"] = os.environ["WANDB_MODE"]
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        write_json(out / "timing.json", dict(
            model=family, status="pending", source_config=str(source.resolve()),
            config_sha256=sha256(path), optimizer_steps=int(train["max_optimizer_steps"]),
            codec_steps=int(config["codec"]["max_optimizer_steps"]) if family == "latent_wm" else 0,
            training_validation_ode_steps=inference_steps, training_attempts=[], evaluations=[],
        ))
    report(args.root)


def evaluate(trainer, family, steps):
    import torch
    from train.carswm_metrics import (distribution_metrics, contact_confusion_matrix,
                                     contact_macro_f1_from_confusion)

    model = trainer.ema.model if trainer.ema is not None and trainer.ema_use_for_validation else trainer.model
    model.eval()
    cfg = trainer.train_config.get("probabilistic_validation") or {}
    count, limit = int(cfg.get("num_samples", 8)), int(cfg.get("max_batches", 8))
    seed = int((trainer.train_config.get("rollout_validation") or {}).get("source_seed", 1234))
    width = model.latent_dim if family == "latent_wm" else model.flow_dim
    windows, sample_seconds, totals, confusion = 0, 0.0, {}, None
    def sync():
        if torch.device(trainer.device).type == "cuda":
            torch.cuda.synchronize(trainer.device)

    sync()
    started = time.perf_counter()
    with torch.no_grad():
        for index, raw in enumerate(trainer.val_loader):
            if limit and index >= limit:
                break
            batch = trainer.batch_to_device(raw)
            prepared = model.prepare_batch(batch)
            size = batch["q"].shape[0]
            generator = torch.Generator(device="cpu").manual_seed(seed + index * 1009)
            noise = torch.randn(size, count, model.future_horizon, width, generator=generator).to(trainer.device)
            def sample():
                with trainer.autocast_context():
                    return model.sample(batch, num_samples=count, steps=steps,
                                        solver=model.flow_solver, source_noise=noise)
            if index == 0:
                warmup = sample()  # Excluded from sampling_seconds; same warmup for every step count.
                del warmup
            sync()
            before = time.perf_counter()
            result = sample()
            sync()
            sample_seconds += time.perf_counter() - before
            values = distribution_metrics(
                {key: result[key + "_pred"].float() for key in ("q", "tau")},
                {key: prepared[key + "_future"].float() for key in ("q", "tau")},
                result.get("contact_probability"), prepared.get("contact_future"),
            )
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.mean()) * size
            if "contact_probability" in result:
                matrix = contact_confusion_matrix(result["contact_probability"].float(), prepared["contact_future"])
                confusion = matrix if confusion is None else confusion + matrix
            windows += size
    if windows == 0:
        raise RuntimeError("No validation windows available")
    metrics = {key: value / windows for key, value in totals.items()}
    if confusion is not None:
        metrics["contact_macro_f1"] = float(contact_macro_f1_from_confusion(confusion))
    sync()
    return dict(ode_steps=steps, nfe=steps * (2 if model.flow_solver == "heun" else 1),
                solver=model.flow_solver, windows=windows, samples_per_window=count,
                evaluation_seconds=time.perf_counter() - started, sampling_seconds=sample_seconds,
                sampling_ms_per_window=sample_seconds * 1000 / windows, metrics=metrics)


def run(args):
    import torch
    from train.trainer.contact_world_model_train import ContactWorldModelTrainer
    from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer

    out = args.root / args.family
    path = out / "timing.json"
    record = json.loads(path.read_text())
    if record["config_sha256"] != sha256(out / "config.yaml"):
        raise RuntimeError("Saved run config was edited; use a new WM_RUN_ROOT")
    if record["status"] == "complete":
        print(f"Already complete: {args.family}", flush=True)
        return
    record.pop("error", None)
    config = yaml.safe_load((out / "config.yaml").read_text())
    train = config["train"]
    checkpoint = out / "checkpoints" / "latest.pt"
    if checkpoint.exists():
        train["resume_from"] = str(out)
    already_trained = bool(record.get("training_completed_at"))
    if already_trained and not checkpoint.exists():
        raise RuntimeError("Completed training has no latest checkpoint")
    if already_trained:
        train["wandb"]["enabled"] = False
    cls = ContactWorldModelTrainer if args.family == "contact_wm" else LatentContactWorldModelTrainer
    trainer = None
    attempt = None
    def save():
        write_json(path, record)
        report(args.root)
    def stop(*_):
        raise KeyboardInterrupt("Training sweep interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        if not already_trained:
            attempt = dict(started_at=now(), pid=os.getpid(), status="running")
            record["training_attempts"].append(attempt)
            record["status"] = "training"
            save()
            started = time.perf_counter()
            try:
                trainer = cls(config)
                trainer.train()
                if getattr(trainer, "stop_requested", False):
                    raise KeyboardInterrupt("Trainer stopped before completion")
                if trainer.global_step != record["optimizer_steps"]:
                    raise RuntimeError(f"Training stopped before completion: step={trainer.global_step}")
                if args.family == "latent_wm" and not bool(trainer.model.codec_ready):
                    raise RuntimeError("Latent codec was not finalized")
                if not checkpoint.exists():
                    raise RuntimeError("Final checkpoint is missing")
                attempt["status"] = "complete"
                record["training_completed_at"] = now()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                attempt.update(ended_at=now(), seconds=time.perf_counter() - started)
                if trainer is not None:
                    attempt.update(optimizer_steps=trainer.global_step, codec_steps=getattr(trainer, "codec_step", 0))
                save()
        else:
            trainer = cls(config)
            trainer.setup()
        record["status"] = "evaluating"
        record["checkpoint"] = str(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        if record["evaluations"] and record.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("Checkpoint changed after partial evaluation; use a new run directory")
        record["checkpoint_sha256"] = checkpoint_hash
        record["evaluation_weights"] = "ema" if trainer.ema is not None and trainer.ema_use_for_validation else "raw"
        record["validation_indices_sha256"] = hashlib.sha256(
            json.dumps(list(map(int, trainer.val_loader.dataset.indices))).encode()).hexdigest()
        save()
        for steps in STEPS:
            if any(entry["ode_steps"] == steps for entry in record["evaluations"]):
                continue
            entry = evaluate(trainer, args.family, steps)
            record["evaluations"].append(entry)
            print(json.dumps({"model": args.family, **entry}), flush=True)
            save()
        record["status"] = "complete"
        record["completed_at"] = now()
        save()
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if attempt is not None and attempt["status"] == "running":
            attempt["status"] = record["status"]
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--contact-config", type=Path)
    parser.add_argument("--latent-config", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        print(json.dumps(report(args.root), indent=2))
        print(f"Results: {args.root / 'training_times.csv'}; {args.root / 'evaluation_results.csv'}")


if __name__ == "__main__":
    main()
