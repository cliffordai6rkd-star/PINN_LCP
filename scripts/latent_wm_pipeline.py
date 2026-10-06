"""Resolve the three-job recipe and verify completed latent WM checkpoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch
import yaml

from model.pinn_model.latent_contact_world_model import CONDITIONAL_MODEL_VERSION, load_latent_checkpoint

ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIGS = {
    "nero_all": "nero_pretrain_all.yaml",
    "xarm_peel_cucumber": "xarm_peel_cucumber_sft.yaml",
    "xarm_erase_board": "xarm_erase_board_sft.yaml",
}


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def normalize_resume_config(config):
    value = copy.deepcopy(config)
    value.setdefault("train", {}).pop("resume_from", None)
    # Canonical defaults injected by the independent latent trainer.
    data = value.setdefault("dataloader", {})
    data.setdefault("normalize_mode", "gaussian")
    data.setdefault("normalize_lowdim_keys", ["q", "dq", "delta_q", "tau", "action"])
    return value


def resolve_configs(run_root, *, codec_steps=20000, pretrain_steps=250000, sft_steps=50000,
                    pretrain_batch=256, sft_batch=128, pretrain_device="cuda:0",
                    peel_device="cuda:0", erase_device="cuda:0", wandb_mode=None):
    if min(codec_steps, pretrain_steps, sft_steps, pretrain_batch, sft_batch) < 1:
        raise ValueError("Training budgets and batch sizes must be positive")
    run_root = Path(run_root).resolve()
    shared = run_root/"shared/pretrained_for_sft.pt"
    devices = {"nero_all": pretrain_device, "xarm_peel_cucumber": peel_device, "xarm_erase_board": erase_device}
    configs = {}
    for task, name in TASK_CONFIGS.items():
        config = yaml.safe_load((ROOT/"config/train_cfg/latent_wm"/name).read_text())
        pretrain = task == "nero_all"
        train = config["train"]
        train.update(output_dir=str(run_root/task), device=devices[task], stage="all" if pretrain else "sft",
                     batch_size=pretrain_batch if pretrain else sft_batch, gradient_every=1,
                     lr=1e-4 if pretrain else 3e-5, max_optimizer_steps=pretrain_steps if pretrain else sft_steps,
                     checkpoint_every_steps=min(50000 if pretrain else 10000, pretrain_steps if pretrain else sft_steps))
        train["scheduler"]["warmup_steps"] = min(500, (min(codec_steps, pretrain_steps) if pretrain else sft_steps)-1)
        config["codec"].update(lr=3e-4, max_optimizer_steps=codec_steps, checkpoint_path=None)
        if not pretrain:
            config["sft"].update(pretrained_checkpoint=str(shared), flow_mode="frozen", use_ema=True)
        if wandb_mode:
            if wandb_mode not in {"online", "offline", "disabled"}:
                raise ValueError("WANDB_MODE must be online, offline or disabled")
            train["wandb"]["mode"] = wandb_mode
            if wandb_mode == "disabled":
                train["wandb"]["enabled"] = False
        configs[task] = config
    return configs


def validate_sources(configs):
    for task, config in configs.items():
        for source in config["train_data"]["sources"]:
            info_path = ROOT/source["root"]/"meta/info.json"
            info = json.loads(info_path.read_text())
            if info.get("fps") != 100:
                raise ValueError(f"{source['root']} must contain the converted 100 Hz data")
            required = {"observation.joint", "observation.velocity", "observation.delta_q", "observation.torque",
                        "action.ee_pose", "timing.state_timestamp_ns", "timing.action_index", "timing.action_anchor_timestamp_ns"}
            if task == "nero_all":
                required |= {"observation.tau_ext", "observation.tau_label_valid", "observation.contact_phase"}
                manifest = json.loads(info_path.with_name("world_model_timeline.json").read_text())
                teacher = ROOT/"outputs/tau_free_sequence/nero/epoch_124_val_tau_mse_nm2_0.005756.pt"
                teacher_hash = hashlib.sha256(teacher.read_bytes()).hexdigest()
                if (manifest.get("torque_labels") or {}).get("checkpoint_sha256") != teacher_hash:
                    raise ValueError(f"{source['root']} torque labels use another Nero teacher")
                if (manifest.get("contact_labels") or {}).get("precontact_duration_s") != 1.:
                    raise ValueError(f"{source['root']} alignment prefix must be one second")
                if (manifest.get("contact_labels") or {}).get("contact_threshold") != config["contact_gate"]["contact_threshold"]:
                    raise ValueError(f"{source['root']} exported contact threshold differs from the training recipe")
            if required-set(info["features"]):
                raise ValueError(f"{source['root']} missing WM features: {sorted(required-set(info['features']))}")
        teacher = (config["dataloader"].get("tau_ext_generation") or {}).get("checkpoint")
        if teacher and not (ROOT/teacher).is_file():
            raise FileNotFoundError(f"Missing xArm torque teacher: {teacher}")


def prepare(run_root, *, sft_parallel=True, **kwargs):
    run_root = Path(run_root).resolve()
    configs = resolve_configs(run_root, **kwargs)
    validate_sources(configs)
    # Reject a changed recipe before touching existing training state/config files.
    for task, config in configs.items():
        for previous in (run_root/"configs"/TASK_CONFIGS[task], run_root/task/"resolved_config.yaml"):
            if previous.exists() and normalize_resume_config(yaml.safe_load(previous.read_text())) != normalize_resume_config(config):
                raise ValueError(f"Existing run configuration differs: {previous}; choose a new LATENT_RUN_ROOT")
    for task, config in configs.items():
        write_atomic(run_root/"configs"/TASK_CONFIGS[task], yaml.safe_dump(config, sort_keys=False))
    order = ["nero_all", ["xarm_peel_cucumber", "xarm_erase_board"]] if sft_parallel else list(TASK_CONFIGS)
    plan = {"run_root": str(run_root), "order": order,
            "tasks": {task: {"config": str(run_root/"configs"/TASK_CONFIGS[task]),
                "output": config["train"]["output_dir"], "batch_size": config["train"]["batch_size"],
                "lr": config["train"]["lr"], "optimizer_steps": config["train"]["max_optimizer_steps"],
                "device": config["train"]["device"], "stage": config["train"]["stage"]} for task, config in configs.items()},
            "codec": {"batch_size": configs["nero_all"]["train"]["batch_size"], "lr": 3e-4,
                      "optimizer_steps": configs["nero_all"]["codec"]["max_optimizer_steps"]},
            "position_encoding": "contact_wm_learned_index", "window": {"history": 50, "future": 32, "action": 8}}
    write_atomic(run_root/"pipeline_plan.json", json.dumps(plan, indent=2)+"\n")
    print(json.dumps(plan, indent=2))


def completed_checkpoint(run_root, task):
    run_root = Path(run_root).resolve()
    config = yaml.safe_load((run_root/"configs"/TASK_CONFIGS[task]).read_text())
    output = Path(config["train"]["output_dir"])
    status_path = output/"status.json"
    if not status_path.is_file():
        return None
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        return None
    steps = config["train"]["max_optimizer_steps"]
    stage = "flow" if task == "nero_all" else "sft"
    if status.get("stage") != stage or status.get("flow_step") != steps or not status.get("codec_ready"):
        raise ValueError(f"Completed status has incompatible progress: {status_path}")
    checkpoint = output/"checkpoints"/f"step_{steps:08d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing completed scheduled checkpoint: {checkpoint}")
    model, payload = load_latent_checkpoint(checkpoint, use_ema=True)
    if payload["model_version"] != CONDITIONAL_MODEL_VERSION or not model.learned_positions or not model.codec_ready:
        raise ValueError("Completed checkpoint is not the finalized Contact-WM-position latent v2 model")
    if not torch.isfinite(model.latent_mean).all() or not torch.isfinite(model.latent_std).all() or (model.latent_std <= 0).any():
        raise ValueError("Completed checkpoint has invalid latent statistics")
    if normalize_resume_config(payload["config"]) != normalize_resume_config(config):
        raise ValueError("Completed checkpoint configuration differs from the pipeline recipe")
    state = payload["latent_training"]
    if payload["global_step"] != steps or state["stage"] != stage or state["codec_step"] != config["codec"]["max_optimizer_steps"]:
        raise ValueError("Completed checkpoint stage/budget mismatch")
    validation = state.get("final_validation")
    if not validation or any(not math.isfinite(float(value)) for value in validation.values() if isinstance(value, (int, float))):
        raise ValueError("Completed checkpoint needs finite final validation metrics")
    if task != "nero_all":
        if model.sft_flow_mode != "frozen" or any(p.requires_grad for name in model.FLOW_MODULES for p in getattr(model, name).parameters()):
            raise ValueError("SFT checkpoint does not implement the frozen Flow policy")
    return checkpoint


def seal_pretrained(run_root):
    run_root = Path(run_root).resolve()
    source = completed_checkpoint(run_root, "nero_all")
    if source is None:
        raise RuntimeError("Nero pretraining is incomplete; SFT cannot start")
    destination = run_root/"shared/pretrained_for_sft.pt"
    seal_path = destination.with_suffix(".json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if seal_path.exists():
        seal = json.loads(seal_path.read_text())
        if seal["sha256"] != digest:
            raise ValueError("Pretrained checkpoint changed after sealing; use a new run root for a new transfer experiment")
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("Shared pretrained checkpoint was modified")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".pt.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    elif hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise ValueError("Shared pretrained checkpoint belongs to another source")
    write_atomic(seal_path, json.dumps({"source": str(source), "checkpoint": str(destination), "sha256": digest}, indent=2)+"\n")
    print(f"Verified shared SFT source: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "seal"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_CONFIGS))
    for name, default in (("codec-steps", 20000), ("pretrain-steps", 250000), ("sft-steps", 50000),
                          ("pretrain-batch", 256), ("sft-batch", 128)):
        parser.add_argument("--"+name, type=int, default=default)
    for name in ("pretrain-device", "peel-device", "erase-device"):
        parser.add_argument("--"+name, default="cuda:0")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--sft-parallel", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        names = ("codec_steps", "pretrain_steps", "sft_steps", "pretrain_batch", "sft_batch",
                 "pretrain_device", "peel_device", "erase_device", "wandb_mode", "sft_parallel")
        prepare(args.run_root, **{name: getattr(args, name) for name in names})
    elif args.action == "check":
        if args.task is None:
            parser.error("check requires --task")
        try:
            checkpoint = completed_checkpoint(args.run_root, args.task)
        except Exception as exc:
            print(f"Invalid completed checkpoint: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        if checkpoint is None:
            raise SystemExit(1)
        print(f"Verified completed {args.task}: {checkpoint}")
    else:
        seal_pretrained(args.run_root)


if __name__ == "__main__":
    main()
