"""Independently fit an inverse source or distill latent CARS-WM Heun sampling.

The existing LeRobot v3 dataset and normalization contract are reused.  This
script never changes the base checkpoint; it writes a separate deployable one.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from model.pinn_model.inverse_gaussian_source import invert_heun
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from train.nomalizer import Normalizer


def episode_train_indices(dataset, config):
    """Recreate BaseTrainer's episode split and verify its saved index hash."""
    episodes = list(dataset.dataset.meta.episodes)
    if len(episodes) < 2:
        raise ValueError("the original episode split needs at least two episodes")
    names = [int(ep.get("episode_index", i)) for i, ep in enumerate(episodes)]
    train = config.get("train") or {}
    explicit = train.get("val_episode_indices")
    if explicit is None:
        count = min(max(1, int(len(episodes) * float(train.get("val_ratio", 0.1)))), len(episodes)-1)
        order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(int(train.get("seed", 42))))
        val = {names[i] for i in order[:count].tolist()}
    else:
        val = set(int(x) for x in explicit)
    by_start = {int(ep["dataset_from_index"]): names[i] for i, ep in enumerate(episodes)}
    indices = [i for i, raw in enumerate(dataset.valid_indices)
               if by_start[dataset.raw_idx_to_episode_start[raw]] not in val]
    if not indices:
        raise ValueError("episode split has no training windows")
    digest = hashlib.sha256(torch.tensor(indices).numpy().tobytes()).hexdigest()
    return indices, digest


def heun_rollout(velocity, source, encoded, steps, *, return_queries=False):
    state = source
    queries = []
    dt = 1.0 / steps
    for i in range(steps):
        first = velocity(state, i*dt, encoded)
        mid = state + dt*first
        second = velocity(mid, (i+1)*dt, encoded)
        if return_queries:
            queries.extend(((state, i*dt), (mid, (i+1)*dt)))
        state = state + 0.5*dt*(first+second)
    return (state, queries) if return_queries else state


def make_student(base, base_payload, steps, *, source_hidden_dim, temperature, flow_layers=None,
                 mode="source-only"):
    if mode not in {"source-only", "distill-only", "joint"}:
        raise ValueError("unknown post-training mode")
    flow_layers = len(base.flow_blocks) if flow_layers is None else flow_layers
    if not 1 <= flow_layers <= len(base.flow_blocks):
        raise ValueError("student flow layers must be between 1 and the teacher depth")
    if mode == "source-only" and flow_layers != len(base.flow_blocks):
        raise ValueError("source-only must preserve the original Flow depth")
    config = copy.deepcopy(base_payload["config"])
    source_mode = "gaussian" if mode == "distill-only" else "conditional_gaussian"
    config["model"].update(flow_source_mode=source_mode,
                           flow_inference_steps=steps, flow_solver="heun",
                           flow_layers=flow_layers)
    if source_mode == "conditional_gaussian":
        config["model"].update(conditional_source_hidden_dim=source_hidden_dim,
                               conditional_source_temperature=temperature)
    else:
        for key in tuple(config["model"]):
            if key.startswith("conditional_source_"):
                del config["model"][key]
    # This script exports its final raw student, rather than an EMA envelope.
    if isinstance((config.get("train") or {}).get("ema"), dict):
        config["train"]["ema"]["enabled"] = False
    student = LatentContactWorldModel(config)
    state = {key: value for key, value in base.state_dict().items()
             if not (key.startswith("flow_blocks.") and int(key.split(".")[1]) >= flow_layers)}
    missing, unexpected = student.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("source_model.") for key in missing):
        raise ValueError(f"base checkpoint transfer mismatch: missing={missing}, unexpected={unexpected}")
    student.set_stage("flow")
    configure_training_stage(student, "distill" if mode == "distill-only" else "source")
    return student, config


def resolve_training_mode(args):
    defaults = {"source-only": (1000, 0), "distill-only": (0, 1000), "joint": (1000, 1000)}
    source_updates, flow_updates = defaults[args.mode]
    args.source_updates = source_updates if args.source_updates is None else args.source_updates
    args.joint_updates = flow_updates if args.joint_updates is None else args.joint_updates
    if min(args.source_updates, args.joint_updates) < 0:
        raise ValueError("update counts must be nonnegative")
    if args.mode == "source-only" and (args.source_updates < 1 or args.joint_updates != 0):
        raise ValueError("source-only needs positive source updates and zero Flow updates")
    if args.mode == "distill-only" and (args.source_updates != 0 or args.joint_updates < 1):
        raise ValueError("distill-only needs zero source updates and positive Flow updates")
    if args.mode == "joint" and (args.source_updates < 1 or args.joint_updates < 1):
        raise ValueError("joint needs positive updates for both stages")
    if args.mode == "source-only" and args.teacher_steps is not None:
        raise ValueError("source-only inverts the deployment sampler; teacher steps are unused")
    if args.mode == "distill-only" and args.temperature != 1.0:
        raise ValueError("distill-only keeps the standard Gaussian source at temperature 1")
    return args


def configure_training_stage(student, stage):
    """Enable exactly the experimental component; keep dropout disabled."""
    if stage == "source":
        if student.source_mode != "conditional_gaussian":
            raise ValueError("source fitting requires a conditional Gaussian")
        names = ("source_model",)
    elif stage in {"distill", "joint"}:
        if stage == "distill" and student.source_mode != "gaussian":
            raise ValueError("independent distillation requires the standard Gaussian")
        names = tuple(name for name in student.FLOW_MODULES
                      if name != "source_model" or stage == "joint")
    else:
        raise ValueError("unknown training stage")
    student.requires_grad_(False)
    for name in names:
        getattr(student, name).requires_grad_(True)
    # eval() disables dropout without disabling gradients. Repeated velocity
    # queries must represent the same map as the deployment sampler.
    student.eval()
    return names


def flow_distillation_losses(student, teacher, encoded, teacher_encoded, source, *, steps, teacher_steps):
    endpoint, queries = heun_rollout(student.velocity, source, encoded, steps, return_queries=True)
    with torch.no_grad():
        reference = heun_rollout(teacher.velocity, source.detach(), teacher_encoded, teacher_steps)
    endpoint_loss = (endpoint-reference).float().square().mean()
    velocity_losses = []
    for state, time in queries:
        with torch.no_grad():
            target_velocity = teacher.velocity(state.detach(), time, teacher_encoded)
        velocity_losses.append((student.velocity(state, time, encoded)-target_velocity).float().square().mean())
    return endpoint_loss, torch.stack(velocity_losses).mean()


def save_checkpoint(path, student, config, base_payload, args, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": student.MODEL_VERSION,
        "carswm_contract": student.checkpoint_contract(),
        "config": config,
        "model": student.state_dict(),
        "normalizer": base_payload.get("normalizer"),
        "dataloader_filters": base_payload.get("dataloader_filters"),
        "inverse_gaussian_posttrain": {
            "mode": args.mode,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "inverse_solver": "fixed_point_heun" if args.mode != "distill-only" else None,
            "solver": "heun",
            "nfe": 2*args.steps,
            "source_mode": student.source_mode,
            "inference_steps": args.steps,
            "teacher_steps": args.teacher_steps if args.mode != "source-only" else None,
            "teacher_nfe": 2*args.teacher_steps if args.mode != "source-only" else None,
            "teacher_source_policy": {"source-only": None, "distill-only": "shared_standard_gaussian_epsilon",
                                      "joint": "shared_student_source"}[args.mode],
            "inverse_iterations": args.inverse_iterations if args.mode != "distill-only" else None,
            "source_updates": args.source_updates,
            "joint_updates": args.joint_updates,
            "distillation_updates": args.joint_updates,
            "batch_order_seed": args.seed,
            "source_noise_seed": args.seed+1,
            "main_process_augmentation_seed": args.seed+2,
            "dropout_during_posttraining": False,
            "teacher_flow_layers": (base_payload["config"].get("model") or {}).get("flow_layers", 4),
            "student_flow_layers": len(student.flow_blocks),
            "flow_transfer": ("unchanged_teacher_weights" if args.mode == "source-only" else
                              "teacher_prefix_blocks_then_distillation"),
            "metrics": metrics,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("source-only", "distill-only", "joint"), default="source-only",
                        help="default: source-only freezes Flow; distill-only retains N(0,I); joint explicitly combines both")
    parser.add_argument("--steps", type=int, default=4, help="deployed Heun steps")
    parser.add_argument("--flow-layers", type=int, default=None,
                        help="optional smaller Flow depth in distill-only/joint; source-only preserves depth")
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--inverse-iterations", type=int, default=8)
    parser.add_argument("--maximum-cycle-rmse", type=float, default=0.05)
    parser.add_argument("--source-updates", type=int, default=None, help="default: 1000 in source-only/joint, 0 in distill-only")
    parser.add_argument("--flow-updates", "--joint-updates", dest="joint_updates", type=int, default=None,
                        help="Flow distillation updates; default: 0 in source-only, 1000 otherwise")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-hidden-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--source-lr", type=float, default=1e-4)
    parser.add_argument("--flow-lr", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--nll-weight", type=float, default=0.1)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        resolve_training_mode(args)
    except ValueError as error:
        parser.error(str(error))
    if min(args.steps, args.inverse_iterations, args.batch_size, args.source_hidden_dim) < 1:
        parser.error("steps, inverse iterations, batch size and source width must be positive")
    if args.teacher_steps is not None and args.teacher_steps < 1:
        parser.error("teacher steps must be positive")
    if not math.isfinite(args.maximum_cycle_rmse) or args.maximum_cycle_rmse <= 0:
        parser.error("maximum cycle RMSE must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in (args.source_lr, args.flow_lr, args.kl_weight,
                                                  args.nll_weight, args.velocity_weight, args.endpoint_weight)):
        parser.error("learning rates and loss weights must be finite and nonnegative")
    if args.mode != "source-only" and args.endpoint_weight + args.velocity_weight <= 0:
        parser.error("Flow distillation needs a positive endpoint or velocity loss weight")
    args.teacher_steps = args.teacher_steps or None
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base, base_payload = load_latent_checkpoint(args.base_checkpoint, device=device)
    if base.flow_solver != "heun":
        raise ValueError("this post-training path requires a Heun base checkpoint")
    if base.source_mode != "gaussian":
        raise ValueError("base checkpoint must have a standard Gaussian flow source")
    args.teacher_steps = args.teacher_steps or base.flow_inference_steps
    base.requires_grad_(False)
    student, config = make_student(base, base_payload, args.steps,
                                   source_hidden_dim=args.source_hidden_dim,
                                   temperature=args.temperature, flow_layers=args.flow_layers, mode=args.mode)
    student = student.to(device)
    dataset = LatentContactWorldModelDataset(base_payload["config"], compute_normalizer=False)
    saved_normalizer = base_payload.get("normalizer")
    if not saved_normalizer or not saved_normalizer.get("stats"):
        raise ValueError("base checkpoint has no training normalizer")
    dataset.set_normalizer(Normalizer(saved_normalizer["stats"], saved_normalizer.get("eps", 1e-6)))
    indices, digest = episode_train_indices(dataset, base_payload["config"])
    saved_digest = ((base_payload.get("latent_training") or {}).get("data_contract") or {}).get("train_indices_sha256")
    if saved_digest != digest:
        raise ValueError("current LeRobot v3 data/split differs from the base training checkpoint")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=dataset.batch_collate,
                        generator=torch.Generator().manual_seed(args.seed))
    noise_generator = torch.Generator(device=device).manual_seed(args.seed+1)
    # Source-network initialization consumes extra RNG draws. Reset the main
    # process augmentation stream so the independent modes see the same data.
    # Worker streams are seeded independently by the DataLoader generator.
    torch.manual_seed(args.seed+2)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}

    stages = {"source-only": (("source", args.source_updates),),
              "distill-only": (("distill", args.joint_updates),),
              "joint": (("source", args.source_updates), ("joint", args.joint_updates))}[args.mode]
    metrics = {"maximum_cycle_rmse": None if args.mode == "distill-only" else 0.0,
               "train_windows": len(indices), "stage_trainable_modules": {}, "stage_samples_seen": {}}
    for stage, updates in stages:
        names = configure_training_stage(student, stage)
        metrics["stage_trainable_modules"][stage] = list(names)
        metrics["stage_samples_seen"][stage] = 0
        groups = [{"params": list(getattr(student, name).parameters()),
                   "lr": args.source_lr if name == "source_model" else args.flow_lr} for name in names]
        optimizer = torch.optim.AdamW(groups)
        base.eval()
        for update in range(updates):
            batch = next_batch()
            metrics["stage_samples_seen"][stage] += batch["q"].shape[0]
            nll = kl = None
            worst = None
            with torch.no_grad():
                base_encoded = base.encode_conditions(batch, cache_condition_kv=True)
                if stage != "distill":
                    target = base.target_latent(batch).float()
                    inverse, cycle = invert_heun(base.velocity, target, base_encoded,
                                                steps=args.steps, iterations=args.inverse_iterations)
                    if not torch.isfinite(cycle).all():
                        raise RuntimeError("inverse solve produced a nonfinite cycle error")
                    worst = float(cycle.max())
                    metrics["maximum_cycle_rmse"] = max(metrics["maximum_cycle_rmse"], worst)
                    if worst > args.maximum_cycle_rmse:
                        raise RuntimeError(f"inverse cycle RMSE {worst:.5f} exceeds limit; increase iterations or steps")
            # All condition encoders are copied from the teacher and frozen.
            # Reuse their tokens, but student K/V projections must keep gradients.
            encoded = {key: value for key, value in base_encoded.items() if key != "condition_kv_cache"}
            condition = encoded["source_condition"]
            loss = condition.new_zeros(())
            if stage != "distill":
                nll = student.source_model.nll_per_sample(condition, inverse).mean()
                kl = student.source_model.kl_per_sample(condition).mean()
                loss = (1.0 if stage == "source" else args.nll_weight)*nll + args.kl_weight*kl
            if stage != "source":
                epsilon = torch.randn((condition.shape[0], student.future_horizon, student.latent_dim),
                                      device=device, dtype=condition.dtype, generator=noise_generator)
                source = student.source_from_noise(epsilon, encoded)
                endpoint_loss, velocity_loss = flow_distillation_losses(
                    student, base, encoded, base_encoded, source, steps=args.steps, teacher_steps=args.teacher_steps)
                loss = loss + args.endpoint_weight*endpoint_loss + args.velocity_weight*velocity_loss
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite {stage} loss at update {update+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            if (update+1) % 100 == 0 or update+1 == updates:
                print(json.dumps({"stage": stage, "update": update+1, "loss": float(loss.detach()),
                                  "source_nll": float(nll.detach()) if nll is not None else None,
                                  "source_kl": float(kl.detach()) if kl is not None else None,
                                  "cycle_rmse": worst}), flush=True)
        metrics[stage+"_updates"] = updates
    student.eval()
    save_checkpoint(args.output, student, config, base_payload, args, metrics)
    # Confirm the artifact is self-contained and the strict checkpoint contract passes.
    loaded, _ = load_latent_checkpoint(args.output, device="cpu")
    assert loaded.source_model is not None and loaded.flow_inference_steps == args.steps
    print(json.dumps({"checkpoint": str(args.output), "nfe": 2*args.steps,
                      "base_nfe": 2*base.flow_inference_steps, **metrics}), flush=True)


if __name__ == "__main__":
    main()
