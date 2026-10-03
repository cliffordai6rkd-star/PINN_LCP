"""Fit an inverse Gaussian source and distill latent CARS-WM to fewer Heun steps.

The existing LeRobot v3 dataset and normalization contract are reused.  This
script never changes the base checkpoint; it writes a separate deployable one.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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


def make_student(base, base_payload, steps, *, source_hidden_dim, temperature):
    config = copy.deepcopy(base_payload["config"])
    config["model"].update(flow_source_mode="conditional_gaussian",
                           flow_inference_steps=steps, flow_solver="heun",
                           conditional_source_hidden_dim=source_hidden_dim,
                           conditional_source_temperature=temperature)
    student = LatentContactWorldModel(config)
    missing, unexpected = student.load_state_dict(base.state_dict(), strict=False)
    if unexpected or any(not key.startswith("source_model.") for key in missing):
        raise ValueError(f"base checkpoint transfer mismatch: missing={missing}, unexpected={unexpected}")
    student.set_stage("flow")
    return student, config


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
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "inverse_solver": "fixed_point_heun",
            "inference_steps": args.steps,
            "teacher_steps": args.teacher_steps,
            "inverse_iterations": args.inverse_iterations,
            "source_updates": args.source_updates,
            "joint_updates": args.joint_updates,
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
    parser.add_argument("--steps", type=int, default=4, help="deployed Heun steps")
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--inverse-iterations", type=int, default=8)
    parser.add_argument("--maximum-cycle-rmse", type=float, default=0.05)
    parser.add_argument("--source-updates", type=int, default=1000)
    parser.add_argument("--joint-updates", type=int, default=1000)
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
    if min(args.steps, args.inverse_iterations, args.batch_size, args.source_hidden_dim) < 1:
        parser.error("steps, inverse iterations, batch size and source width must be positive")
    if min(args.source_updates, args.joint_updates) < 0 or args.source_updates + args.joint_updates == 0:
        parser.error("at least one post-training stage must have positive updates")
    if args.teacher_steps is not None and args.teacher_steps < 1:
        parser.error("teacher steps must be positive")
    if not math.isfinite(args.maximum_cycle_rmse) or args.maximum_cycle_rmse <= 0:
        parser.error("maximum cycle RMSE must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in (args.source_lr, args.flow_lr, args.kl_weight,
                                                  args.nll_weight, args.velocity_weight, args.endpoint_weight)):
        parser.error("learning rates and loss weights must be finite and nonnegative")
    args.teacher_steps = args.teacher_steps or None
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base, base_payload = load_latent_checkpoint(args.base_checkpoint, device=device)
    if base.flow_solver != "heun":
        raise ValueError("this post-training path requires a Heun base checkpoint")
    if base.source_mode != "gaussian":
        raise ValueError("base checkpoint must have a standard Gaussian flow source")
    args.teacher_steps = args.teacher_steps or base.flow_inference_steps
    student, config = make_student(base, base_payload, args.steps,
                                   source_hidden_dim=args.source_hidden_dim,
                                   temperature=args.temperature)
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
                        num_workers=args.num_workers, collate_fn=dataset.batch_collate)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw = next(iterator)
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}

    # Conditions are frozen: inverse labels and source training remain stable.
    student.requires_grad_(False)
    student.source_model.requires_grad_(True)
    source_optimizer = torch.optim.AdamW(student.source_model.parameters(), lr=args.source_lr)
    metrics = {"maximum_cycle_rmse": 0.0, "train_windows": len(indices)}
    for stage, updates in (("source", args.source_updates), ("joint", args.joint_updates)):
        if stage == "joint":
            for name in student.FLOW_MODULES:
                if name != "source_model":
                    getattr(student, name).requires_grad_(True)
            optimizer = torch.optim.AdamW([
                {"params": student.source_model.parameters(), "lr": args.source_lr},
                {"params": (p for name in student.FLOW_MODULES if name != "source_model"
                            for p in getattr(student, name).parameters()), "lr": args.flow_lr},
            ])
        else:
            optimizer = source_optimizer
        student.train()
        base.eval()
        for update in range(updates):
            batch = next_batch()
            with torch.no_grad():
                base_encoded = base.encode_conditions(batch)
                target = base.target_latent(batch).float()
                inverse, cycle = invert_heun(base.velocity, target, base_encoded,
                                              steps=args.steps, iterations=args.inverse_iterations)
                worst = float(cycle.max())
                metrics["maximum_cycle_rmse"] = max(metrics["maximum_cycle_rmse"], worst)
                if worst > args.maximum_cycle_rmse:
                    raise RuntimeError(f"inverse cycle RMSE {worst:.5f} exceeds limit; increase iterations or steps")
            encoded = student.encode_conditions(batch)
            condition = encoded["source_condition"]
            nll = student.source_model.nll_per_sample(condition, inverse).mean()
            kl = student.source_model.kl_per_sample(condition).mean()
            loss = nll + args.kl_weight * kl if stage == "source" else args.nll_weight*nll + args.kl_weight*kl
            if stage == "joint":
                epsilon = torch.randn_like(inverse)
                source = student.source_from_noise(epsilon, encoded)
                endpoint, queries = heun_rollout(student.velocity, source, encoded, args.steps,
                                                 return_queries=True)
                with torch.no_grad():
                    reference = heun_rollout(base.velocity, source.detach(), base_encoded, args.teacher_steps)
                endpoint_loss = (endpoint-reference).float().square().mean()
                velocity_losses = []
                for state, time in queries:
                    with torch.no_grad():
                        teacher_velocity = base.velocity(state.detach(), time, base_encoded)
                    velocity_losses.append((student.velocity(state, time, encoded) -
                                            teacher_velocity).float().square().mean())
                velocity_loss = torch.stack(velocity_losses).mean()
                loss = loss + args.endpoint_weight*endpoint_loss + args.velocity_weight*velocity_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            if (update+1) % 100 == 0 or update+1 == updates:
                print(json.dumps({"stage": stage, "update": update+1, "loss": float(loss.detach()),
                                  "source_nll": float(nll.detach()), "source_kl": float(kl.detach()),
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
