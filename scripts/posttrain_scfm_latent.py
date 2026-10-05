"""Independent SCFM post-training of a Gaussian-source latent CARS-WM checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_scfm import load_contact_checkpoint, make_contact_student
from model.pinn_model.latent_contact_world_model import load_latent_checkpoint
from model.pinn_model.latent_pretrained import convert_scale, normalizer_envelope
from model.pinn_model.scfm_distillation import (
    SCFMSettings, frozen_snapshot, sample_schedule, scfm_loss, update_flow_ema,
)
from scripts.posttrain_inverse_gaussian_latent import episode_train_indices, make_student
from train.carswm_metrics import (
    contact_confusion_matrix, contact_macro_f1_from_confusion, distribution_metrics, energy_score,
)
from train.nomalizer import Normalizer

UPSTREAM = "41d435d36cfcdeb945cb9562ddb87ca5e5cd285a"


def index_hash(indices):
    return hashlib.sha256(torch.tensor(indices, dtype=torch.int64).numpy().tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def precision_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def relocate_data(config, root, repo_id):
    """Relocate a single dataset without changing its sampling contract."""
    result = copy.deepcopy(config)
    sources = (result.get("train_data") or {}).get("sources")
    if sources and (root or repo_id):
        if len(sources) != 1 or not isinstance(sources[0], dict):
            raise ValueError("CLI relocation requires one mapping in train_data.sources")
        destination = sources[0]
    else:
        destination = result["dataloader"]
    if root:
        destination["root"] = str(root.resolve())
    if repo_id:
        destination["repo_id"] = repo_id
    return result


class BatchStream:
    """Deterministic epoch orders with an explicit next-batch cursor for resume."""
    def __init__(self, dataset, indices, settings, workers, state=None):
        self.dataset, self.indices = dataset, indices
        self.settings, self.workers = settings, workers
        self.epoch, self.cursor = (state["epoch"], state["cursor"]) if state else (0, 0)
        self.iterator = None

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def next(self):
        count = (len(self.indices)+self.settings.batch_size-1)//self.settings.batch_size
        if self.cursor == count:
            self.epoch += 1
            self.cursor = 0
            self.iterator = None
        if self.iterator is None:
            order = torch.randperm(len(self.indices), generator=torch.Generator().manual_seed(
                self.settings.seed+self.epoch)).tolist()
            batches = [order[start:start+self.settings.batch_size]
                       for start in range(0, len(order), self.settings.batch_size)]
            loader = DataLoader(Subset(self.dataset, self.indices), batch_sampler=batches[self.cursor:],
                                num_workers=self.workers, collate_fn=self.dataset.batch_collate,
                                worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(self.settings.seed+self.epoch))
            self.iterator = iter(loader)
        value = next(self.iterator)
        self.cursor += 1
        return value


@torch.no_grad()
def validate(student, teacher, loader, settings, device, envelope):
    """Fixed held-out noise: original fine teacher, original coarse, SCFM coarse.

    Metrics use true targets, not just teacher agreement. Contact F1 is computed
    from the global confusion matrix; other metrics are window-weighted means.
    """
    generator = torch.Generator(device=device).manual_seed(settings.seed+10000)
    totals, confusion, windows = {}, {}, 0
    contact_available = None
    variants = {"teacher_fine": (teacher, settings.reference_steps, teacher.flow_solver),
                "original_coarse": (teacher, settings.steps, "euler"),
                "scfm_coarse": (student, settings.steps, "euler")}
    for batch_index, raw in enumerate(loader):
        if batch_index >= settings.validation_batches:
            break
        batch = to_device(raw, device)
        prepared = teacher.prepare_batch(batch)
        has_contact = "contact_future" in prepared
        if contact_available is not None and contact_available != has_contact:
            raise ValueError("inconsistent availability of validation contact labels")
        contact_available = has_contact
        size = batch["q"].shape[0]
        legacy = hasattr(teacher, "flow_dim") and not hasattr(teacher, "latent_dim")
        dimension = teacher.flow_dim if legacy else teacher.latent_dim
        state_key = "flow_state_pred" if legacy else "latent"
        noise = torch.randn(size, settings.validation_samples, teacher.future_horizon, dimension,
                            device=device, generator=generator)
        with precision_context(device, settings.precision):
            outputs = {name: model.sample(batch, steps=steps, solver=solver,
                                         num_samples=settings.validation_samples, source_noise=noise)
                       for name, (model, steps, solver) in variants.items()}
        for name, output in outputs.items():
            metrics = distribution_metrics(
                {key: output[key+"_pred"].float() for key in ("q", "tau")},
                {key: prepared[key+"_future"].float() for key in ("q", "tau")},
                output["contact_probability"].float() if has_contact else None,
                prepared.get("contact_future"))
            metrics.pop("contact_macro_f1", None)
            p = output["contact_probability"].float().clamp_min(1e-8)
            reference_p = outputs["teacher_fine"]["contact_probability"].float().clamp_min(1e-8)
            metrics["paired_teacher_contact_kl"] = (reference_p*(reference_p.log()-p.log())).sum(-1).mean((1, 2))
            metrics["paired_teacher_flow_rmse" if legacy else "paired_teacher_latent_rmse"] = (
                output[state_key].float()-outputs["teacher_fine"][state_key].float()
            ).square().mean(dim=(1, 2, 3)).sqrt()
            # Physical metrics are reported per stream, never combining radians
            # and Nm into an unlabelled distance. Clipped quantiles cannot invert.
            if envelope.get("normalize_mode") != "quantile":
                for key in ("q", "tau"):
                    samples = convert_scale(key, output[key+"_pred"].float(), envelope, inverse=True)
                    target = convert_scale(key, prepared[key+"_future"].float(), envelope, inverse=True)
                    metrics[key+"_physical_energy_score"] = energy_score(samples, target)
                    metrics[key+"_physical_mean_rmse"] = (
                        samples.mean(1)-target).square().mean(dim=(1, 2)).sqrt()
                    reference_samples = convert_scale(key, outputs["teacher_fine"][key+"_pred"].float(), envelope, inverse=True)
                    metrics[key+"_physical_paired_teacher_rmse"] = (samples-reference_samples).square().mean((1, 2, 3)).sqrt()
            for key, values in metrics.items():
                if not torch.isfinite(values).all():
                    raise RuntimeError(f"nonfinite validation metric: {name}.{key}")
                totals.setdefault(name, {}).setdefault(key, 0.0)
                totals[name][key] += float(values.sum())
            if has_contact:
                matrix = contact_confusion_matrix(output["contact_probability"], prepared["contact_future"])
                confusion[name] = confusion.get(name, torch.zeros_like(matrix)) + matrix
        windows += size
    if not windows:
        raise ValueError("no held-out windows available")
    report = {name: {key: value/windows for key, value in values.items()} for name, values in totals.items()}
    for name, (_, steps, solver) in variants.items():
        if contact_available:
            report[name].update(contact_macro_f1=float(contact_macro_f1_from_confusion(confusion[name])),
                                contact_confusion=confusion[name].cpu().tolist())
        report[name].update(solver=solver, steps=steps,
                            nfe=steps*(2 if solver == "heun" else 1))
    return {"windows": windows, "samples_per_window": settings.validation_samples,
            "contact_labels_available": contact_available,
            "space": "normalized q/tau except explicitly physical metrics",
            "noise_policy": "fixed paired standard Gaussian", "variants": report}


def save_checkpoint(path, student, config, teacher_payload, metadata, training_state):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_version": student.MODEL_VERSION, "carswm_contract": student.checkpoint_contract(),
               "config": config, "model": student.state_dict(), "normalizer": teacher_payload["normalizer"],
               "dataloader_filters": teacher_payload.get("dataloader_filters"),
               "latent_training": teacher_payload.get("latent_training"),
               "scfm_posttrain": copy.deepcopy(metadata), "scfm_training_state": training_state}
    for key in ("sample_rate_hz", "derived_target_config", "tau_label_contract"):
        if key in teacher_payload:
            payload[key] = copy.deepcopy(teacher_payload[key])
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main(architecture="latent"):
    if architecture not in {"latent", "contact"}:
        raise ValueError("unknown SCFM architecture")
    legacy = architecture == "contact"
    load_checkpoint = load_contact_checkpoint if legacy else load_latent_checkpoint
    parser = argparse.ArgumentParser(description=("SCFM post-training of schema-10 GRU CARS-WM" if legacy else __doc__))
    default_config = "contact_cwm_scfm.yaml" if legacy else "latent_cwm_scfm.yaml"
    parser.add_argument("--config", type=Path, default=ROOT/"config/train_cfg"/default_config)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, help="relocate the original LeRobot v3 data; preserve preprocessing")
    parser.add_argument("--repo-id", help="dataset repo_id at the relocated root")
    parser.add_argument("--interpolate-vla", action="store_true", help="explicit approximate 25->100 Hz GRU pilot; no true contact labels")
    parser.add_argument("--unlabeled-native", action="store_true", help="GRU-only native q/tau supervision while contact label assets are unavailable")
    parser.add_argument("--save-condition", type=Path, help="save one normalized held-out batch for paired latency runs")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--few-shot-windows", type=int)
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate --resume candidate; no training")
    args = parser.parse_args()
    config_input = yaml.safe_load(args.config.read_text()) or {}
    values = dict(config_input.get("scfm") or {})
    for key in ("updates", "steps", "precision", "few_shot_windows"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    try:
        settings = SCFMSettings(**values).validate()
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.num_workers < 0 or (args.evaluate_only and not args.resume):
        parser.error("workers must be nonnegative; evaluate-only requires --resume")
    if args.interpolate_vla and (not legacy or not args.data_root):
        parser.error("--interpolate-vla requires the GRU contact entry point and --data-root")
    if args.unlabeled_native and (not legacy or args.interpolate_vla):
        parser.error("--unlabeled-native is GRU-only and cannot be combined with interpolation")
    if args.output.resolve() == args.base_checkpoint.resolve():
        parser.error("output must not overwrite the original teacher")
    if args.evaluate_only and args.output.resolve() == args.resume.resolve():
        parser.error("evaluation JSON must not overwrite the resumed checkpoint")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("supported devices: cpu and cuda")
    if settings.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 requires a CUDA device with BF16 support; use fp32 on CPU")
    torch.manual_seed(settings.seed)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    teacher, teacher_payload = load_checkpoint(args.base_checkpoint, device=device)
    teacher.eval().requires_grad_(False)
    if (teacher.flow_source_mode if legacy else teacher.source_mode) != "gaussian":
        raise ValueError("SCFM is an independent Gaussian-source route; supply the original teacher")
    if teacher_payload.get("scfm_posttrain") or teacher_payload.get("inverse_gaussian_posttrain"):
        raise ValueError("use an original Flow checkpoint, not another post-training experiment")
    augmentation = (teacher_payload["config"].get("dataloader") or {}).get("action_augmentation") or {}
    if augmentation.get("enabled", False):
        raise ValueError("SCFM's repeatable held-out validation requires action augmentation disabled")
    if legacy:
        student, config = make_contact_student(teacher, teacher_payload, settings.steps)
    else:
        student, config = make_student(teacher, teacher_payload, settings.steps, source_hidden_dim=256,
                                      temperature=1.0, mode="distill-only")
    config["model"]["flow_solver"] = "euler"
    student.flow_solver = "euler"
    student.to(device)
    fast, slow = frozen_snapshot(student), frozen_snapshot(student)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad),
                                 lr=settings.learning_rate)
    data_config = relocate_data(teacher_payload["config"], args.data_root, args.repo_id)
    dataset_class = ContactWorldModelDataset if legacy else LatentContactWorldModelDataset
    if args.interpolate_vla:
        from data_process.interpolated_contact_dataset import InterpolatedContactDataset
        dataset = InterpolatedContactDataset(data_config, args.data_root)
    elif args.unlabeled_native:
        from data_process.native_continuous_contact_dataset import NativeContinuousContactDataset
        dataset = NativeContinuousContactDataset(data_config, compute_normalizer=False)
    else:
        dataset = dataset_class(data_config, compute_normalizer=False)
    provenance = copy.deepcopy(getattr(dataset, "provenance", {"type": "original_wm_dataset"}))
    print(json.dumps({"dataset_provenance": provenance}), flush=True)
    saved = teacher_payload.get("normalizer") or {}
    if not saved.get("stats"):
        raise ValueError("original checkpoint must contain its training normalizer")
    normalizer = Normalizer(saved["stats"], saved.get("eps", 1e-6))
    dataset.set_normalizer(normalizer)
    envelope = normalizer_envelope(normalizer, teacher_payload["config"])
    train_indices, digest = episode_train_indices(dataset, teacher_payload["config"])
    train_set = set(train_indices)
    val_indices = [index for index in range(len(dataset)) if index not in train_set]
    contract = (teacher_payload.get("latent_training") or {}).get("data_contract") or {}
    if not legacy and (contract.get("train_indices_sha256") != digest or contract.get("val_indices_sha256") != index_hash(val_indices)):
        raise ValueError("LeRobot v3 episode split differs from original checkpoint (train/val hashes)")
    if contract.get("raw_valid_indices_sha256") and contract["raw_valid_indices_sha256"] != index_hash(dataset.valid_indices):
        raise ValueError("LeRobot v3 retained raw windows differ from original checkpoint")
    if not val_indices:
        raise ValueError("SCFM requires held-out validation episodes")
    if settings.few_shot_windows:
        if settings.few_shot_windows > len(train_indices):
            raise ValueError("few-shot window count exceeds original training split")
        selection = torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(settings.seed+3))
        train_indices = [train_indices[i] for i in selection[:settings.few_shot_windows].tolist()]
    val_order = torch.randperm(len(val_indices), generator=torch.Generator().manual_seed(settings.seed+10002))
    evaluated_val_indices = [val_indices[i] for i in val_order[:settings.validation_batches*settings.batch_size].tolist()]
    if args.save_condition:
        protected = [args.base_checkpoint, args.output] + ([args.resume] if args.resume else [])
        if args.save_condition.resolve() in {path.resolve() for path in protected}:
            parser.error("saved condition must not overwrite a checkpoint or evaluation output")
        args.save_condition.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset.batch_collate([dataset[evaluated_val_indices[0]]]), args.save_condition)
    val_loader = DataLoader(Subset(dataset, evaluated_val_indices), batch_size=settings.batch_size, shuffle=False,
                            collate_fn=dataset.batch_collate, num_workers=0,
                            generator=torch.Generator().manual_seed(settings.seed+10001))
    metadata = {"method": "scfm_dual_target_velocity", "upstream_commit": UPSTREAM,
                "architecture": architecture, "dataset_config": copy.deepcopy(data_config.get("train_data") or data_config["dataloader"]),
                "original_split_hash_verified": not legacy,
                "dataset_provenance": provenance,
                "upstream_url": "https://github.com/caitree/scfm", "base_checkpoint": str(args.base_checkpoint.resolve()),
                "base_sha256": file_hash(args.base_checkpoint), "settings": settings.as_dict(),
                "time_convention": "0=noise, 1=data; s=1-sigma_flux", "source_mode": "gaussian",
                "export_weights": "raw_student", "solver": "euler", "nfe": settings.steps,
                "train_indices_sha256": digest, "val_indices_sha256": index_hash(val_indices),
                "selected_train_indices_sha256": index_hash(train_indices), "train_windows": len(train_indices),
                "evaluated_val_indices_sha256": index_hash(evaluated_val_indices),
                "evaluated_val_windows": len(evaluated_val_indices), "training_device_type": device.type,
                "selected_train_indices": train_indices if settings.few_shot_windows else None,
                "trainable_modules": [name for name in student.FLOW_MODULES if name != "source_model"],
                "dropout": False, "seed_streams": {"batch": settings.seed, "noise": settings.seed+1,
                                                       "schedule": settings.seed+2}, "completed_updates": 0}
    noise_rng = torch.Generator(device=device).manual_seed(settings.seed+1)
    schedule_rng = torch.Generator().manual_seed(settings.seed+2)
    stream_state = None
    if args.resume:
        candidate, payload = load_checkpoint(args.resume, device=device)
        previous = payload.get("scfm_posttrain") or {}
        for key in ("method", "architecture", "dataset_provenance", "base_sha256", "selected_train_indices_sha256", "val_indices_sha256",
                    "evaluated_val_indices_sha256", "training_device_type"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"SCFM resume mismatch: {key}")
        prior_settings = dict(previous["settings"])
        current_settings = settings.as_dict()
        prior_settings.pop("updates")
        current_settings.pop("updates")
        if prior_settings != current_settings:
            raise ValueError("resume settings changed (only total updates may change)")
        student.load_state_dict(candidate.state_dict(), strict=True)
        metadata = previous
        metadata["settings"] = settings.as_dict()
        state = payload["scfm_training_state"]
        fast.load_state_dict(state["fast"], strict=True)
        slow.load_state_dict(state["slow"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        noise_rng.set_state(state["noise_rng"].cpu())
        schedule_rng.set_state(state["schedule_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device=device)
        stream_state = state["batch_stream"]
        del candidate, payload, state
    stream = BatchStream(dataset, train_indices, settings, args.num_workers, stream_state)

    def evaluation():
        result = validate(student, teacher, val_loader, settings, device, envelope)
        print(json.dumps({"validation_update": metadata["completed_updates"], **result}), flush=True)
        return result

    if args.evaluate_only:
        result = evaluation()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        return
    if settings.updates <= metadata["completed_updates"]:
        raise ValueError("total updates must exceed completed updates when resuming")
    if not args.resume:
        metadata["initial_validation"] = evaluation()
    best_score = metadata.get("best_paired_teacher_flow_rmse", float("inf"))
    for update in range(metadata["completed_updates"], settings.updates):
        batch = to_device(stream.next(), device)
        with torch.no_grad(), precision_context(device, settings.precision):
            encoded = teacher.encode_conditions(batch, cache_condition_kv=False)
            prepared = encoded["_prepared_batch"]
            target = (teacher._target_flow_state(prepared, prepared["q"]) if legacy
                      else teacher.target_latent(prepared)).float()
        # Frozen condition tokens shared across models; no cached Flow projections.
        encoded = {key: value for key, value in encoded.items() if key not in {"_prepared_batch", "condition_kv_cache"}}
        noise = torch.randn(target.shape, device=device, generator=noise_rng)
        schedule = sample_schedule(target.shape[0], settings, schedule_rng, device=device)
        with precision_context(device, settings.precision):
            loss, stats = scfm_loss(student, teacher, fast, slow, target, noise, encoded, schedule)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite SCFM loss at update {update+1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_((p for p in student.parameters() if p.requires_grad),
                                                 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Both snapshots are updated after the optimizer; the next target uses
        # the preceding update's snapshots, exactly as in upstream training.
        update_flow_ema(fast, student, settings.fast_ema_decay)
        update_flow_ema(slow, student, settings.slow_ema_decay)
        metadata["completed_updates"] = update+1
        if (update+1) % 10 == 0 or update+1 == settings.updates:
            print(json.dumps({"update": update+1, "grad_norm": float(grad_norm), **stats}), flush=True)
        if (update+1) % settings.validate_every == 0 or update+1 == settings.updates:
            metadata["last_validation"] = evaluation()
            # Preserve the candidate best matching the original fine teacher.
            # This is agreement on held-out conditions, not physical success.
            key = "paired_teacher_flow_rmse" if legacy else "paired_teacher_latent_rmse"
            score = metadata["last_validation"]["variants"]["scfm_coarse"][key]
            improved = score < best_score
            if improved:
                best_score = score
                metadata["best_paired_teacher_flow_rmse"] = score
                metadata["best_update"] = update+1
        else:
            improved = False
        if (update+1) % settings.save_every == 0 or update+1 == settings.updates or improved:
            state = {"fast": fast.state_dict(), "slow": slow.state_dict(), "optimizer": optimizer.state_dict(),
                     "noise_rng": noise_rng.get_state(), "schedule_rng": schedule_rng.get_state(),
                     "torch_rng": torch.get_rng_state(), "batch_stream": stream.state_dict(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            metadata["last_train"] = stats
            save_checkpoint(args.output, student, config, teacher_payload, metadata, state)
            if improved:
                save_checkpoint(args.output.with_name(args.output.stem+"_best"+args.output.suffix),
                                student, config, teacher_payload, metadata, state)
    # Confirm the deployment loader can load this self-contained artifact strictly.
    restored, _ = load_checkpoint(args.output, device="cpu")
    if restored.flow_solver != "euler" or restored.flow_inference_steps != settings.steps:
        raise RuntimeError("exported sampler contract mismatch")
    print(json.dumps({"checkpoint": str(args.output), "updates": settings.updates, "nfe": settings.steps}), flush=True)


if __name__ == "__main__":
    main()
