"""Independent T64 Heun -> S8/S4/S2 CaRS-WM distillation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import math
import shutil
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import (
    ContactWorldModelStudent, integrate_teacher_interval,
)
from train.base_trainer import BaseTrainer
from train.carswm_metrics import distribution_metrics
from train.contact_world_model_loss import ContactWorldModelLoss
from train.nomalizer import Normalizer

log = logging.getLogger(__name__)


def _resolve_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.is_dir():
        latest = path / "checkpoints" / "latest.pt"
        if not latest.exists():
            latest = path / "latest.pt"
        if latest.exists():
            path = latest
        else:
            files = sorted((path / "checkpoints").glob("*.pt")) or sorted(path.glob("*.pt"))
            if not files:
                raise FileNotFoundError(f"no checkpoint in {path}")
            path = files[-1]
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path.resolve()


def _merge(destination, overlay):
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), Mapping):
            _merge(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)


def _equal_nested(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and set(left) == set(right)
                and all(_equal_nested(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(_equal_nested(a, b) for a, b in zip(left, right)))
    return left == right


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_contract(model):
    contract = copy.deepcopy(model.checkpoint_contract())
    contract.pop("student", None)
    # Integration is the only allowed contract difference across stages.
    contract["flow"].pop("solver")
    contract["flow"].pop("steps")
    return contract


def _stage_config_contract(config):
    model = copy.deepcopy(config.get("model") or {})
    model.pop("flow_inference_steps", None)
    model.pop("flow_solver", None)
    return {
        "model": model,
        "dataloader": config.get("dataloader"),
        "action_contract": config.get("action_contract"),
        "contact_gate": config.get("contact_gate"),
        "train_data": config.get("train_data"),
        "downsample": (config.get("train") or {}).get("downsample", False),
    }


class ContactWorldModelDistillTrainer(BaseTrainer):
    """Train the full student Euler chain against one frozen T64 teacher."""

    def __init__(self, config: Mapping):
        raw = copy.deepcopy(dict(config))
        distill = raw.get("distillation") or {}
        self.student_steps = int(distill.get("student_steps", 4))
        self.teacher_steps = int(distill.get("teacher_steps", 64))
        if self.student_steps not in (2, 4, 8) or self.teacher_steps != 64:
            raise ValueError("distillation requires student_steps=2/4/8 and teacher_steps=64")
        if self.teacher_steps % self.student_steps:
            raise ValueError("teacher_steps must be divisible by student_steps")
        teacher_path = distill.get("teacher_checkpoint_path")
        if not teacher_path:
            raise ValueError("distillation.teacher_checkpoint_path is required")
        self.teacher_checkpoint_path = _resolve_checkpoint(teacher_path)
        init_path = distill.get("student_init_checkpoint_path")
        self.student_init_checkpoint_path = _resolve_checkpoint(init_path) if init_path else None
        if self.student_steps == 2 and self.student_init_checkpoint_path is None:
            raise ValueError("S2 requires a trained S4 student_init_checkpoint_path")
        self.teacher_checkpoint = torch.load(self.teacher_checkpoint_path, map_location="cpu", weights_only=False)
        self.teacher_config = copy.deepcopy(self.teacher_checkpoint.get("config") or {})
        if not self.teacher_config:
            raise ValueError("teacher checkpoint has no config")
        if (self.teacher_config.get("distillation") or {}).get("enabled"):
            raise ValueError("teacher_checkpoint_path must be the original teacher, not a student")
        if not (self.teacher_checkpoint.get("ema") or {}).get("enabled", False):
            raise ValueError("teacher checkpoint must contain EMA deployment weights")
        if not isinstance(self.teacher_checkpoint.get("model"), Mapping):
            raise ValueError("teacher checkpoint has no EMA model state")
        # The original checkpoint owns every data/model/time contract field.
        unexpected = set(raw) - {"train", "distillation"}
        if unexpected:
            raise ValueError(f"distillation YAML may only contain train/distillation: {sorted(unexpected)}")
        effective = copy.deepcopy(self.teacher_config)
        _merge(effective.setdefault("train", {}), raw.get("train") or {})
        for resume_key in ("resume_from", "resume_checkpoint", "resume"):
            if resume_key not in (raw.get("train") or {}):
                effective["train"].pop(resume_key, None)
        effective["distillation"] = copy.deepcopy(distill)
        effective["distillation"]["teacher_checkpoint_path"] = str(self.teacher_checkpoint_path)
        effective["distillation"]["teacher_checkpoint_sha256"] = _sha256(self.teacher_checkpoint_path)
        if self.student_init_checkpoint_path:
            effective["distillation"]["student_init_checkpoint_path"] = str(self.student_init_checkpoint_path)
        effective.setdefault("model", {})["flow_inference_steps"] = self.student_steps
        effective["model"]["flow_solver"] = "euler"
        # Do not inherit teacher validation/visualization settings that run a
        # different model or use a different checkpoint selection criterion.
        effective["train"]["monitor_key"] = "val_terminal_mse"
        effective["train"]["scheduler_monitor_key"] = "val_terminal_mse"
        effective["train"]["early_stopping_monitor_key"] = "val_terminal_mse"
        effective["train"]["save_latest_checkpoint"] = True
        effective["train"]["device_batch_keys"] = None
        effective["train"]["scheduler"] = copy.deepcopy((raw.get("train") or {}).get("scheduler"))
        effective["train"]["early_stopping"] = copy.deepcopy((raw.get("train") or {}).get("early_stopping", {"enabled": False}))
        effective["train"]["train_eval"] = copy.deepcopy((raw.get("train") or {}).get("train_eval", {"enabled": False}))
        effective["train"]["amp"] = copy.deepcopy((raw.get("train") or {}).get("amp", {"enabled": False}))
        effective["train"]["allow_tf32"] = False
        effective["train"].setdefault("ema", {})["enabled"] = True
        effective["train"]["ema"]["use_for_validation"] = True
        super().__init__(effective)
        self.teacher = None
        self.loss_calculator = ContactWorldModelLoss(effective)
        self._teacher_normalizer = self.teacher_checkpoint.get("normalizer")
        self._validate_options()

    def _validate_options(self):
        d = self.config["distillation"]
        if self.max_train_steps is None:
            raise ValueError("train.max_optimizer_steps is required for distillation")
        if self.scheduler_config and self.scheduler_config.get("name") == "reduce_on_plateau":
            raise ValueError("distillation scheduler must advance by optimizer steps")
        self.weights = {}
        for name, default in (("local_weight", 1.0), ("terminal_weight", 1.0),
                              ("q_d1_weight", 0.0), ("q_d2_weight", 0.0),
                              ("contact_distill_weight", 0.0), ("contact_ce_weight", 0.0)):
            value = float(d.get(name, default))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"distillation.{name} must be finite and non-negative")
            self.weights[name] = value
        if not any(self.weights.values()):
            raise ValueError("at least one distillation loss weight must be positive")
        if (self.weights["q_d1_weight"] or self.weights["q_d2_weight"]) and "q" not in self.loss_calculator.predicted_state_streams:
            raise ValueError("q difference losses require q in model.outputs")
        self.student_start_probability_max = float(d.get("student_start_probability_max", 0.5))
        self.student_start_probability_warmup_steps = int(d.get("student_start_probability_warmup_steps", 5000))
        if not 0 <= self.student_start_probability_max <= 1 or self.student_start_probability_warmup_steps < 0:
            raise ValueError("invalid student-start probability schedule")
        self.validation_seed = int(d.get("validation_seed", 2026))
        self.validation_num_samples = int(d.get("validation_num_samples", 4))
        self.validation_max_batches = int(d.get("validation_max_batches", 0))
        self.latency_batches = int(d.get("latency_batches", 1))
        if self.validation_num_samples < 2 or self.validation_max_batches < 0 or self.latency_batches < 0:
            raise ValueError("invalid validation sample/batch count")
        if self.val_ratio <= 0 and self.val_episode_indices is None and self.split_mode != "purged_kfold":
            raise ValueError("distillation requires an independent validation split")
        if self.split_mode == "sample":
            raise ValueError("sample-level split leaks overlapping trajectory windows")

    def build_model(self):
        teacher = ContactWorldModel(self.teacher_config)
        teacher.validate_checkpoint(self.teacher_checkpoint)
        teacher.load_state_dict(self.teacher_checkpoint["model"], strict=True)
        teacher.float().eval().requires_grad_(False)
        student = ContactWorldModelStudent.from_teacher(teacher, student_steps=self.student_steps)
        self.teacher = teacher.to(self.device)
        student._config = self.config
        if self.student_init_checkpoint_path:
            payload = torch.load(self.student_init_checkpoint_path, map_location="cpu", weights_only=False)
            previous_config = payload.get("config") or {}
            if not (previous_config.get("distillation") or {}).get("enabled"):
                raise ValueError("student initialization checkpoint is not a student")
            previous_steps = int((previous_config.get("distillation") or {})["student_steps"])
            if (previous_config["distillation"].get("teacher_checkpoint_sha256")
                    != self.config["distillation"]["teacher_checkpoint_sha256"]):
                raise ValueError("student initialization used a different original teacher")
            allowed = {8: (8,), 4: (8, 4), 2: (4,)}[self.student_steps]
            if previous_steps not in allowed:
                raise ValueError(f"S{self.student_steps} cannot initialize from S{previous_steps}")
            if previous_steps != self.student_steps and int(payload.get("global_step") or 0) <= 0:
                raise ValueError("cross-stage initialization requires a trained student checkpoint")
            previous = ContactWorldModelStudent(previous_config)
            previous.validate_checkpoint(payload)
            if _data_contract(previous) != _data_contract(student):
                raise ValueError("student initialization data/model/time contract differs from teacher")
            if not _equal_nested(_stage_config_contract(previous_config), _stage_config_contract(self.config)):
                raise ValueError("student initialization architecture or data config differs from teacher")
            if not _equal_nested(payload.get("normalizer"), self._teacher_normalizer):
                raise ValueError("student initialization normalizer differs from teacher")
            previous.load_state_dict(payload["model"], strict=True)
            student.load_state_dict(previous.state_dict(), strict=True)
        if not all(parameter.requires_grad for parameter in student.parameters()):
            raise RuntimeError("student contains frozen parameters")
        return student

    def build_dataset(self):
        return ContactWorldModelDataset(self.config, compute_normalizer=False)

    def fit_dataset_normalizer(self, train_dataset):
        payload = self._teacher_normalizer
        if not isinstance(payload, Mapping) or not payload.get("stats"):
            raise ValueError("teacher checkpoint has no normalizer stats")
        if payload.get("normalize_mode") != self.config["dataloader"].get("normalize_mode"):
            raise ValueError("teacher normalizer mode differs from data config")
        normalizer = Normalizer(copy.deepcopy(payload["stats"]), eps=float(payload.get("eps", 1e-6)))
        self.dataset.set_normalizer(normalizer)
        self.loss_calculator.set_normalizer(normalizer)

    @staticmethod
    def _sample_indices(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            return list(dataset.indices)
        return list(range(len(dataset)))

    def build_train_sampler(self, train_dataset):
        # Preserve the teacher trainer's phase sampling and inverse sampling
        # importance weights without importing any recursive OPD training.
        from train.trainer.contact_world_model_train import ContactWorldModelTrainer
        return ContactWorldModelTrainer.build_train_sampler(self, train_dataset)

    def _stream_loss(self, prediction, target, batch):
        offset = 0
        per_sample = prediction.new_zeros(prediction.shape[0])
        stream_weights = {"q": self.loss_calculator.q_weight, "dq": self.loss_calculator.dq_weight,
                          "delta_q": self.loss_calculator.delta_q_weight, "tau": self.loss_calculator.tau_weight}
        for key in self.model.predicted_state_streams:
            width = self.model.joint_dim
            part = (prediction[..., offset:offset + width] - target[..., offset:offset + width]).square()
            per_sample = per_sample + stream_weights[key] * part.flatten(1).mean(dim=1)
            offset += width
        return self.loss_calculator._weighted_mean(per_sample, batch.get("importance_weight"))

    def _student_trajectory(self, source, encoded):
        return self.model.integrate_flow(source, encoded, steps=self.student_steps, return_states=True)

    @torch.no_grad()
    def _teacher_trajectory(self, source, encoded):
        states = [source]
        value = source
        h = 1.0 / self.teacher_steps
        for index in range(self.teacher_steps):
            t = source.new_full((source.shape[0], 1), index * h)
            first, _ = self.teacher.flow_velocity(value, t, encoded)
            second, _ = self.teacher.flow_velocity(value + h * first, t + h, encoded)
            value = value + 0.5 * h * (first + second)
            states.append(value)
        return states

    def _teacher_fp32(self):
        device_type = torch.device(self.device).type
        return torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "npu"} else nullcontext()

    def _student_deployment_train_mode(self):
        # cuDNN needs GRU.training=True to retain its backward reserve, while
        # deployment has no GRU or attention/dropout stochasticity.  Keep all
        # other modules in eval and zero only GRU's internal inter-layer dropout.
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.GRU):
                module.train()
                module.dropout = 0.0

    def compute_loss(self, batch):
        # Distillation uses deployment dropout behavior while autograd remains
        # enabled for the complete student integration chain.
        self._student_deployment_train_mode()
        prepared = self.model.prepare_batch(batch)
        encoded_student = self.model.encode_conditions(prepared)
        reference = prepared[self.model.inputs[0]]
        noise = self.model._gaussian_flow_source(reference).float()
        with torch.no_grad(), self._teacher_fp32():
            encoded_teacher = self.teacher.encode_conditions(prepared)
            teacher_states = self._teacher_trajectory(noise, encoded_teacher)
        student_states = self._student_trajectory(noise, encoded_student)
        target = teacher_states[-1]
        generated = student_states[-1]
        terminal = self._stream_loss(generated, target, prepared)
        zero = terminal.new_zeros(())
        local = zero
        probability = 1.0 if self.student_start_probability_warmup_steps == 0 else min(
            self.global_step / self.student_start_probability_warmup_steps, 1.0
        )
        probability *= self.student_start_probability_max
        if self.weights["local_weight"]:
            index = int(torch.randint(self.student_steps, ()).item())
            h = 1.0 / self.student_steps
            from_student = bool(torch.rand(()) < probability)
            x = (student_states[index] if from_student else teacher_states[index * (self.teacher_steps // self.student_steps)]).detach()
            t = x.new_full((x.shape[0], 1), index * h)
            delta = x.new_full((x.shape[0], 1), h)
            predicted_velocity, _ = self.model.flow_velocity_student(x, t, delta, encoded_student)
            if from_student:
                with self._teacher_fp32():
                    next_state = integrate_teacher_interval(
                        self.teacher, x.float(), encoded_teacher, t.float(), h,
                        teacher_steps=self.teacher_steps,
                    )
            else:
                next_state = teacher_states[(index + 1) * (self.teacher_steps // self.student_steps)]
            local = self._stream_loss(predicted_velocity, ((next_state - x) / h).detach(), prepared)
        q_d1 = q_d2 = zero
        if "q" in self.model.predicted_state_streams:
            index = self.model.predicted_state_streams.index("q") * self.model.joint_dim
            q_student = generated[..., index:index + self.model.joint_dim]
            q_teacher = target[..., index:index + self.model.joint_dim]
            # Adjacent differences along physical future frames at the model's
            # state rate, in the original teacher-normalized q coordinates.
            if self.weights["q_d1_weight"] and q_student.shape[1] > 1:
                q_d1 = self.loss_calculator._weighted_mean(
                    (torch.diff(q_student, dim=1) - torch.diff(q_teacher, dim=1)).square().flatten(1).mean(1),
                    prepared.get("importance_weight"),
                )
            if self.weights["q_d2_weight"] and q_student.shape[1] > 2:
                q_d2 = self.loss_calculator._weighted_mean(
                    (torch.diff(q_student, n=2, dim=1) - torch.diff(q_teacher, n=2, dim=1)).square().flatten(1).mean(1),
                    prepared.get("importance_weight"),
                )
        contact_kl = contact_ce = zero
        if self.weights["contact_distill_weight"] or self.weights["contact_ce_weight"]:
            logits = self.model.contact_logits(generated, encoded_student)
            if self.weights["contact_distill_weight"]:
                with torch.no_grad(), self._teacher_fp32():
                    teacher_logits = self.teacher.contact_logits(target, encoded_teacher)
                per_sample = F.kl_div(F.log_softmax(logits.float(), dim=-1),
                                      F.softmax(teacher_logits.float(), dim=-1), reduction="none").sum(-1).mean(-1)
                contact_kl = self.loss_calculator._weighted_mean(per_sample, prepared.get("importance_weight"))
            if self.weights["contact_ce_weight"]:
                if not (self.config.get("contact_gate") or {}).get("enabled", False):
                    raise ValueError("contact_ce_weight requires enabled contact labels")
                labels = prepared["contact_future"].squeeze(-1).round().long()
                per_sample = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none").mean(1)
                contact_ce = self.loss_calculator._weighted_mean(per_sample, prepared.get("importance_weight"))
        loss = (self.weights["local_weight"] * local + self.weights["terminal_weight"] * terminal
                + self.weights["q_d1_weight"] * q_d1 + self.weights["q_d2_weight"] * q_d2
                + self.weights["contact_distill_weight"] * contact_kl
                + self.weights["contact_ce_weight"] * contact_ce)
        return loss, {"loss_dict": {"local": local.detach(), "terminal": terminal.detach(),
                                   "q_d1": q_d1.detach(), "q_d2": q_d2.detach(),
                                   "contact_kl": contact_kl.detach(), "contact_ce": contact_ce.detach(),
                                   "student_start_probability": probability, "total": loss.detach()}}

    def _sync_device(self):
        kind = torch.device(self.device).type
        if kind == "cuda":
            torch.cuda.synchronize(self.device)
        elif kind == "npu":
            torch.npu.synchronize(self.device)

    @torch.no_grad()
    def _latency_ms(self, model, prepared, noise, *, steps, solver):
        model.predict(prepared, source_noise=noise, steps=steps, solver=solver)
        self._sync_device()
        start = time.perf_counter()
        for _ in range(3):
            model.predict(prepared, source_noise=noise, steps=steps, solver=solver)
        self._sync_device()
        return (time.perf_counter() - start) * 1000.0 / 3.0

    @torch.no_grad()
    def validate_one_epoch(self, epoch, *, force=False):
        if self.step_based_training and not force and getattr(self, "_validated_step", None) != self.global_step:
            self.last_val_epoch_metrics = {}
            return None
        if getattr(self, "_validated_step", None) == self.global_step:
            return self.last_val_epoch_metrics["terminal_mse"]
        if self.val_loader is None:
            raise ValueError("distillation validation loader is absent")
        model = self.ema.model if self.ema is not None and self.ema_use_for_validation else self.model
        model.eval()
        self.teacher.eval()
        sums, counts = {}, {}

        def add(name, value, phase=None):
            value = value.float().reshape(-1)
            if phase is not None:
                value = value[phase]
            if value.numel():
                sums[name] = sums.get(name, 0.0) + float(value.sum().item())
                counts[name] = counts.get(name, 0) + value.numel()

        for batch_index, raw in enumerate(self.val_loader):
            if self.validation_max_batches and batch_index >= self.validation_max_batches:
                break
            batch = self.batch_to_device(raw)
            prepared = model.prepare_batch(batch)
            b = prepared[model.inputs[0]].shape[0]
            generator = torch.Generator(device="cpu").manual_seed(self.validation_seed + batch_index * 1009)
            noises = torch.randn((self.validation_num_samples, b, model.future_horizon, model.flow_dim), generator=generator)
            noises = noises.to(device=self.device, dtype=prepared[model.inputs[0]].dtype)
            outputs = {"student": [], "teacher": [], "short_teacher": []}
            encoded_student = model.encode_conditions(prepared)
            with self._teacher_fp32():
                encoded_teacher = self.teacher.encode_conditions(prepared)
            for noise in noises:
                with self._teacher_fp32():
                    outputs["teacher"].append(self.teacher.integrate_flow(noise.float(), encoded_teacher, steps=64, solver="heun").float())
                    outputs["short_teacher"].append(self.teacher.integrate_flow(noise.float(), encoded_teacher, steps=self.student_steps, solver="euler").float())
                outputs["student"].append(model.integrate_flow(noise, encoded_student, steps=self.student_steps).float())
            outputs = {key: torch.stack(value, dim=1) for key, value in outputs.items()}
            targets = torch.cat([prepared[f"{key}_future"].float() for key in model.predicted_state_streams], dim=-1)
            labels_enabled = (self.config.get("contact_gate") or {}).get("enabled", False)
            phase_values = prepared.get("future_phase") if labels_enabled else None
            if phase_values is None and labels_enabled and "contact_future" in prepared:
                phase_values = prepared["contact_future"].amax(dim=1)
            phases = phase_values.reshape(-1).long() if phase_values is not None else None
            frame_phases = prepared["contact_future"].squeeze(-1).round().long() if labels_enabled and "contact_future" in prepared else None
            phase_names = (("free", "establishing", "contact") if model.contact_state_count == 3
                           else tuple(f"phase_{index}" for index in range(model.contact_state_count)))
            for name, samples in outputs.items():
                add(f"{name}_label_mse", (samples - targets[:, None]).square().mean(dim=(1, 2, 3)))
                if frame_phases is not None:
                    frame_error = (samples - targets[:, None]).square().mean(dim=(1, 3))
                    for phase_index, phase_name in enumerate(phase_names):
                        mask = frame_phases == phase_index
                        add(f"{name}_{phase_name}_frame_label_mse", frame_error[mask])
                        if name != "teacher":
                            paired_frame = (samples - outputs["teacher"]).square().mean(dim=(1, 3))
                            add(f"{name}_{phase_name}_frame_teacher_mse", paired_frame[mask])
                streams = {}
                truths = {}
                for stream_index, key in enumerate(model.predicted_state_streams):
                    sl = slice(stream_index * model.joint_dim, (stream_index + 1) * model.joint_dim)
                    streams[key] = samples[..., sl]
                    truths[key] = targets[..., sl]
                    add(f"{name}_{key}_label_mse", (streams[key] - truths[key][:, None]).square().mean(dim=(1, 2, 3)))
                    if name != "teacher":
                        teacher_stream = outputs["teacher"][..., sl]
                        add(f"{name}_{key}_teacher_mse", (streams[key] - teacher_stream).square().mean(dim=(1, 2, 3)))
                    if key == "q":
                        for order in (1, 2):
                            if streams[key].shape[2] <= order:
                                continue
                            diff = torch.diff(streams[key], n=order, dim=2)
                            truth_diff = torch.diff(truths[key], n=order, dim=1)
                            add(f"{name}_q_d{order}_label_mse", (diff - truth_diff[:, None]).square().mean(dim=(1, 2, 3)))
                            if name != "teacher":
                                teacher_diff = torch.diff(outputs["teacher"][..., sl], n=order, dim=2)
                                add(f"{name}_q_d{order}_teacher_mse", (diff - teacher_diff).square().mean(dim=(1, 2, 3)))
                distribution = distribution_metrics(streams, truths)
                for metric in ("energy_score", "sample_spread", "min_ade", "coverage_90"):
                    add(f"{name}_{metric}", distribution[metric])
                    if phases is not None:
                        for phase_index, phase_name in enumerate(phase_names):
                            add(f"{name}_{phase_name}_{metric}", distribution[metric], phases == phase_index)
                if name != "teacher":
                    add(f"{name}_terminal_mse", (samples - outputs["teacher"]).square().mean(dim=(1, 2, 3)))
                    if phases is not None:
                        paired = (samples - outputs["teacher"]).square().mean(dim=(1, 2, 3))
                        for phase_index, phase_name in enumerate(phase_names):
                            add(f"{name}_{phase_name}_terminal_mse", paired, phases == phase_index)
            if batch_index < self.latency_batches:
                single = {key: value[:1] if torch.is_tensor(value) and value.ndim else value for key, value in prepared.items()}
                noise = noises[0, :1]
                for name, measured, steps, solver in (("student", model, self.student_steps, "euler"),
                                                      ("short_teacher", self.teacher, self.student_steps, "euler"),
                                                      ("teacher", self.teacher, 64, "heun")):
                    with self._teacher_fp32():
                        add(f"{name}_latency_ms", torch.tensor([self._latency_ms(measured, single, noise, steps=steps, solver=solver)]))
        metrics = {key: value / counts[key] for key, value in sums.items()}
        if "student_terminal_mse" not in metrics:
            raise ValueError("validation produced no samples")
        metrics["terminal_mse"] = metrics["student_terminal_mse"]
        self.last_val_epoch_metrics = metrics
        self._validated_step = self.global_step
        return metrics["terminal_mse"]

    def save_step_checkpoint(self, epoch, metrics):
        # Select EMA checkpoints by complete K-step, same-noise validation.
        if self._last_step_checkpoint == self.global_step:
            return
        val_loss = self.validate_one_epoch(epoch, force=True)
        metrics = {**dict(metrics), "val_loss": val_loss,
                   **{f"val_{key}": value for key, value in self.last_val_epoch_metrics.items()}}
        super().save_step_checkpoint(epoch, metrics)
        best_dir = self.ckpt_dir / "best_terminal"
        best_dir.mkdir(parents=True, exist_ok=True)
        selected = best_dir / f"step_{self.global_step:08d}_mse_{val_loss:.8f}.pt"
        shutil.copy2(self.ckpt_dir / "latest.pt", selected)
        ranked = sorted(best_dir.glob("step_*_mse_*.pt"), key=lambda path: float(path.stem.rsplit("_", 1)[-1]))
        for extra in ranked[self.top_k:]:
            extra.unlink()
        log.info("complete-sampling validation step=%d terminal_mse=%.6f", self.global_step, val_loss)


def main():
    parser = argparse.ArgumentParser(description="Distill original T64 CaRS-WM into S8/S4/S2")
    parser.add_argument("--config", "-c", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    ContactWorldModelDistillTrainer(config).train()


if __name__ == "__main__":
    main()
