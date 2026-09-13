"""Independent T32 -> few-step CARS-WM flow distillation entry point."""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Mapping

import torch
import yaml

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
from train.base_trainer import BaseTrainer
from train.contact_world_model_loss import ContactWorldModelLoss
from train.nomalizer import Normalizer

log = logging.getLogger(__name__)


def _resolve_checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.is_dir():
        files = sorted((path / "checkpoints").glob("epoch_*.pt")) or sorted(path.glob("epoch_*.pt"))
        if not files:
            raise FileNotFoundError(f"no epoch_*.pt checkpoint in {path}")
        path = files[-1]
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
    return path


class ContactWorldModelDistillTrainer(BaseTrainer):
    """Distill a frozen EMA T32 teacher into an S8 or S4 student."""

    def __init__(self, config: Mapping):
        raw = copy.deepcopy(dict(config))
        distill = raw.get("distillation") or {}
        self.student_steps = int(distill.get("student_steps", 8))
        self.teacher_steps = int(distill.get("teacher_steps", 32))
        if self.student_steps not in (4, 8):
            raise ValueError("distillation.student_steps must be 4 or 8")
        teacher_path = distill.get("teacher_checkpoint_path")
        if not teacher_path:
            raise ValueError("distillation.teacher_checkpoint_path is required")
        self.teacher_checkpoint_path = _resolve_checkpoint(teacher_path)
        self.student_init_checkpoint_path = distill.get("student_init_checkpoint_path")
        if self.student_init_checkpoint_path:
            self.student_init_checkpoint_path = _resolve_checkpoint(self.student_init_checkpoint_path)
        self.teacher_checkpoint = torch.load(self.teacher_checkpoint_path, map_location="cpu", weights_only=False)
        teacher_config = copy.deepcopy(dict(self.teacher_checkpoint.get("config") or {}))
        if not teacher_config:
            raise ValueError("teacher checkpoint does not contain its config")
        # Data/model contract comes exclusively from the checkpoint.  Training
        # controls and distillation options come from the student config.
        effective = copy.deepcopy(teacher_config)
        effective["train"] = copy.deepcopy(raw.get("train") or {})
        effective["distillation"] = copy.deepcopy(distill)
        effective.setdefault("model", {})["flow_inference_steps"] = self.student_steps
        effective["model"]["flow_solver"] = "euler"
        super().__init__(effective)
        self.teacher_config = teacher_config
        self.teacher = None
        self.loss_calculator = ContactWorldModelLoss(effective)
        self._teacher_normalizer = self.teacher_checkpoint.get("normalizer")

    def build_model(self):
        # Teacher weights are copied after BaseTrainer has constructed the
        # student so optimizer/EMA state are created for student parameters.
        teacher = ContactWorldModel(self.teacher_config)
        state = self.teacher_checkpoint.get("model") or self.teacher_checkpoint.get("model_raw")
        if not isinstance(state, Mapping):
            raise KeyError("teacher checkpoint has no EMA model weights")
        teacher.load_state_dict(state, strict=True)
        teacher.eval().requires_grad_(False)
        self.teacher = teacher.to(self.device)
        student = ContactWorldModelStudent.from_teacher(self.teacher, student_steps=self.student_steps)
        if self.student_init_checkpoint_path:
            payload = torch.load(self.student_init_checkpoint_path, map_location="cpu", weights_only=False)
            state = payload.get("model") or payload.get("model_raw")
            if not isinstance(state, Mapping):
                raise KeyError("student initialization checkpoint has no model weights")
            student.load_state_dict(state, strict=True)
        return student

    def build_dataset(self):
        return ContactWorldModelDataset(self.config, compute_normalizer=False)

    def fit_dataset_normalizer(self, train_dataset):
        # Distillation must run in the teacher's recorded normalization space.
        payload = self._teacher_normalizer
        if not isinstance(payload, Mapping) or not payload.get("stats"):
            raise ValueError("teacher checkpoint has no normalizer stats")
        normalizer = Normalizer(copy.deepcopy(payload["stats"]), eps=float(payload.get("eps", 1e-6)))
        self.dataset.set_normalizer(normalizer)
        self.loss_calculator.set_normalizer(normalizer)

    def _teacher_local_target(self, x, start_s, delta_s, encoded):
        """Integrate frozen teacher from s to s+delta_s on the 1/32 Heun grid."""
        value = x
        substeps = max(1, int(round(float(delta_s) * self.teacher_steps)))
        h = 1.0 / self.teacher_steps
        for index in range(substeps):
            t = x.new_full((x.shape[0], 1), start_s + index * h)
            first, _ = self.teacher.flow_velocity(value, t, encoded)
            proposal = value + h * first
            second, _ = self.teacher.flow_velocity(proposal, t + h, encoded)
            value = value + 0.5 * h * (first + second)
        return value

    def _student_trajectory(self, source, encoded):
        value = source
        states = [value]
        h = 1.0 / self.student_steps
        delta = value.new_full((value.shape[0], 1), h)
        for index in range(self.student_steps):
            t = value.new_full((value.shape[0], 1), index * h)
            velocity, _ = self.model.flow_velocity_student(value, t, delta, encoded)
            value = value + h * velocity
            states.append(value)
        return value, states

    def compute_loss(self, batch):
        # Encode conditions with each frozen/trainable network, while keeping
        # the raw observations and action chunk unchanged.
        encoded_student = self.model.encode_conditions(batch)
        with torch.no_grad():
            encoded_teacher = self.teacher.encode_conditions(batch)
        reference = batch[self.model.inputs[0]]
        noise = self.model._gaussian_flow_source(reference)
        teacher_terminal = self.teacher.integrate_flow(noise, encoded_teacher, steps=self.teacher_steps, solver="heun").detach()
        student_terminal, student_states = self._student_trajectory(noise, encoded_student)

        # Sample a local starting point.  Early training uses teacher states;
        # student states are introduced linearly up to 50% by step 5000.
        use_student_p = 0.5 * min(float(self.global_step) / 5000.0, 1.0)
        choose_student = torch.rand((), device=reference.device) < use_student_p
        index = int(torch.randint(0, self.student_steps, ()).item())
        if choose_student:
            x_local = student_states[index].detach()
            start_s = index / self.student_steps
            encoded_local = encoded_teacher
        else:
            with torch.no_grad():
                teacher_states = [noise]
                value = noise
                h = 1.0 / self.teacher_steps
                for i in range(self.teacher_steps):
                    t = value.new_full((value.shape[0], 1), i * h)
                    first, _ = self.teacher.flow_velocity(value, t, encoded_teacher)
                    proposal = value + h * first
                    second, _ = self.teacher.flow_velocity(proposal, t + h, encoded_teacher)
                    value = value + 0.5 * h * (first + second)
                    teacher_states.append(value)
                x_local = teacher_states[index * (self.teacher_steps // self.student_steps)].detach()
            start_s = index / self.student_steps
            encoded_local = encoded_teacher
        delta = 1.0 / self.student_steps
        delta_tensor = reference.new_full((reference.shape[0], 1), delta)
        t_local = reference.new_full((reference.shape[0], 1), start_s)
        student_velocity, _ = self.model.flow_velocity_student(x_local, t_local, delta_tensor, encoded_student)
        with torch.no_grad():
            target_next = self._teacher_local_target(x_local, start_s, delta, encoded_local)
            target_velocity = (target_next - x_local) / delta

        local_sq = (student_velocity - target_velocity).square()
        terminal_sq = (student_terminal - teacher_terminal).square()
        # Preserve per-stream continuous importance weights used by the base
        # loss, and remain entirely in normalized teacher coordinates.
        weights = {"q": self.loss_calculator.q_weight, "dq": self.loss_calculator.dq_weight,
                   "delta_q": self.loss_calculator.delta_q_weight, "tau": self.loss_calculator.tau_weight}
        local_parts, terminal_parts = [], []
        offset = 0
        for key in self.model.predicted_state_streams:
            sl = slice(offset, offset + self.model.joint_dim); offset += self.model.joint_dim
            w = weights.get(key, 1.0)
            local_parts.append(w * local_sq[..., sl].mean())
            terminal_parts.append(w * terminal_sq[..., sl].mean())
        local_loss = torch.stack(local_parts).sum()
        terminal_loss = torch.stack(terminal_parts).sum()
        loss = local_loss + terminal_loss
        return loss, {"loss_local": local_loss.detach(), "loss_terminal": terminal_loss.detach(), "loss": loss.detach()}


def main():
    parser = argparse.ArgumentParser(description="Distill CARS-WM T32 into S8/S4")
    parser.add_argument("--config", "-c", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    ContactWorldModelDistillTrainer(config).train()


if __name__ == "__main__":
    main()
