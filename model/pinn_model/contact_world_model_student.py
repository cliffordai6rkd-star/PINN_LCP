"""Few step flow student used by the independent distillation trainer.

The student keeps the complete CARS-WM condition and flow decoder capacity,
but augments the flow-time conditioning with an embedding of the integration
step ``delta_s``.  Integration is explicit Euler with one decoder invocation
per step; the decoder predicts the interval-average velocity.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
import torch.nn as nn

from model.pinn_model.contact_world_model import ContactWorldModel


class DeltaSEmbedding(nn.Module):
    """Fourier-style embedding for the (possibly non-unit) flow step size."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, delta_s: torch.Tensor) -> torch.Tensor:
        if delta_s.ndim == 1:
            delta_s = delta_s[:, None]
        if delta_s.ndim != 2 or delta_s.shape[-1] != 1:
            raise ValueError(
                f"delta_s must have shape [B] or [B, 1], got {tuple(delta_s.shape)}"
            )
        features = torch.cat(
            (
                delta_s,
                torch.sin(math.pi * delta_s),
                torch.cos(math.pi * delta_s),
                torch.sin(2.0 * math.pi * delta_s),
                torch.cos(2.0 * math.pi * delta_s),
            ),
            dim=-1,
        )
        return self.projection(features)


class ContactWorldModelStudent(ContactWorldModel):
    """Student flow model with interval-step conditioning and Euler updates."""

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.student_steps = int((config.get("distillation") or {}).get(
            "student_steps", self.flow_inference_steps
        ))
        if self.student_steps <= 0:
            raise ValueError("distillation.student_steps must be positive")
        self.flow_delta_embedding = DeltaSEmbedding(self.hidden_dim)
        # A zero initialized residual leaves the copied teacher velocity intact
        # at initialization while allowing the student to learn a step-specific
        # correction.
        self.student_flow_output = nn.Sequential(
            nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.flow_dim)
        )
        nn.init.zeros_(self.student_flow_output[-1].weight)
        nn.init.zeros_(self.student_flow_output[-1].bias)

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        contract["student"] = {"integration": "delta_s_euler", "steps": self.student_steps}
        return contract

    @classmethod
    def from_teacher(cls, teacher: ContactWorldModel, *, student_steps: int = 8):
        """Construct a student and strictly copy the complete current teacher.

        The delta-step embedding and zero-initialized student output head are
        intentionally left at their constructor values.
        """

        config = copy.deepcopy(teacher._config)
        config.setdefault("model", {})["flow_inference_steps"] = int(student_steps)
        config.setdefault("model", {})["flow_solver"] = "euler"
        config.setdefault("distillation", {})["student_steps"] = int(student_steps)
        student = cls(config)
        # Strictly restore the complete current teacher architecture first.
        # Only the student's newly defined delta-step/head modules are new.
        base = ContactWorldModel(config)
        base.load_state_dict(teacher.state_dict(), strict=True)
        for name, module in base.named_children():
            setattr(student, name, module)
        return student

    def flow_velocity_student(self, trajectory_state, flow_time, delta_s, encoded):
        expected = (trajectory_state.shape[0], self.future_horizon, self.flow_dim)
        if trajectory_state.ndim != 3 or tuple(trajectory_state.shape) != expected:
            raise ValueError(
                "trajectory_state must have shape "
                f"[B, {self.future_horizon}, {self.flow_dim}], got {tuple(trajectory_state.shape)}"
            )
        if tuple(flow_time.shape) != (trajectory_state.shape[0], 1):
            raise ValueError("flow_time must have shape [B, 1]")
        if tuple(delta_s.shape) != (trajectory_state.shape[0], 1):
            raise ValueError("delta_s must have shape [B, 1]")
        features = (
            self.flow_input_projection(trajectory_state)
            + self.flow_time_embedding(flow_time)[:, None, :]
            + self.flow_delta_embedding(delta_s)[:, None, :]
            + self.future_pos_embedding(
                torch.arange(trajectory_state.shape[1], device=trajectory_state.device)
            )[None].to(trajectory_state.dtype)
        )
        for block in self.flow_blocks:
            features = block(
                features,
                encoded["state_tokens"],
                encoded["action_tokens"],
                encoded["action_padding_mask"],
            )
        velocity = self.flow_output(features) + self.student_flow_output(features)
        return velocity, features

    def flow_velocity(self, trajectory_state, flow_time, encoded, delta_s=None):
        """Dispatch to the student head when ``delta_s`` is supplied.

        Omitting ``delta_s`` uses the base CFM velocity for teacher-interval
        integration and ordinary flow diagnostics.
        """

        if delta_s is None:
            return super().flow_velocity(trajectory_state, flow_time, encoded)
        return self.flow_velocity_student(trajectory_state, flow_time, delta_s, encoded)

    def integrate_flow(self, source_state, encoded, *, steps=None, solver=None):
        """Integrate with one student decoder call per step (Euler update)."""

        steps = self.student_steps if steps is None else int(steps)
        if steps <= 0:
            raise ValueError("Flow integration steps must be positive")
        if solver is not None and str(solver).lower() not in {"euler", "student"}:
            raise ValueError("student flow integration only supports Euler updates")
        trajectory = source_state
        delta = 1.0 / float(steps)
        delta_tensor = trajectory.new_full((trajectory.shape[0], 1), delta)
        for step in range(steps):
            flow_time = trajectory.new_full((trajectory.shape[0], 1), step * delta)
            velocity, _ = self.flow_velocity_student(
                trajectory, flow_time, delta_tensor, encoded
            )
            trajectory = trajectory + delta * velocity
        return trajectory

    def predict_differentiable(self, batch, *, steps=None, solver=None, source_noise=None):
        encoded = self.encode_conditions(batch)
        reference = batch[self.inputs[0]]
        source = self._gaussian_flow_source(reference, source_noise)
        generated = self.integrate_flow(source, encoded, steps=steps, solver=solver)
        result = {**encoded, "flow_source_state": source, "flow_source_noise": source}
        result.update(self._decoded_output(generated, encoded))
        return result


@torch.no_grad()
def integrate_teacher_interval(
    teacher: ContactWorldModel,
    trajectory_state: torch.Tensor,
    encoded,
    start_time: torch.Tensor,
    delta_s: float | torch.Tensor,
    *,
    teacher_steps: int = 32,
):
    """Integrate a frozen teacher from ``s`` to ``s + delta_s`` with Heun.

    ``teacher_steps`` is the number of uniform steps over the full unit flow;
    the interval therefore uses ``round(delta_s * teacher_steps)`` substeps.
    The returned tensor is detached, making it safe as a local distillation
    target while preserving the student's gradient path elsewhere.
    """

    if teacher_steps <= 0:
        raise ValueError("teacher_steps must be positive")
    if start_time.ndim == 1:
        start_time = start_time[:, None]
    if start_time.ndim != 2 or start_time.shape[-1] != 1:
        raise ValueError("start_time must have shape [B, 1]")
    delta = torch.as_tensor(delta_s, device=trajectory_state.device, dtype=trajectory_state.dtype)
    if delta.ndim == 0:
        delta = delta.expand(trajectory_state.shape[0]).reshape(-1, 1)
    elif delta.ndim == 1:
        delta = delta[:, None] if delta.numel() > 1 else delta.expand(trajectory_state.shape[0])[:, None]
    if tuple(delta.shape) != tuple(start_time.shape):
        raise ValueError("delta_s must be scalar or shape [B, 1]")
    substeps = torch.round(delta * teacher_steps).to(torch.long)
    if torch.any(substeps < 1):
        raise ValueError("delta_s is smaller than one teacher step")
    if torch.any(start_time + delta > 1.0 + 1e-6):
        raise ValueError("teacher interval exceeds flow time 1")
    # Distillation currently uses a common delta per minibatch. Supporting a
    # varying number of substeps would require ragged integration, so enforce
    # a single count and retain a vectorized implementation.
    if torch.unique(substeps).numel() != 1:
        raise ValueError("delta_s must be constant across a batch")
    n = int(substeps[0].item())
    h = delta / float(n)
    state = trajectory_state
    for idx in range(n):
        t = start_time + idx * h
        first, _ = teacher.flow_velocity(state, t, encoded)
        proposal = state + h[:, None, :] * first
        second, _ = teacher.flow_velocity(proposal, t + h, encoded)
        state = state + 0.5 * h[:, None, :] * (first + second)
    return state.detach()
