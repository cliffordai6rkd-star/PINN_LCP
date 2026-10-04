"""SCFM velocity interface for the existing schema-10 GRU world model.

No parameters are added. Exports load directly in ContactWorldModel.
The shared future position embedding stays frozen to preserve the contact head.
"""
import copy

import torch

from model.pinn_model.contact_world_model import ContactWorldModel


class SCFMContactWorldModel(ContactWorldModel):
    FLOW_MODULES = ("flow_input_projection", "flow_time_embedding", "flow_blocks", "flow_output")

    def velocity(self, state, time, encoded):
        return self.flow_velocity(state, time.reshape(-1, 1), encoded)[0]


def load_contact_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SCFMContactWorldModel(payload["config"])
    model.validate_checkpoint(payload)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def make_contact_student(base, payload, steps):
    config = copy.deepcopy(payload["config"])
    config["model"].update(flow_inference_steps=steps, flow_solver="euler")
    if isinstance((config.get("train") or {}).get("ema"), dict):
        config["train"]["ema"]["enabled"] = False
    student = SCFMContactWorldModel(config)
    student.load_state_dict(base.state_dict(), strict=True)
    student.requires_grad_(False)
    for name in student.FLOW_MODULES:
        getattr(student, name).requires_grad_(True)
    return student.eval(), config
