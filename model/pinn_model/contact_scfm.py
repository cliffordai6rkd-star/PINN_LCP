"""SCFM velocity interface for the existing schema-10 GRU world model.

No parameters are added. Exports load directly in ContactWorldModel.
The shared future position embedding stays frozen to preserve the contact head.
"""
import copy

import torch

from model.pinn_model.contact_world_model import ContactWorldModel


class SCFMContactWorldModel(ContactWorldModel):
    FLOW_MODULES = ("flow_input_projection", "flow_time_embedding", "flow_blocks", "flow_output")

    def __init__(self, config, *, checkpoint_schema=None):
        # Schema 11 changes offline contact semantics, not network parameters.
        # Keep schema 10 artifacts usable rather than globally rewriting them.
        if checkpoint_schema is None:
            checkpoint_schema = 11 if "contact_threshold" in (config.get("contact_gate") or {}) else 10
        if checkpoint_schema not in {10, 11}:
            raise ValueError("supported GRU checkpoint schemas: 10 and 11")
        self.checkpoint_schema = checkpoint_schema
        super().__init__(config)

    def checkpoint_contract(self):
        contract = super().checkpoint_contract()
        if self.checkpoint_schema == 11:
            cfg = self._config.get("contact_gate") or {}
            contract["schema_version"] = 11
            contract["contact"] = {
                "classes": ["free", "alignment", "contact"] if self.contact_state_count == 3 else
                           [f"phase_{i}" for i in range(self.contact_state_count)],
                "label_mode": str(cfg.get("label_mode", "three_phase")),
                "phase_rule": "strict_threshold_temporal_precontact",
                "tau_ext_source": str(cfg.get("tau_ext_source", "tau_measured_minus_tau_free")),
                "norm": str(cfg.get("metric", "tau_ext_l1")),
                "contact_threshold": float(cfg.get("contact_threshold", 10.0)),
                "precontact_duration_s": float(cfg.get("precontact_duration_s", 1.0)),
                "comparison": ">",
            }
        return contract

    def velocity(self, state, time, encoded):
        return self.flow_velocity(state, time.reshape(-1, 1), encoded)[0]


def load_contact_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SCFMContactWorldModel(payload["config"], checkpoint_schema=payload.get("carswm_contract", {}).get("schema_version"))
    model.validate_checkpoint(payload)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def make_contact_student(base, payload, steps):
    config = copy.deepcopy(payload["config"])
    config["model"].update(flow_inference_steps=steps, flow_solver="euler")
    if isinstance((config.get("train") or {}).get("ema"), dict):
        config["train"]["ema"]["enabled"] = False
    student = SCFMContactWorldModel(config, checkpoint_schema=base.checkpoint_schema)
    student.load_state_dict(base.state_dict(), strict=True)
    student.requires_grad_(False)
    for name in student.FLOW_MODULES:
        getattr(student, name).requires_grad_(True)
    return student.eval(), config
