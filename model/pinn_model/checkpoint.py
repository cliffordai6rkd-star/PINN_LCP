"""Semantic model dispatch; architecture and weights remain strictly validated."""
from collections.abc import Mapping


def model_from_checkpoint(payload):
    from model.pinn_model.contact_world_model import ContactWorldModel
    from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
    from model.pinn_model.deterministic_world_model import DeterministicRobotStateWorldModel

    config = payload.get('config') or payload.get('cfg')
    if not isinstance(config, Mapping):
        raise ValueError('world model checkpoint has no config mapping')
    contract = payload.get('carswm_contract') or payload.get('deterministic_wm_contract') or {}
    if not isinstance(contract, Mapping):
        raise ValueError('world model contract must be a mapping')
    student = contract.get('student')
    declares_student = bool((config.get('distillation') or {}).get('enabled'))
    if student is not None:
        if not isinstance(student, Mapping) or student.get('type') != 'ContactWorldModelStudent':
            raise ValueError('unknown student checkpoint type')
        if student.get('integration') != 'delta_s_euler':
            raise ValueError('unsupported student integration contract')
        if not isinstance(student.get('steps'), int) or isinstance(student.get('steps'), bool) or student['steps'] <= 0:
            raise ValueError('student contract requires positive integer steps')
        model = ContactWorldModelStudent(config)
    elif declares_student:
        raise ValueError('student config requires formal carswm_contract.student metadata')
    elif payload.get('model_version') == DeterministicRobotStateWorldModel.MODEL_VERSION:
        model = DeterministicRobotStateWorldModel(config)
    else:
        model = ContactWorldModel(config)
    model.validate_checkpoint(payload)
    return model
