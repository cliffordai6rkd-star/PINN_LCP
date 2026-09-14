import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_contact_world_model import config, batch
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import ContactWorldModelStudent


def model_and_batch():
    torch.manual_seed(71)
    cfg = config(outputs=['q', 'tau'])
    return ContactWorldModel(cfg).eval(), batch(cfg)


def test_raw_flow_projection_retains_common_shift_and_scale_and_flow_gradients():
    model, values = model_and_batch()
    projection = model.flow_input_projection
    assert isinstance(projection, nn.Linear)
    x = torch.randn(2, model.future_horizon, model.flow_dim)
    original = projection(x)
    assert original.shape == (2, model.future_horizon, model.hidden_dim)
    assert not torch.allclose(original, projection(x+3))
    assert not torch.allclose(original, projection(x*2))
    torch.testing.assert_close(projection(x+3)-original,
                               F.linear(torch.full_like(x, 3), projection.weight))
    out = model(values, flow_time=0.4)
    (out['flow_velocity_pred']-out['flow_velocity_target']).square().mean().backward()
    assert torch.any(projection.weight.grad != 0)
    assert isinstance(model.flow_output[0], nn.LayerNorm)
    student = ContactWorldModelStudent.from_teacher(model, student_steps=4)
    assert isinstance(student.flow_input_projection, nn.Linear)
    torch.testing.assert_close(student.flow_input_projection.weight, projection.weight)


def test_temporal_contact_reads_other_times_but_not_other_batch_trajectories():
    model, values = model_and_batch()
    encoded = model.encode_conditions(values)
    trajectory = torch.randn(2, model.future_horizon, model.flow_dim)
    first = model.contact_logits(trajectory, encoded)
    changed_time = trajectory.clone()
    changed_time[0, 1:] += 7
    second = model.contact_logits(changed_time, encoded)
    assert not torch.allclose(first[0, 0], second[0, 0])
    changed_batch = trajectory.clone()
    changed_batch[1] *= -13
    third = model.contact_logits(changed_batch, encoded)
    torch.testing.assert_close(first[0], third[0], rtol=0, atol=0)


def test_contact_structure_and_shared_future_positions():
    model, values = model_and_batch()
    assert isinstance(model.contact_state_projection[0], nn.Linear)
    assert len(model.contact_state_projection) == 2
    assert isinstance(model.contact_fusion, nn.Linear)
    assert model.contact_fusion.in_features == 3*model.hidden_dim
    temporal = model.contact_temporal
    assert isinstance(temporal, nn.TransformerEncoderLayer)
    assert temporal.norm_first and temporal.self_attn.batch_first
    assert temporal.self_attn.num_heads == model.flow_attention_heads
    assert temporal.linear1.out_features == 2*model.hidden_dim
    assert isinstance(model.contact_head[0], nn.LayerNorm)
    encoded = model.encode_conditions(values)
    trajectory = torch.randn(2, model.future_horizon, model.flow_dim)
    seen = {}
    def capture(module, args):
        seen['input'] = args[0]
    handle = temporal.register_forward_pre_hook(capture)
    model.contact_logits(trajectory, encoded)
    handle.remove()
    state = model.contact_state_projection(trajectory)
    action = model._aligned_action_features(encoded)
    summary = model.contact_condition_norm(encoded['condition_summary'])[:, None].expand(-1, model.future_horizon, -1)
    expected = model.contact_fusion(torch.cat((state, action, summary), -1))
    expected = expected + model.future_pos_embedding(torch.arange(model.future_horizon))[None]
    torch.testing.assert_close(seen['input'], expected)


def test_contact_ce_backpropagates_through_endpoint_into_flow():
    model, values = model_and_batch()
    model.train()
    out = model(values, flow_time=0.3)
    out['flow_state_pred'].retain_grad()
    loss = F.cross_entropy(out['contact_logits'].transpose(1, 2), values['contact_future'][..., 0].long())
    loss.backward()
    assert torch.any(out['flow_state_pred'].grad != 0)
    for module in (model.contact_temporal, model.contact_fusion, model.flow_input_projection,
                   model.flow_blocks, model.flow_output):
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())


@pytest.mark.parametrize('student', [False, True])
def test_contact_runs_once_per_complete_trajectory_outside_integrator(student):
    model, values = model_and_batch()
    if student:
        model = ContactWorldModelStudent.from_teacher(model, student_steps=4).eval()
    calls = []
    handle = model.contact_temporal.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    encoded = model.encode_conditions(values)
    source = torch.randn(2, model.future_horizon, model.flow_dim)
    model.integrate_flow(source, encoded, steps=3)
    assert calls == []
    forward = model(values, flow_time=0.4, source_noise=source)
    assert len(calls) == 1
    prediction = model.predict(values, steps=3, source_noise=source)
    assert len(calls) == 2
    samples = model.sample(values, num_samples=3, steps=2)
    assert len(calls) == 5
    handle.remove()
    for out in (forward, prediction):
        assert out['contact_logits'].shape == (2, model.future_horizon, 3)
        assert out['contact_probability'].shape == (2, model.future_horizon, 3)
        assert out['contact_state_pred'].shape == (2, model.future_horizon, 1)
        assert out['q_pred'].shape == (2, model.future_horizon, model.joint_dim)
    assert samples['contact_logits'].shape == (2, 3, model.future_horizon, 3)


def test_current_contact_checkpoint_strict_loading_and_old_rejection(tmp_path):
    model, _ = model_and_batch()
    payload = {'model_version': model.MODEL_VERSION, 'carswm_contract': model.checkpoint_contract(),
               'model': model.state_dict()}
    path = tmp_path/'current.pt'
    torch.save(payload, path)
    saved = torch.load(path, weights_only=False)
    restored = ContactWorldModel(model._config)
    restored.validate_checkpoint(saved)
    restored.load_state_dict(saved['model'], strict=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    for change in ({'model_version': 'carswm_v8'},
                   {'carswm_contract': {**saved['carswm_contract'], 'schema_version': 9}}):
        with pytest.raises(ValueError, match='Incompatible|mismatch'):
            restored.validate_checkpoint({**saved, **change})
    missing = dict(saved['model'])
    missing.pop('contact_temporal.self_attn.in_proj_weight')
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(missing, strict=True)
