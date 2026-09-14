import copy

import pytest
import torch

from test_contact_world_model import config, batch
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
from train.contact_world_model_loss import ContactWorldModelLoss


def setup_case():
    cfg = config(outputs=['q', 'tau'])
    cfg['dataloader'].update(state_history_horizon=6, prediction_horizon=4,
                             action_condition_horizon=8)
    cfg['train'] = {'downsample': True}
    cfg['loss']['free_dynamics_weight'] = 0.1
    values = batch(cfg)
    values['contact'] = torch.zeros(2, 6, 1)
    values['contact_future'].fill_(2)
    values['history_valid_mask'] = torch.ones(2, 6, dtype=torch.bool)
    values['importance_weight'] = torch.tensor([1.0, 3.0])
    return cfg, values


def test_current_tau_alignment_and_shared_raw_gru_inputs():
    cfg, values = setup_case()
    model = ContactWorldModel(cfg).eval()
    values['tau'] = torch.arange(24).reshape(2, 6, 2).float()
    seen, head_input = {}, []
    def capture(key):
        def hook(module, args, output):
            seen.setdefault(key, []).append((args[0], output[0]))
        return hook
    handles = [module.register_forward_hook(capture(key)) for key, module in model.state_encoders.items()]
    handles.append(model.free_dynamics_head.register_forward_pre_hook(
        lambda module, args: head_input.append(args[0])))
    out = model(values, flow_time=0.5)
    for handle in handles:
        handle.remove()
    assert all(len(calls) == 1 for calls in seen.values())
    expected = torch.cat([seen[key][0][1][:, -1] for key in ('q','dq','delta_q')], -1)
    torch.testing.assert_close(head_input[0], expected)
    for key in model.inputs:
        torch.testing.assert_close(seen[key][0][0], values[key][:, 1::2])
    torch.testing.assert_close(out['_prepared_batch']['tau'][:, -1], values['tau'][:, -1])
    loss, count = ContactWorldModelLoss(cfg).free_dynamics_loss(out, values)
    errors = (out['free_tau_pred'] - values['tau'][:, -1]).square().mean(-1)
    torch.testing.assert_close(loss, (errors * values['importance_weight']).sum()/4)
    assert count == 2
    assert out['state_tokens'].shape == (2, 12, 8)


def test_aux_prediction_independent_of_tau_action_and_future():
    cfg, values = setup_case()
    model = ContactWorldModel(cfg).eval()
    first = model(values, flow_time=0.2)['free_tau_pred']
    changed = dict(values)
    for key in ('tau', 'action', 'q_future', 'tau_future', 'contact_future'):
        changed[key] = torch.randn_like(values[key]) * 100
    second = model(changed, flow_time=0.8)['free_tau_pred']
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_aux_loss_gradients_only_reach_motion_grus_and_aux_head():
    cfg, values = setup_case()
    model = ContactWorldModel(cfg)
    out = model(values, flow_time=0.5)
    loss, _ = ContactWorldModelLoss(cfg).free_dynamics_loss(out, values)
    loss.backward()
    for key in ('q','dq','delta_q'):
        assert all(p.grad is not None and torch.any(p.grad != 0) for p in model.state_encoders[key].parameters())
    assert all(p.grad is not None and torch.any(p.grad != 0) for p in model.free_dynamics_head.parameters())
    for name, parameter in model.named_parameters():
        if not name.startswith(('state_encoders.q.', 'state_encoders.dq.', 'state_encoders.delta_q.', 'free_dynamics_head.')):
            assert parameter.grad is None, name


@pytest.mark.parametrize('reason', ['contact', 'padding', 'nan_label', 'invalid_label'])
def test_skipped_rows_cannot_become_free_after_downsampling(reason):
    cfg, values = setup_case()
    if reason == 'padding':
        values['history_valid_mask'][:, 0] = False
    else:
        values['contact'][:, 0] = {'contact': 2, 'nan_label': float('nan'), 'invalid_label': -1}[reason]
    model = ContactWorldModel(cfg)
    out = model(values, flow_time=0.5)
    assert not out['_prepared_batch']['free_dynamics_mask'].any()
    loss, count = ContactWorldModelLoss(cfg).free_dynamics_loss(out, values)
    assert loss == 0 and count == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(model.free_dynamics_head[-1].weight.grad) == 0


def test_weighted_free_aggregation_ignores_nonfree_and_invalid_targets():
    cfg, _ = setup_case()
    prediction = torch.tensor([[1., 3.], [2., 4.], [100., 100.]], requires_grad=True)
    target = torch.tensor([[0., 0.], [0., 0.], [float('nan'), float('nan')]])
    values = {'tau': target[:, None], 'free_dynamics_mask': torch.tensor([True, True, False]),
              'importance_weight': torch.tensor([1., 3., 200.])}
    loss, count = ContactWorldModelLoss(cfg).free_dynamics_loss({'free_tau_pred': prediction}, values)
    assert loss == 8.75 and count == 2
    values['importance_weight'][:] = 0
    zero, _ = ContactWorldModelLoss(cfg).free_dynamics_loss({'free_tau_pred': prediction}, values)
    assert zero == 0 and torch.isfinite(zero)


def test_total_loss_and_metrics_add_only_auxiliary_contribution():
    cfg, values = setup_case()
    out = ContactWorldModel(cfg)(values, flow_time=0.5)
    calculator = ContactWorldModelLoss(cfg)
    total, metrics = calculator(out, values)
    baseline = copy.deepcopy(cfg)
    baseline['loss']['free_dynamics_weight'] = 0
    existing, existing_metrics = ContactWorldModelLoss(baseline)(out, values)
    torch.testing.assert_close(total, existing + metrics['free_dynamics_contribution'])
    torch.testing.assert_close(metrics['free_dynamics_contribution'], metrics['free_dynamics_loss']*0.1)
    assert metrics['free_dynamics_count'] == 2
    torch.testing.assert_close(metrics['flow_loss'], existing_metrics['flow_loss'])
    torch.testing.assert_close(metrics['contact_loss'], existing_metrics['contact_loss'])


def test_predict_sample_skip_aux_head_and_tau_is_never_masked():
    cfg, values = setup_case()
    model = ContactWorldModel(cfg)
    seen = []
    handle = model.state_encoders['tau'].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    model(values, flow_time=0.5)
    handle.remove()
    torch.testing.assert_close(seen[0], values['tau'][:, 1::2])
    assert not hasattr(model, '_mask_tau_history')
    def fail(*args):
        raise AssertionError('auxiliary head called during deployment prediction')
    handle = model.free_dynamics_head.register_forward_pre_hook(fail)
    for output in (model.predict(values, steps=1), model.sample(values, steps=1)):
        assert 'free_tau_pred' not in output
        assert 'q_pred' in output and 'tau_pred' in output
    handle.remove()


def test_strict_checkpoint_roundtrip_and_student_copy(tmp_path):
    cfg, values = setup_case()
    model = ContactWorldModel(cfg).eval()
    payload = {'model_version': model.MODEL_VERSION, 'carswm_contract': model.checkpoint_contract(),
               'model': model.state_dict(), 'config': cfg}
    path = tmp_path/'current.pt'
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=False)
    restored = ContactWorldModel(cfg).eval()
    restored.validate_checkpoint(loaded)
    restored.load_state_dict(loaded['model'], strict=True)
    torch.testing.assert_close(model(values)['free_tau_pred'], restored(values)['free_tau_pred'])
    for change in ({'model_version': 'carswm_v7'}, {'carswm_contract': {**payload['carswm_contract'], 'schema_version': 8}}):
        with pytest.raises(ValueError, match='Incompatible|mismatch'):
            restored.validate_checkpoint({**payload, **change})
    incomplete = dict(payload['model'])
    incomplete.pop('free_dynamics_head.0.weight')
    with pytest.raises(RuntimeError, match='Missing key'):
        restored.load_state_dict(incomplete, strict=True)
    student = ContactWorldModelStudent.from_teacher(model, student_steps=4)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(student.state_dict()[key], value)


@pytest.mark.parametrize('name', ['tau_history_mask_warmup_steps','tau_free_warmup_steps'])
def test_old_masking_options_are_rejected(name):
    cfg, _ = setup_case()
    cfg['train'][name] = 100
    for constructor in (ContactWorldModel, ContactWorldModelLoss):
        with pytest.raises(ValueError, match='Removed tau history masking'):
            constructor(cfg)


def test_training_device_transfer_keeps_auxiliary_supervision(tmp_path):
    from train.trainer.contact_world_model_train import ContactWorldModelTrainer
    cfg, values = setup_case()
    cfg['train'].update(device='cpu', output_dir=str(tmp_path), device_batch_keys=['q'])
    trainer = ContactWorldModelTrainer(cfg)
    assert {'contact', 'history_valid_mask', 'tau', 'importance_weight'} <= trainer.device_batch_keys
    moved = trainer.batch_to_device(values)
    for key in ('contact', 'history_valid_mask', 'tau', 'importance_weight'):
        torch.testing.assert_close(moved[key], values[key])


@pytest.mark.parametrize('name', ['endpoint_loss', 'kinematic_consistency_weight',
                                 'ddq_smoothness_weight', 'torque_contact_weight'])
def test_unconnected_old_loss_options_are_rejected(name):
    cfg, _ = setup_case()
    cfg['loss'][name] = 0
    with pytest.raises(ValueError, match='Removed unused WM loss options'):
        ContactWorldModelLoss(cfg)


def test_misaligned_history_labels_are_rejected():
    cfg, values = setup_case()
    cfg['train']['downsample'] = False
    values['contact'] = torch.zeros(2, 4, 1)
    values['history_valid_mask'] = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match='must align'):
        ContactWorldModel(cfg)(values)


def test_wm_resume_cannot_substitute_ema_for_missing_raw_weights(tmp_path):
    from train.base_trainer import BaseTrainer
    cfg, _ = setup_case()
    model = ContactWorldModel(cfg)
    payload = {'model_version': model.MODEL_VERSION, 'carswm_contract': model.checkpoint_contract(),
               'model': model.state_dict()}
    path = tmp_path/'missing_raw.pt'
    torch.save(payload, path)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = model
    trainer.resume_from = path
    trainer.ema = None
    with pytest.raises(KeyError, match='missing model_raw'):
        trainer._load_resume_checkpoint()
