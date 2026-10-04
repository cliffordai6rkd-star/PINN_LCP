"""Generated torque labels must agree across WM families, caches and losses."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_process import contact_world_model_dataset as dataset_module
from data_process import wm_tau_labels as labeling
from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from model.pinn_model.contact_gate import ContactGateConfig, contact_phase_labels_from_signal, batched_contact_phase_mask
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel
from train.contact_world_model_loss import ContactWorldModelLoss
from train.latent_contact_world_model_loss import LatentContactWorldModelLoss
from train.trainer.contact_world_model_train import ContactWorldModelTrainer
from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_strict_threshold_actual_one_second_and_contact_priority():
    t = torch.arange(501, dtype=torch.float64)*.01
    signal = torch.zeros(501)
    signal[150] = 10.  # Equality is not contact.
    signal[200:211] = 10.01
    signal[250:261] = 12.
    cfg = ContactGateConfig(contact_threshold=10., precontact_duration_s=1.)
    labels = contact_phase_labels_from_signal(signal, [(0, len(t))], cfg, timestamps_s=t)[:, 0]
    assert (labels[:100] == 0).all()
    assert (labels[100:200] == 1).all()
    assert (labels[200:211] == 2).all()  # Second prefix cannot overwrite contact.
    assert labels[150] == 1 and (labels[261:] == 0).all()
    batched = batched_contact_phase_mask(signal[None], contact_threshold=10., precontact_duration_s=1.)
    torch.testing.assert_close(batched[0], labels)
    irregular = torch.tensor([0., .009, .02, .029, .04, .051], dtype=torch.float64)
    labels = contact_phase_labels_from_signal(torch.tensor([0., 0., 0., 0., 0., 11.]), [(0, 6)],
        ContactGateConfig(precontact_duration_s=.03), timestamps_s=irregular)[:, 0]
    assert labels.tolist() == [0., 0., 0., 1., 1., 2.]


def test_alignment_never_crosses_episodes_unknown_context_or_gaps():
    cfg = ContactGateConfig(precontact_duration_s=1.)
    signal = torch.tensor([0., 0., 11., 0., 0., 11., 0., 11.])
    valid = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    t = torch.tensor([0., .01, .02, .03, .04, .05, .06, 1.])
    labels = contact_phase_labels_from_signal(signal, [(0, 2), (2, 8)], cfg,
                                              timestamps_s=t, valid_mask=valid)[:, 0]
    assert labels.tolist() == [0., 0., 2., -1., 1., 2., 0., 2.]


@pytest.mark.parametrize('removed', ['phase_label_mode', 'precontact_frames', 'thresholds', 'consecutive_frames',
                                    'on_threshold', 'off_threshold', 'force_on_threshold_n', 'force_off_threshold_n'])
def test_old_three_phase_options_are_rejected(removed):
    with pytest.raises(ValueError, match='Removed dual-threshold'):
        ContactGateConfig.from_config({'contact_gate': {'label_mode': 'three_phase', removed: 1}})


def test_recover_export_lowpass_without_mutating_wm_observations():
    from data_process.causal_data_filter import filter_episode_values
    t = np.arange(200)*.01
    original = np.random.default_rng(42).normal(size=(200, 7))
    filtered = filter_episode_values(t, original, [{'type': 'lowpass', 'cutoff_hz': 20., 'order': 2}])
    before = filtered.copy()
    spec = dict(enabled=True, cutoff_hz=20., order=2, causal=True, history_only=True,
                contract='causal_variable_dt_one_pole_cascade_v1')
    recovered = labeling.restore_export_filters(t, {'tau': filtered}, {'tau': spec})
    np.testing.assert_allclose(recovered['tau'], original, atol=2e-6)
    np.testing.assert_array_equal(filtered, before)


@pytest.fixture
def generated_config(monkeypatch, tmp_path):
    size = 700
    row = torch.arange(size).repeat(2)
    q = torch.sin(row[:, None]*.01+torch.arange(7)[None])*.1
    tau = torch.zeros(2*size, 7)
    tau[:, 0] = torch.where((row >= 400) & (row < 500), 12., 2.)
    columns = {'observation.joint': q, 'observation.velocity': q*.1,
               'observation.delta_q': q*.01, 'observation.torque': tau,
               'action.ee_pose': torch.zeros(2*size, 7),
               'timing.state_timestamp_ns': row*10_000_000,
               'timing.action_anchor_timestamp_ns': (row//4)*40_000_000,
               'timing.action_index': row//4}
    class Table:
        column_names = list(columns)
        def with_format(self, *args, **kwargs):
            return self
        def __getitem__(self, item):
            return columns
    source = SimpleNamespace(hf_dataset=Table(), meta=SimpleNamespace(episodes=[
        dict(episode_index=i, dataset_from_index=i*size, dataset_to_index=(i+1)*size) for i in range(2)]))
    monkeypatch.setattr(dataset_module, '_load_lerobot_dataset_class', lambda: lambda **kwargs: source)
    calls = []
    class Teacher:
        def __init__(self, *args, **kwargs):
            calls.append('load')
        def predict(self, t, q, dq, q_cmd, tau):
            calls.append('predict')
            return dict(timestamp_s=t, tau_free=np.zeros_like(tau), tau_measured=tau.copy(), tau_ext=tau.copy(),
                        valid_context=(t >= t[0]+1.) & (t <= t[-1]-1.))
    monkeypatch.setattr(labeling, 'LearnedTorqueModel', Teacher)
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'teacher-v1')
    cfg = dict(wm_v3_only=True,
        dataloader=dict(root=str(tmp_path), repo_id='fixture', action_key='action.ee_pose',
            state_history_horizon=10, prediction_horizon=4, action_condition_horizon=2, action_start_offset=1,
            high_timestamp_key='timing.state_timestamp_ns', anchor_timestamp_key='timing.action_anchor_timestamp_ns',
            normalize_mode=None, tau_ext_generation=dict(enabled=True, checkpoint=str(checkpoint),
            cache_dir=str(tmp_path/'cache'), threads=2)),
        contact_gate=dict(enabled=True, label_mode='three_phase', metric='tau_ext_l1', contact_threshold=10.,
                          precontact_duration_s=1., class_weights=[1., 1., 1.]),
        model=dict(joint_dim=7, action_dim=7, inputs=['q', 'dq', 'delta_q', 'tau'], outputs=['q', 'tau'],
            hidden_dim=8, latent_dim=5, decoder_hidden_dim=8, state_layers=1, action_layers=1,
            flow_layers=1, flow_attention_heads=2, dropout=0.),
        loss=dict(free_dynamics_weight=.1, lambda_free=.1),
        train=dict(device='cpu', output_dir=str(tmp_path/'run'), downsample=False,
                   contact_sampling=dict(enabled=True, phase_weights=[1., 5., 5.])))
    return cfg, columns, calls


def test_both_datasets_label_before_sampling_and_share_cache(generated_config):
    cfg, columns, calls = generated_config
    before = columns['observation.torque'].clone()
    direct = dataset_module.ContactWorldModelDataset(cfg)
    latent = LatentContactWorldModelDataset(cfg)
    assert direct.tau_label_report['cache_misses'] == 1  # Identical episode inputs can share predictions.
    assert latent.tau_label_report['cache_hits'] == 2
    assert calls.count('load') == 1 and calls.count('predict') == 1
    torch.testing.assert_close(direct.contact, latent.contact)
    torch.testing.assert_close(columns['observation.torque'], before)
    assert direct.contact[99] == -1 and direct.contact[100] == 0
    assert direct.contact[299] == 0 and direct.contact[300] == 1 and direct.contact[400] == 2
    for index in direct.valid_indices:
        assert (direct.contact[index+1:index+5] >= 0).all()
    for constructor, dataset in [(ContactWorldModelTrainer, direct), (LatentContactWorldModelTrainer, latent)]:
        trainer = constructor(cfg)
        trainer.dataset = dataset
        subset = torch.utils.data.Subset(dataset, range(len(dataset)))
        sampler = trainer.build_train_sampler(subset)
        phases = torch.tensor([dataset.future_contact_phase(i) for i in dataset.valid_indices])
        torch.testing.assert_close(sampler.weights, torch.tensor([1., 5., 5.], dtype=torch.double)[phases])
    report = json.loads((Path(cfg['train']['output_dir'])/'tau_label_report.json').read_text())
    assert report['phase_rule']['comparison'] == '>'
    assert all(report['phase_counts'][str(i)] > 0 for i in range(3))


def test_cache_invalidates_for_weights_inputs_but_threshold_changes_relabel_only(generated_config):
    cfg, columns, calls = generated_config
    initial = dataset_module.ContactWorldModelDataset(cfg)
    changed = copy.deepcopy(cfg)
    changed['contact_gate']['contact_threshold'] = 13.
    relabeled = dataset_module.ContactWorldModelDataset(changed)
    assert calls.count('predict') == 1 and not (relabeled.contact == 2).any()
    assert initial.tau_label_report['label_contract_sha256'] != relabeled.tau_label_report['label_contract_sha256']
    columns['observation.torque'][200, 0] = 15.
    dataset_module.ContactWorldModelDataset(cfg)
    assert calls.count('predict') == 2
    Path(cfg['dataloader']['tau_ext_generation']['checkpoint']).write_bytes(b'teacher-v2')
    updated = dataset_module.ContactWorldModelDataset(cfg)
    assert calls.count('predict') == 4 and updated.tau_label_report['cache_misses'] == 2


@pytest.mark.parametrize('family', ['contact', 'latent'])
def test_generated_free_mask_drives_auxiliary_gradients_in_both_models(generated_config, family):
    cfg, _, _ = generated_config
    constructor = dataset_module.ContactWorldModelDataset if family == 'contact' else LatentContactWorldModelDataset
    dataset = constructor(cfg)
    anchors = [120, 320, 420, 99]  # Free, alignment, contact, invalid teacher history.
    samples = torch.utils.data.default_collate([dataset[dataset.valid_indices.index(i)] for i in anchors])
    if family == 'contact':
        model = ContactWorldModel(cfg)
        calculator = ContactWorldModelLoss(cfg)
        head = model.free_dynamics_head
    else:
        model = LatentContactWorldModel(cfg)
        model.codec_ready.fill_(True)
        model.codec_snapshot = 'test'
        model.set_stage('flow')
        calculator = LatentContactWorldModelLoss(cfg)
        head = model.free_tau_head
    output = model(samples, flow_time=.5)
    assert output['_prepared_batch']['free_dynamics_mask'].tolist() == [True, False, False, False]
    loss, count = calculator.free_dynamics_loss(output, samples)
    loss.backward()
    assert count == 1 and any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    full, metrics = calculator(output, samples)
    assert torch.isfinite(full) and (metrics.get('free_dynamics_count', metrics.get('free_auxiliary_count')) == 1)


def test_optional_rollout_contact_loss_uses_temporal_alignment_and_preserves_gradients():
    from train.trainer.contact_world_model_opd_train import ContactWorldModelOPDTrainer
    trainer = ContactWorldModelOPDTrainer.__new__(ContactWorldModelOPDTrainer)
    trainer.rollout_contact_enabled = True
    trainer.rollout_contact_gate = ContactGateConfig(metric='tau_ext_l1', contact_threshold=10.)
    trainer.rollout_contact_horizon = 4
    trainer.rollout_contact_temperature = .05
    trainer.rollout_contact_weight = 0.
    trainer.rollout_contact_physical_weight = 1.
    trainer.loss_calculator = SimpleNamespace(_physical=lambda key, value: value, contact_class_weights=None)
    trainer.tau_free_predictor = lambda history, future: torch.zeros_like(future['q'])
    history = {k: torch.zeros(1, 3, 1) for k in ('q', 'dq', 'delta_q')}
    prediction = {k+'_pred': torch.zeros(1, 4, 1) for k in history}
    prediction['tau_pred'] = torch.tensor([[[0.], [0.], [11.], [0.]]], requires_grad=True)
    prediction['contact_logits'] = torch.nn.functional.one_hot(torch.tensor([[1, 1, 2, 0]]), 3).float()*40.
    loss, metrics = trainer._rollout_contact_loss(history, prediction)
    assert metrics['rollout_contact_physical'] < 1e-6
    prediction['tau_pred'] = torch.full((1, 4, 1), 2., requires_grad=True)
    prediction['contact_logits'] = torch.tensor([[[0., 40., 0.]]]).expand(1, 4, 3)
    loss, metrics = trainer._rollout_contact_loss(history, prediction)
    assert metrics['rollout_contact_physical'] > .1  # No upcoming contact: cannot be alignment.
    loss.backward()
    assert prediction['tau_pred'].grad.abs().sum() > 0


def test_real_format2_network_labels_both_wm_families(generated_config, monkeypatch):
    from model.xarm_tau_free import LearnedTorqueModel
    from model.xarm_tau_sequence import SequenceTorqueRegressor
    cfg, _, _ = generated_config
    spec=dict(arch='bilstm',horizon=51,rate=50,filter='zero5',physics_prior=False)
    network=SequenceTorqueRegressor(spec)
    for p in network.parameters():
        p.data.zero_()
    cp=dict(format_version=2,spec=spec,model=network.state_dict(),normalization=dict(
        x_mean=torch.zeros(21),x_std=torch.ones(21),y_mean=torch.zeros(7),y_std=torch.ones(7)))
    torch.save(cp,cfg['dataloader']['tau_ext_generation']['checkpoint'])
    monkeypatch.setattr(labeling,'LearnedTorqueModel',LearnedTorqueModel)
    direct=dataset_module.ContactWorldModelDataset(cfg)
    latent=LatentContactWorldModelDataset(cfg)
    torch.testing.assert_close(direct.contact,latent.contact)
    assert latent.tau_label_report['cache_hits']==2
    assert set(direct.contact[:,0].tolist())=={-1.,0.,1.,2.}
