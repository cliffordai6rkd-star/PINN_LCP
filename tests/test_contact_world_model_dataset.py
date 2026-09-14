"""100 Hz state anchors and independent recorded 25 Hz action tokens."""
from types import SimpleNamespace

import pytest
import torch
import yaml

from data_process import contact_world_model_dataset as dataset_module
from model.pinn_model.contact_world_model import ContactWorldModel


@pytest.fixture
def make_dataset(monkeypatch, tmp_path):
    def make(*, offset=0, future=16, missing=None, hold=4):
        # Two episodes reset both clocks and action indices. Each token is held
        # for four state rows; anchor times deliberately differ from state times.
        n = 160
        rows = torch.arange(n).repeat(2)
        indices = rows // hold
        anchors = indices * (hold * 10_000_000) + 8_000_000
        columns = {
            field: rows.float()[:, None].repeat(1, 2)
            for field in dataset_module.V3_STATE_FIELDS.values()
        }
        columns['action.ee_pose'] = (indices + torch.arange(2).repeat_interleave(n) * 100).float()[:, None].repeat(1, 2)
        columns.update({
            'timing.state_timestamp_ns': rows * 10_000_000 + 10_000_000,
            'timing.action_index': indices,
            'timing.action_anchor_timestamp_ns': anchors,
        })
        if missing:
            columns.pop(missing)
        class Table:
            def with_format(self, *args, **kwargs):
                return self
            def __getitem__(self, item):
                return columns
        source = SimpleNamespace(hf_dataset=Table(), meta=SimpleNamespace(episodes=[
            {'dataset_from_index': 0, 'dataset_to_index': n},
            {'dataset_from_index': n, 'dataset_to_index': 2*n},
        ]))
        monkeypatch.setattr(dataset_module, '_load_lerobot_dataset_class', lambda: lambda **kwargs: source)
        cfg = {
            'wm_v3_only': True,
            'dataloader': {
                'root': str(tmp_path), 'repo_id': 'fixture', 'action_key': 'action.ee_pose',
                'state_history_horizon': 50, 'prediction_horizon': future,
                'action_condition_horizon': 8, 'action_start_offset': offset,
                'high_timestamp_key': 'timing.state_timestamp_ns',
                'anchor_timestamp_key': 'timing.action_anchor_timestamp_ns',
                'normalize_mode': None,
            },
            'model': {'inputs': ['q', 'dq', 'delta_q', 'tau'], 'outputs': ['q', 'tau'],
                      'joint_dim': 2, 'action_dim': 2, 'hidden_dim': 8,
                      'state_layers': 1, 'action_layers': 1, 'flow_layers': 1,
                      'flow_attention_heads': 2, 'dropout': 0.0},
            'train': {'downsample': True},
        }
        return dataset_module.ContactWorldModelDataset(cfg), cfg
    return make


@pytest.mark.parametrize('offset', [0, 1, 2])
def test_held_actions_slide_on_unique_indices_and_stay_in_episode(make_dataset, offset):
    dataset, _ = make_dataset(offset=offset)
    samples = [dataset[dataset.valid_indices.index(i)] for i in range(52, 57)]
    for i, sample in enumerate(samples):
        assert sample['history_indices'][-1] == 52+i
        assert sample['future_indices'][0] == 53+i
        assert torch.all(torch.diff(sample['history_indices']) == 1)
        assert torch.all(torch.diff(sample['future_indices']) == 1)
        expected = torch.arange(13 + offset + i//4, 21 + offset + i//4)
        torch.testing.assert_close(sample['action_chunk_index'], expected)
        torch.testing.assert_close(sample['action'][:, 0], expected.float())
        torch.testing.assert_close(sample['action_chunk_timestamp_ns'], expected * 40_000_000 + 8_000_000)
        if i < 4:
            torch.testing.assert_close(sample['action'], samples[0]['action'])
    torch.testing.assert_close(samples[4]['action'][:-1], samples[0]['action'][1:])
    for episode in dataset.episodes:
        start, end = episode['dataset_from_index'], episode['dataset_to_index']
        valid = [i for i in dataset.valid_indices if start <= i < end]
        assert valid == list(range(valid[0], valid[-1]+1))
        for raw in (valid[0], valid[-1]):
            sample = dataset[dataset.valid_indices.index(raw)]
            assert sample['history_indices'].min() >= start
            assert sample['future_indices'].max() < end
            assert torch.all(torch.diff(sample['action_chunk_index']) == 1)
            assert sample['action_chunk_index'][-1] < 40
            assert torch.all((sample['action'][:, 0] >= 100) == (start == 160))
        with pytest.raises(IndexError, match='action table'):
            dataset._action_for_anchor(end-1, episode)


@pytest.mark.parametrize('future', [16, 32])
@pytest.mark.parametrize('downsample', [False, True])
def test_downsample_preserves_actions_and_encodes_state_lengths(make_dataset, future, downsample):
    dataset, cfg = make_dataset(future=future)
    cfg['train']['downsample'] = downsample
    stride = 2 if downsample else 1
    public = torch.utils.data.default_collate([dataset[52], dataset[53]])
    model = ContactWorldModel(cfg).eval()
    prepared = model.prepare_batch(public)
    assert public['q'].shape == (2, 50, 2)
    assert public['q_future'].shape == (2, future, 2)
    assert prepared['q'].shape == (2, 50//stride, 2)
    assert prepared['q_future'].shape == (2, future//stride, 2)
    torch.testing.assert_close(prepared['q'][:, -1], public['q'][:, -1])
    for key in ('action', 'action_mask', 'action_time', 'action_chunk_index', 'action_chunk_timestamp_ns'):
        assert prepared[key].shape[1] == 8
        torch.testing.assert_close(prepared[key], public[key])
    assert model.action_rate_hz == 25
    assert model.state_rate_hz == 100/stride
    encoded = model.encode_conditions(public)
    assert encoded['state_tokens'].shape == (2, 4 * (50//stride), 8)
    assert encoded['action_tokens'].shape == (2, 8, 8)
    assert 'condition_memory' not in encoded
    assert 'action_time' not in encoded
    assert 'future_time' not in encoded
    output = model(public, flow_time=0.5)
    assert output['flow_target_state'].shape == (2, future//stride, 4)
    repeated = model.prepare_batch(prepared)
    for key in prepared:
        torch.testing.assert_close(repeated[key], prepared[key])


@pytest.mark.parametrize('missing', ['timing.action_index', 'timing.action_anchor_timestamp_ns'])
def test_v3_requires_action_metadata(make_dataset, missing):
    with pytest.raises(KeyError):
        make_dataset(missing=missing)


@pytest.mark.parametrize('name', ['cwm_insert_usb_50hz', 'cwm_all_50hz', 'contact_world_model_npu'])
def test_training_configs_keep_eight_recorded_action_tokens(name):
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/train_cfg' / (name+'.yaml')).read_text())
    data = cfg['dataloader']
    assert data['action_condition_horizon'] == 8
    assert data['expert_fps'] == 25
    assert data['high_fps'] == 100
    assert data['v3_only']
    assert data['action_index_key'] == 'timing.action_index'
    assert data['action_anchor_timestamp_key'] == 'timing.action_anchor_timestamp_ns'


def test_v3_rejects_old_100hz_action_indices(make_dataset):
    with pytest.raises(ValueError, match='reconvert'):
        make_dataset(hold=1)


def test_padding_validity_current_normalized_tau_and_history_alignment(make_dataset):
    from dataclasses import replace
    from train.nomalizer import Normalizer
    dataset, cfg = make_dataset()
    dataset.contact_gate_config = replace(dataset.contact_gate_config, enabled=True)
    dataset.contact.zero_()
    dataset.normalize_mode = 'gaussian'
    dataset.set_normalizer(Normalizer({'tau': {'mean': torch.tensor([10., 10.]),
                                            'std': torch.tensor([2., 2.])}}))
    first = dataset[0]
    assert not first['history_valid_mask'].all()
    assert first['history_valid_mask'][-1]
    sample = dataset[52]
    assert sample['history_valid_mask'].all()
    public = torch.utils.data.default_collate([sample])
    model = ContactWorldModel(cfg)
    prepared = model.prepare_batch(public)
    assert prepared['free_dynamics_mask'].item()
    assert prepared['contact'].shape[1] == prepared['q'].shape[1] == 25
    assert prepared['history_valid_mask'].shape[1] == 25
    raw_tau = dataset.high_tensors['tau'][sample['sample_idx']]
    expected = dataset.normalizer.gaussian_normalize('tau', raw_tau)
    torch.testing.assert_close(prepared['tau'][0, -1], expected)
    for episode in dataset.episodes:
        raw = int(episode['dataset_from_index'])
        sample = dataset[dataset.valid_indices.index(raw)]
        assert sample['history_valid_mask'].sum() == 1
    dataset.contact_gate_config = replace(dataset.contact_gate_config, enabled=False)
    assert not dataset[52]['history_valid_mask'].any()
