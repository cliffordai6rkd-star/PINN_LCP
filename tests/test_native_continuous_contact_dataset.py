"""Missing offline contact assets never turn into fabricated free labels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from data_process.native_continuous_contact_dataset import NativeContinuousContactDataset
from test_contact_world_model import config


def test_native_wrapper_disables_only_dataset_labeling(tmp_path, monkeypatch):
    (tmp_path/'meta').mkdir();(tmp_path/'data').mkdir()
    (tmp_path/'meta/info.json').write_text('{}')
    (tmp_path/'meta/world_model_timeline.json').write_text(json.dumps({'nominal_lerobot_fps':100}))
    (tmp_path/'data/file.parquet').write_bytes(b'content-hash-fixture')
    cfg = config()
    cfg['contact_gate'] = {'enabled':True,'contact_threshold':11.7}
    cfg['dataloader']['tau_ext_generation'] = {'enabled':True,'checkpoint':'original-model.pt'}
    original = copy.deepcopy(cfg)
    seen = []

    def base_init(self, loader, **kwargs):
        seen.append(loader)
        self.lerobot_source_specs = [{'root':tmp_path}]
        self.high_fps = 100
        self.valid_indices = [0]
        self.dataset = SimpleNamespace(meta=SimpleNamespace(episodes=[{}]))

    def base_sample(self, index):
        return {'q':torch.ones(5,2),'contact':torch.zeros(5,1),'contact_future':torch.zeros(4,1),
                'future_phase':torch.tensor(0),'free_dynamics_mask':torch.tensor(True)}

    monkeypatch.setattr(ContactWorldModelDataset,'__init__',base_init)
    monkeypatch.setattr(ContactWorldModelDataset,'_build_sample',base_sample)
    data = NativeContinuousContactDataset(cfg)
    assert cfg == original
    assert not seen[0]['contact_gate']['enabled']
    assert not seen[0]['dataloader']['tau_ext_generation']['enabled']
    assert seen[0]['dataloader']['high_keys']['tau_ext'] is None
    assert set(data._build_sample(0)) == {'q'}
    assert data.provenance['contact_labels_available'] is False
    assert data.provenance['interpolation'] is False
