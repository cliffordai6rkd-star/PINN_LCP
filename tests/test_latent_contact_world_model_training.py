"""Grid, native episode windows, optimizer counts and full stage resume."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from test_contact_world_model_dataset import make_dataset
from test_latent_contact_world_model import threads
from test_latent_contact_world_model import make_next_checkpoint
from data_process.relative_time_grid import RelativeTimeGrid, GridMetadataError, grid_sinusoidal
from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer


def test_four_phases_quantization_and_no_clamping():
    grid=RelativeTimeGrid()
    origin=1_800_000_000_000_000_001
    current=origin+torch.arange(4,dtype=torch.int64)*10_000_000
    history=current[:,None]+torch.arange(-5,1)*10_000_000
    action=torch.tensor([origin+40_000_000,origin+80_000_000,origin+120_000_000]).expand(4,-1)
    positions=grid.positions(history_ns=history,action_ns=action,future_horizon=4)
    assert positions['action_grid_positions'].tolist()==[[4,8,12],[3,7,11],[2,6,10],[1,5,9]]
    delta=torch.tensor([-15_000_000,-5_000_000,-4_999_999,4_999_999,5_000_000,15_000_000])
    assert grid.quantize(delta).tolist()==[-2,-1,0,0,1,2]
    one=history[:1]
    action0=one[:,-1:]+torch.tensor([[1_000_000,41_000_000]])
    assert grid.positions(history_ns=one,action_ns=action0,future_horizon=4)['action_grid_positions'].tolist()==[[0,4]]
    action5=one[:,-1:]+torch.tensor([[50_000_000,90_000_000]])
    assert grid.positions(history_ns=one,action_ns=action5,future_horizon=4)['action_grid_positions'].tolist()==[[5,9]]


def test_metadata_missing_bad_cadence_and_wrong_next_anchor():
    grid=RelativeTimeGrid();history=torch.tensor([[0,10_000_000]])
    with pytest.raises(GridMetadataError,match='required'): grid.positions(future_horizon=4)
    for action in (torch.tensor([[40_000_000,100_000_000]]),torch.tensor([[9_000_000,49_000_000]])):
        with pytest.raises(GridMetadataError,match='cadence'): grid.positions(history_ns=history,action_ns=action,future_horizon=4)
    with pytest.raises(GridMetadataError,match='int64'): grid.quantize(torch.tensor([0.5]))


def test_explicit_grid_cross_plan_tokens_and_shared_signed_sinusoidal():
    grid=RelativeTimeGrid()
    explicit={'anchor_grid_position':torch.tensor([200]),'history_grid_positions':torch.tensor([[198,199,200]]),
              'action_grid_positions':torch.tensor([[201,205,210,214]]),'future_grid_positions':torch.tensor([[201,202,203]])}
    result=grid.positions(explicit=explicit,future_horizon=3)
    assert result['action_grid_positions'].tolist()==[[1,5,10,14]]
    positive=grid_sinusoidal(torch.tensor([[1,4]]),8,dtype=torch.float64)
    negative=grid_sinusoidal(torch.tensor([[-1,-4]]),8,dtype=torch.float64)
    torch.testing.assert_close(positive[...,0::2],-negative[...,0::2])
    torch.testing.assert_close(positive[...,1::2],negative[...,1::2])


def test_adapter_native_action_phases_and_episode_reset(make_dataset):
    original,cfg=make_dataset(offset=1,future=4)
    adapted=LatentContactWorldModelDataset(cfg)
    for raw in range(52,56):
        sample=adapted[adapted.valid_indices.index(raw)]
        assert sample['action_grid_positions'][0]==4-(raw-52)
        assert sample['future_grid_positions'].tolist()==[1,2,3,4]
        assert sample['history_grid_positions'][-1]==0
        torch.testing.assert_close(sample['action'],original[original.valid_indices.index(raw)]['action'])
    for episode in adapted.episodes:
        raw=int(episode['dataset_from_index'])
        sample=adapted[adapted.valid_indices.index(raw)]
        assert sample['history_indices'].min()>=raw
        assert sample['action_grid_positions'][0]==4
        assert not sample['history_valid_mask'].any()  # Disabled labels cannot confirm free.


def test_adapter_explicit_drop_policy_and_error(make_dataset,monkeypatch):
    original,cfg=make_dataset(offset=1,future=4)
    # Modify source timing with a real 8ms action jump. Reconstruct via the
    # same source loader; the old dataset still keeps its original semantics.
    source=original.source_dataset
    columns=source.hf_dataset[:]
    columns['timing.action_anchor_timestamp_ns'][40:44]+=8_000_000
    with pytest.raises(GridMetadataError,match='windows violate'):
        LatentContactWorldModelDataset(cfg)
    cfg['dataloader']['grid_invalid_window_policy']='drop'
    adapted=LatentContactWorldModelDataset(cfg)
    assert adapted.grid_audit['invalid_windows']>0
    assert len(adapted)<len(original)


def training_case(make_dataset,tmp_path):
    _,cfg=make_dataset(offset=1,future=4)
    cfg=copy.deepcopy(cfg)
    cfg['train']={'device':'cpu','output_dir':str(tmp_path),'downsample':False,'batch_size':8,'num_workers':0,
                  'val_num_workers':0,'val_ratio':0.5,'split_mode':'episode','seed':42,'stage':'all',
                  'max_optimizer_steps':6,'checkpoint_every_steps':2,'top_k':2,'recovery_every_steps':1,
                  'wandb':{'enabled':False},'ema':{'enabled':True,'decay':0.9},'val_every':100,
                  'probabilistic_validation':{'enabled':True,'num_samples':2,'max_batches':1},
                  'device_batch_keys':['q'],'gradient_every':2}
    cfg['codec']={'max_optimizer_steps':2}
    cfg['model'].update(latent_dim=5,decoder_hidden_dim=8,dropout=0.0)
    cfg['dataloader']['normalize_mode']=None
    return cfg


def test_two_stage_real_optimizer_counts_topk_and_reconstruction(make_dataset,tmp_path):
    cfg=training_case(make_dataset,tmp_path)
    trainer=LatentContactWorldModelTrainer(cfg);trainer.train()
    assert trainer.codec_step==2 and trainer.global_step==6
    assert (tmp_path/'codec.pt').exists()
    assert sorted(p.name for p in (tmp_path/'checkpoints').glob('step_*.pt'))==['step_00000004.pt','step_00000006.pt']
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['status']=='complete' and status['flow_step']==6
    assert trainer.final_validation['validation_nfe']==32
    assert trainer.codec_validation['contact_0_support']>0
    for name in trainer.model.CODEC_MODULES:
        assert not getattr(trainer.model,name).training
        assert all(not p.requires_grad for p in getattr(trainer.model,name).parameters())
    codec=torch.load(tmp_path/'codec.pt',weights_only=False)
    assert codec['statistics']['training_future_frames']==len(trainer.train_dataset)*4
    assert {'history_grid_positions','action_grid_positions','future_grid_positions'}<=trainer.device_batch_keys


@pytest.mark.parametrize('stage',['codec','flow'])
def test_resume_exact_raw_ema_rng_and_batch_cursor(make_dataset,tmp_path,stage):
    cfg=training_case(make_dataset,tmp_path/'interrupted')
    uninterrupted=LatentContactWorldModelTrainer({**cfg,'train':{**cfg['train'],'output_dir':str(tmp_path/'full')}})
    uninterrupted.train()
    partial=LatentContactWorldModelTrainer(cfg);partial.setup()
    if stage=='flow':
        partial.run_stage();partial.finalize_codec()
    partial.run_stage(stop_after_updates=1)
    cursor=partial.batch_cursor
    saved=torch.load(partial.ckpt_dir/'latest.pt',weights_only=False)
    assert saved['latent_training']['batch_cursor']==cursor
    cfg['train']['resume_from']=str(partial.output_dir)
    resumed=LatentContactWorldModelTrainer(cfg);resumed.train()
    assert resumed.codec_step==2 and resumed.global_step==6
    for key,value in uninterrupted.model.state_dict().items():
        if torch.is_tensor(value): torch.testing.assert_close(value,resumed.model.state_dict()[key],rtol=0,atol=0)
    for key,value in uninterrupted.ema.model.state_dict().items():
        if torch.is_tensor(value): torch.testing.assert_close(value,resumed.ema.model.state_dict()[key],rtol=0,atol=0)


def test_codec_checkpoint_skips_stage_a(make_dataset,tmp_path):
    cfg=training_case(make_dataset,tmp_path/'codec')
    cfg['train']['stage']='codec'
    trainer=LatentContactWorldModelTrainer(cfg);trainer.train()
    cfg['train'].update(stage='flow',output_dir=str(tmp_path/'flow'))
    cfg['codec']['checkpoint_path']=str(tmp_path/'codec/codec.pt')
    loaded=LatentContactWorldModelTrainer(cfg);loaded.train()
    assert loaded.codec_step==2 and loaded.global_step==6
    assert loaded.model.codec_snapshot==trainer.model.codec_snapshot


def test_pretrained_trainer_resume_without_original_next_file(make_dataset,tmp_path):
    cfg=training_case(make_dataset,tmp_path/'run')
    # The synthetic source is expanded to the explicitly required 7 joints.
    from data_process import contact_world_model_dataset as dataset_module
    source=dataset_module._load_lerobot_dataset_class()()
    for key,value in source.hf_dataset[:].items():
        if key.startswith(('observation.','action.')):
            source.hf_dataset[:][key]=value[:,:1].expand(-1,7).clone()
    cfg['dataloader']['state_history_horizon']=6
    cfg['model'].update(joint_dim=7,action_dim=7,hidden_dim=128)
    path=tmp_path/'next.pt'
    cfg['model']['pretrained_taufree_path']=str(path)
    make_next_checkpoint(path,cfg)
    trainer=LatentContactWorldModelTrainer(cfg);trainer.setup()
    trainer.run_stage();trainer.finalize_codec();trainer.run_stage(stop_after_updates=1)
    expected=trainer.model.motion_encoder.weight_ih_l0.clone()
    path.unlink()
    cfg['train']['resume_from']=str(trainer.output_dir)
    resumed=LatentContactWorldModelTrainer(cfg);resumed.train()
    torch.testing.assert_close(resumed.model.motion_encoder.weight_ih_l0,expected,rtol=0,atol=0)
    assert not resumed.model.motion_encoder.training and resumed.model.free_tau_head is None
    assert resumed.global_step==6


def test_task_configs_preserve_workspace_baselines():
    root=Path(__file__).resolve().parents[1]/'config/train_cfg'
    for source,dest in [('cwm_insert_usb_100hz_40step.yaml','latent_cwm_insert_usb_100hz_40step.yaml'),
                        ('cwm_peel_cucumber_40step.yaml','latent_cwm_peel_cucumber_40step.yaml')]:
        original=yaml.safe_load((root/'pretrain'/source).read_text());derived=yaml.safe_load((root/dest).read_text())
        for key in ('train_data','action_contract','contact_gate'): assert original[key]==derived[key]
        for key in ('batch_size','lr','weight_decay','scheduler','amp','device','split_mode','val_ratio','seed','contact_sampling','ema'):
            assert original['train'][key]==derived['train'][key]
        for key in original['dataloader']:
            assert original['dataloader'][key]==derived['dataloader'][key]
        assert derived['train']['max_optimizer_steps']==250000 and derived['train']['top_k']==5
        assert derived['codec']['max_optimizer_steps']==10000


def test_independent_trainer_defaults_and_canonical_normalization():
    trainer=LatentContactWorldModelTrainer({'train':{'device':'cpu'}})
    assert str(trainer.output_dir)=='outputs/latent_carswm_lstm_grid/default'
    assert trainer.flow_max_steps==250000 and trainer.codec_max_steps==10000
    assert trainer.top_k==5 and trainer.checkpoint_every_steps==50000
    assert trainer.config['dataloader']['normalize_mode']=='gaussian'


def test_codec_periodic_validation_uses_reconstruction_loss(make_dataset,tmp_path):
    cfg=training_case(make_dataset,tmp_path)
    cfg['train'].update(stage='codec',val_every=1)
    cfg['codec']['max_optimizer_steps']=10  # Pass an epoch before finishing Stage A.
    trainer=LatentContactWorldModelTrainer(cfg);trainer.train()
    records=[json.loads(line) for line in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    mid=[r for r in records if r['stage']=='codec_validation' and r['codec_step']<10]
    assert mid and all('codec_q_mse' in r and 'latent_fm_loss' not in r for r in mid)
    assert trainer.codec_step==10 and trainer.global_step==0
