"""Semantic tests for the independent latent model, codec and NEXT transfer."""
import copy

import pytest
import torch
from torch import nn

from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from model.pinn_model.latent_pretrained import normalizer_envelope
from model.tau_other_lstm import TauOtherLSTMRegressor
from train.latent_contact_world_model_loss import LatentContactWorldModelLoss
from train.nomalizer import Normalizer


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(*, pretrained=False):
    return {"dataloader":{"state_history_horizon":6,"prediction_horizon":4,"action_condition_horizon":3,
                          "high_fps":100,"expert_fps":25,"normalize_mode":"gaussian",
                          "normalize_lowdim_keys":["q","dq","delta_q","tau","action"]},
            "model":{"joint_dim":7 if pretrained else 2,"action_dim":7,"hidden_dim":128 if pretrained else 8,
                     "latent_dim":5,"decoder_hidden_dim":8,"flow_layers":1,"flow_attention_heads":2,
                     "dropout":0.1 if pretrained else 0.0,"pretrained_taufree_path":"placeholder" if pretrained else None},
            "train":{"downsample":False}, "loss":{"lambda_free":0.1}}


def batch(cfg, b=3):
    d,m = cfg['dataloader'],cfg['model']
    l,h,k,j=d['state_history_horizon'],d['prediction_horizon'],d['action_condition_horizon'],m['joint_dim']
    values={key:torch.randn(b,l,j) for key in ('q','dq','delta_q','tau')}
    values.update(action=torch.randn(b,k,m['action_dim']),action_mask=torch.ones(b,k,dtype=torch.bool),
                  history_grid_positions=torch.arange(1-l,1).expand(b,-1),
                  action_grid_positions=torch.arange(k).expand(b,-1)*4+torch.arange(b)[:,None]+1,
                  future_grid_positions=torch.arange(1,h+1).expand(b,-1),
                  q_future=torch.randn(b,h,j),tau_future=torch.randn(b,h,j),
                  contact_future=(torch.arange(b*h)%3).reshape(b,h,1).float(),contact=torch.zeros(b,l,1),
                  history_valid_mask=torch.ones(b,l,dtype=torch.bool),importance_weight=torch.arange(1,b+1).float())
    return values


def ready_model(cfg):
    model=LatentContactWorldModel(cfg)
    model.codec_ready.fill_(True)
    model.codec_snapshot='synthetic'
    model.set_stage('flow')
    return model


def test_exact_three_two_layer_lstm_and_motion_order():
    cfg=config(); values=batch(cfg); model=ready_model(cfg)
    lstms=[module for module in model.modules() if isinstance(module,nn.LSTM)]
    assert len(lstms)==3
    assert all(m.num_layers==2 and m.batch_first and not m.bidirectional for m in lstms)
    seen={}
    def record(name):
        def hook(module,args): seen[name]=args[0]
        return hook
    handles=[getattr(model,name).register_forward_pre_hook(record(name)) for name in ('motion_encoder','tau_encoder','action_encoder')]
    out=model(values,flow_time=0.3)
    for handle in handles: handle.remove()
    torch.testing.assert_close(seen['motion_encoder'],torch.cat([values[k] for k in ('q','dq','delta_q')],-1))
    assert seen['motion_encoder'].shape[-1]==6
    assert out['history'].shape==(3,12,8) and out['action'].shape==(3,3,8)
    for name in model.CODEC_MODULES:
        assert sum(isinstance(m,nn.Linear) for m in getattr(model,name).modules())==2


def test_three_conditioners_participate_in_main_fm_gradient():
    cfg=config();model=ready_model(cfg);values=batch(cfg)
    loss,_=LatentContactWorldModelLoss(cfg).flow_loss(model(values),values)
    loss.backward()
    for name in ('motion_encoder','tau_encoder','action_encoder'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters())
    assert all(p.grad is None for name in model.CODEC_MODULES for p in getattr(model,name).parameters())


def test_codec_and_flow_gradient_isolation_and_fixed_velocity_target():
    cfg=config();values=batch(cfg);model=LatentContactWorldModel(cfg);calculator=LatentContactWorldModelLoss(cfg)
    out=model(values);loss,_=calculator(out,values);loss.backward()
    assert all(p.grad is not None for name in model.CODEC_MODULES for p in getattr(model,name).parameters())
    assert all(p.grad is None for name in model.CONDITION_MODULES for p in getattr(model,name).parameters())
    model.zero_grad(set_to_none=True);model.codec_ready.fill_(True);model.set_stage('flow');model.train()
    assert all(not getattr(model,name).training for name in model.CODEC_MODULES)
    before={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith(model.CODEC_MODULES)}
    z0=torch.randn(3,4,5);out=model(values,flow_time=0.25,source_noise=z0)
    target=model.target_latent(values)
    assert not target.requires_grad and not out['flow_target_latent'].requires_grad
    torch.testing.assert_close(out['flow_state'],0.75*z0+0.25*target)
    torch.testing.assert_close(out['flow_velocity_target'],target-z0)
    loss,metrics=calculator(out,values)
    expected=((out['flow_velocity_pred']-(target-z0)).square().mean((1,2))*values['importance_weight']).mean()
    torch.testing.assert_close(metrics['latent_fm_loss'],expected.detach())
    optimizer=torch.optim.AdamW(model.parameter_groups(),lr=0.01);loss.backward();optimizer.step()
    for name,p in model.named_parameters():
        if name in before: torch.testing.assert_close(p,before[name],rtol=0,atol=0)


def test_free_aux_only_motion_and_head_and_current_tau():
    cfg=config();values=batch(cfg);model=ready_model(cfg).eval();calculator=LatentContactWorldModelLoss(cfg)
    out=model(values,flow_time=0.4);free,count=calculator.free_dynamics_loss(out,values)
    expected=(out['free_tau_pred']-values['tau'][:,-1]).square().mean(-1)
    torch.testing.assert_close(free,(expected*values['importance_weight']).sum()/values['importance_weight'].sum())
    assert count==3
    free.backward()
    for name,p in model.named_parameters():
        if name.startswith(('motion_encoder.','free_tau_head.')):
            assert p.grad is not None
        else: assert p.grad is None,name
    changed=dict(values)
    for key in ('tau','action','q_future','tau_future'): changed[key]=torch.randn_like(values[key])*100
    changed['contact_future']=2-values['contact_future']
    torch.testing.assert_close(model(changed,flow_time=0.4)['free_tau_pred'],out['free_tau_pred'])


@pytest.mark.parametrize('reason',['contact','padding','nan','missing_contact'])
def test_free_mask_before_stride_and_empty_sets(reason):
    cfg=config();cfg['train']['downsample']=True;values=batch(cfg)
    if reason=='padding': values['history_valid_mask'][:,0]=False
    elif reason=='missing_contact': values['history_valid_mask'].zero_()
    else: values['contact'][:,0]=float('nan') if reason=='nan' else 2
    model=ready_model(cfg);out=model(values)
    assert not out['_prepared_batch']['free_dynamics_mask'].any()
    loss,count=LatentContactWorldModelLoss(cfg).free_dynamics_loss(out,values)
    assert loss==0 and count==0 and torch.isfinite(loss)
    loss.backward()
    assert model.free_tau_head[-1].weight.grad.count_nonzero()==0
    assert out['_prepared_batch']['future_grid_positions'][0].tolist()==[1,3]
    assert out['_prepared_batch']['history_grid_positions'][0].tolist()==[-4,-2,0]
    torch.testing.assert_close(out['_prepared_batch']['tau'][:,-1],values['tau'][:,-1])


def test_nonfree_nan_and_zero_weight_aux_semantics():
    prediction=torch.tensor([[1.,3.],[2.,4.],[float('nan'),float('nan')]],requires_grad=True)
    values={'tau':torch.tensor([[[0.,0.]],[[0.,0.]],[[float('nan'),float('nan')]]]),
            'free_dynamics_mask':torch.tensor([True,True,False]),'importance_weight':torch.tensor([1.,3.,200.])}
    loss,count=LatentContactWorldModelLoss(config()).free_dynamics_loss({'free_tau_pred':prediction},values)
    assert loss==8.75 and count==2
    values['importance_weight'].zero_();zero,_=LatentContactWorldModelLoss(config()).free_dynamics_loss({'free_tau_pred':prediction},values)
    assert zero==0 and torch.isfinite(zero)


def test_sample_only_conditions_full_heun_interval_and_shared_decoding():
    cfg=config();model=ready_model(cfg).eval();values=batch(cfg)
    condition={key:values[key] for key in model.CONDITION_KEYS}
    times=[]
    def velocity(z,time,encoded,**kwargs):
        times.append(float(time));return torch.ones_like(z)*2
    model.velocity=velocity
    model.latent_mean.fill_(3);model.latent_std.fill_(4)
    source=torch.randn(3,2,4,5)
    sampled=model.sample(condition,num_samples=2,source_noise=source)
    assert len(times)==32 and times[0]==0 and times[-1]==1 and sampled['nfe']==32
    torch.testing.assert_close(sampled['latent'],source+2)
    torch.testing.assert_close(sampled['raw_latent'],(source+2)*4+3)
    decoded=model.decode(sampled['raw_latent'])
    for key in ('q_pred','tau_pred','contact_logits'):
        torch.testing.assert_close(sampled[key],decoded[key])
        assert sampled[key].shape[:3]==(3,2,4)
    assert not torch.equal(sampled['q_pred'][:,0],sampled['q_pred'][:,1])
    again=model.sample(condition,num_samples=2,source_noise=source)
    torch.testing.assert_close(again['q_pred'],sampled['q_pred'],rtol=0,atol=0)


def make_next_checkpoint(path,cfg,*,change=None):
    next_cfg={'dataloader':{'horizon':6,'expected_fps':100,'pad_history':False,'normalize_mode':'gaussian',
                            'normalize_lowdim_keys':['q','dq','delta_q','tau']},
              'model':{'architecture':'lstm','target_key':'tau','inputs':['q','dq','delta_q'],
                       'input_dims':{'q':7,'dq':7,'delta_q':7},'output_dim':7,'hidden_dim':128,'num_layers':2,
                       'history_mode':'stateless_sliding_window','head_hidden_dim':256,'dropout':0.1}}
    model=TauOtherLSTMRegressor(next_cfg).eval()
    stats={key:{'mean':torch.arange(7).float()+2,'std':torch.arange(7).float()+0.5}
           for key in ('q','dq','delta_q','tau')}
    envelope=normalizer_envelope(Normalizer(stats),next_cfg)
    payload={'config':next_cfg,'model':model.state_dict(),'normalizer':envelope,
             'sample_rate_hz':100,'dataloader_filters':{}}
    if change: change(payload)
    torch.save(payload,path)
    return model,envelope


def test_next_numerical_transfer_different_normalizers_freeze_and_selfcontained_restore(tmp_path):
    cfg=config(pretrained=True);path=tmp_path/'next.pt';cfg['model']['pretrained_taufree_path']=str(path)
    next_model,envelope=make_next_checkpoint(path,cfg)
    model=LatentContactWorldModel(cfg)
    wmstats={key:{'mean':torch.arange(7).float()-3,'std':torch.arange(7).float()+3}
             for key in ('q','dq','delta_q','tau')}
    model.wm_normalizer=normalizer_envelope(Normalizer(wmstats),cfg)
    model.initialize_pretrained_motion()
    values=batch(cfg);model.codec_ready.fill_(True);model.set_stage('flow');model.train()
    assert not model.motion_encoder.training and model.tau_encoder.training and model.action_encoder.training
    assert model.free_tau_head is None
    physical={key:Normalizer(wmstats).gaussian_denormalize(key,values[key]) for key in ('q','dq','delta_q')}
    normalized={key:Normalizer(envelope['stats']).gaussian_normalize(key,physical[key]) for key in physical}
    expected=next_model.recurrent(torch.cat(list(normalized.values()),-1))[0]
    actual=model.motion_features(values)
    torch.testing.assert_close(actual,expected)
    optimizer=torch.optim.AdamW(model.parameter_groups())
    included={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(not p.requires_grad and id(p) not in included for p in model.motion_encoder.parameters())
    loss,_=LatentContactWorldModelLoss(cfg)(model(values),values);loss.backward()
    assert all(p.grad is None for p in model.motion_encoder.parameters())
    assert all(p.grad is not None for p in model.tau_encoder.parameters())
    saved=tmp_path/'latent.pt'
    torch.save({'config':cfg,'model_version':model.MODEL_VERSION,'carswm_contract':model.checkpoint_contract(),
                'model':model.state_dict()},saved)
    path.unlink()
    restored,_=load_latent_checkpoint(saved)
    torch.testing.assert_close(restored.motion_features(values),actual)
    assert restored.pretrained_contract['selected_file']==str(path)
    assert not restored.motion_encoder.training


@pytest.mark.parametrize('field,value,match',[
    ('architecture','gru','architecture'),('hidden_dim',64,'hidden_dim'),('inputs',['dq','q','delta_q'],'inputs'),
    ('num_layers',1,'num_layers'),('target_key','tau_other','target_key')])
def test_bad_next_semantics_fail(tmp_path,field,value,match):
    cfg=config(pretrained=True);path=tmp_path/'next.pt';cfg['model']['pretrained_taufree_path']=str(path)
    make_next_checkpoint(path,cfg,change=lambda p:p['config']['model'].update({field:value}))
    with pytest.raises(ValueError,match=match): LatentContactWorldModel(cfg).initialize_pretrained_motion()


def test_missing_next_key_and_preprocessing_incompatibility(tmp_path):
    cfg=config(pretrained=True);path=tmp_path/'next.pt';cfg['model']['pretrained_taufree_path']=str(path)
    make_next_checkpoint(path,cfg,change=lambda p:p['model'].pop('recurrent.weight_ih_l0'))
    with pytest.raises(RuntimeError,match='Missing key'): LatentContactWorldModel(cfg).initialize_pretrained_motion()
    make_next_checkpoint(path,cfg)
    cfg['dataloader']['filters']={'dq':{'enabled':True,'operations':[{'type':'lowpass','cutoff_hz':15.0,'order':1}]}}
    with pytest.raises(ValueError,match='incompatible preprocessing'): LatentContactWorldModel(cfg).initialize_pretrained_motion()


@pytest.mark.parametrize('change,match',[
    (lambda p:p.update(sample_rate_hz=50),'sample_rate_hz'),
    (lambda p:p['config']['dataloader'].update(horizon=5),'history horizon'),
    (lambda p:p['config']['model']['input_dims'].update(q=6),'input_dims.q'),
    (lambda p:p.pop('normalizer'),'normalizer'),
    (lambda p:p.pop('dataloader_filters'),'preprocessing')])
def test_next_cadence_dimensions_and_missing_contracts(tmp_path,change,match):
    cfg=config(pretrained=True);path=tmp_path/'next.pt';cfg['model']['pretrained_taufree_path']=str(path)
    make_next_checkpoint(path,cfg,change=change)
    with pytest.raises(ValueError,match=match): LatentContactWorldModel(cfg).initialize_pretrained_motion()


def test_checkpoint_contract_strict_and_parameter_freezing_interface():
    cfg=config();model=ready_model(cfg)
    restored=LatentContactWorldModel(cfg);restored.load_state_dict(model.state_dict(),strict=True)
    restored.validate_checkpoint({'model_version':model.MODEL_VERSION,'carswm_contract':model.checkpoint_contract()})
    with pytest.raises(ValueError,match='mismatch'): restored.validate_checkpoint_contract({})
    restored.freeze_modules(restored.FLOW_MODULES);restored.train()
    assert all(not p.requires_grad for name in restored.FLOW_MODULES for p in getattr(restored,name).parameters())
    assert restored.motion_encoder.training


def test_input_scale_validation_does_not_depend_on_user_normalizer_extension(monkeypatch):
    from model.pinn_model.latent_pretrained import convert_scale
    monkeypatch.delattr(Normalizer,'validate',raising=False)
    envelope={'stats':{'q':{'mean':torch.zeros(2),'std':torch.ones(2)}},'eps':1e-6,
              'normalize_mode':'gaussian','normalize_lowdim_keys':['q']}
    result=convert_scale('q',torch.ones(2),envelope)
    torch.testing.assert_close(result,torch.ones(2)/(1+1e-6))
    envelope['stats']['q']['std'][0]=float('nan')
    with pytest.raises(ValueError,match='normalizer q.std'): convert_scale('q',torch.ones(2),envelope)
