import numpy as np
import pytest
import torch

from data_process.tool import xarm_tau_offline_experiment as experiment
from data_process.tool.xarm_tau_curve_report import phase_labels


def test_phase_bands_and_hysteresis_precontact():
    t=np.arange(400)/100
    score=np.ones(400)
    score[200:300]=10
    score[230:240]=5  # Between a and b: hysteresis keeps contact active.
    bands,phases=phase_labels(t,score,3,8)
    assert np.all(phases[:100]==0)
    assert np.all(phases[100:200]==1)
    assert np.all(phases[200:300]==2)
    assert np.all(phases[300:]==0)
    assert np.all(bands[230:240]==1)


def test_brief_spike_does_not_create_contact_or_precontact():
    t=np.arange(300)/100
    score=np.ones(300)
    score[150:154]=20
    _,phases=phase_labels(t,score,3,8)
    assert np.all(phases==0)


def test_precontact_never_overwrites_previous_contact():
    t=np.arange(500)/100
    score=np.ones(500)
    score[200:270]=10
    score[320:400]=10
    _,phases=phase_labels(t,score,3,8)
    assert np.all(phases[200:270]==2)
    assert np.all(phases[270:320]==1)


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError,match='low < high'):
        phase_labels(np.arange(10),np.zeros(10),5,3)


def synthetic_episode(n=600):
    t=np.arange(n)/100
    q=np.repeat(np.sin(t)[:,None],7,axis=1)
    return {'t':t,'timestamp_us':np.arange(n)*10000,'valid':np.ones(n,dtype=bool),
            'name':'synthetic','q':q,'dq':np.cos(q),'delta_q':q*.001,'q_cmd':q*1.001,'tau':q*2}


def test_windows_remain_within_their_blocks():
    first,second=synthetic_episode(),synthetic_episode()
    spec={'arch':'bilstm','filter':'zero5','rate':50,'horizon':51}
    x,y,ends,norm=experiment.build_cache([first,second],spec,torch.device('cpu'))
    windows=experiment.window_indexes(ends,spec).numpy()
    assert np.all((windows[:,0]//300)==(windows[:,-1]//300))
    assert len(x)==len(y)==600
    assert np.isfinite(norm['x_std']).all()


def test_inference_restores_physics_prior_once(monkeypatch):
    episode=synthetic_episode()
    spec={'arch':'mlp','filter':'zero5','rate':50,'horizon':1,'physics_prior':'mock'}
    norm={'x_mean':np.zeros(21,dtype=np.float32),'x_std':np.ones(21,dtype=np.float32),
          'y_mean':np.full(7,2,dtype=np.float32),'y_std':np.ones(7,dtype=np.float32)}
    model=experiment.Regressor(spec)
    for p in model.parameters():p.data.zero_()
    monkeypatch.setattr(experiment,'physics_prior',lambda processed,spec:np.full_like(processed['tau'],10))
    prediction,target,valid=experiment.infer(model,episode,spec,norm,torch.device('cpu'))
    np.testing.assert_allclose(prediction,12)
    assert len(prediction)==len(episode['t'])
    assert not valid[:100].any()
    assert valid[100:-100].all()
    assert target.shape==prediction.shape
