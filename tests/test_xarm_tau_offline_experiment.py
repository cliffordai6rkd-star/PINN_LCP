import numpy as np
import pytest
import torch

from data_process.tool import xarm_tau_offline_experiment as experiment
from data_process.tool.xarm_tau_curve_report import phase_labels


def test_report_uses_strict_threshold_and_temporal_alignment():
    t=np.arange(400)/100
    score=np.ones(400)
    score[200:300]=11
    score[230:240]=10  # Equality stops contact; the next onset gives alignment.
    phases=phase_labels(t,score,10)
    assert np.all(phases[:100]==0)
    assert np.all(phases[100:200]==1)
    assert np.all(phases[200:230]==2)
    assert np.all(phases[230:240]==1)
    assert np.all(phases[240:300]==2)
    assert np.all(phases[300:]==0)


def test_brief_spike_is_contact_without_old_debounce():
    t=np.arange(300)/100
    score=np.ones(300)
    score[150:154]=20
    phases=phase_labels(t,score,10)
    assert np.all(phases[:50]==0)
    assert np.all(phases[50:150]==1)
    assert np.all(phases[150:154]==2)
    assert np.all(phases[154:]==0)


def test_precontact_never_overwrites_previous_contact():
    t=np.arange(500)/100
    score=np.ones(500)
    score[200:270]=10
    score[320:400]=10
    phases=phase_labels(t,score,8)
    assert np.all(phases[200:270]==2)
    assert np.all(phases[270:320]==1)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError,match='contact_threshold'):
        phase_labels(np.arange(10),np.zeros(10),-1)


def test_visualization_helpers_match_training_and_exclude_unknown_context():
    from data_process.tool.contact_signal_rerun import _phase_labels
    t=np.arange(400)*.01
    score=np.zeros(400)
    score[250:280]=11.
    expected=phase_labels(t,score)
    np.testing.assert_array_equal(_phase_labels(score,t,10.),expected)
    valid=np.ones(400,dtype=bool)
    valid[200]=False
    phase=phase_labels(t,score,valid_mask=valid)
    assert np.all(phase[:200]==0) and phase[200]==-1
    assert np.all(phase[201:250]==1)


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
