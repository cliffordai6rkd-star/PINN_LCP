"""Training/deployment equivalence and absence of measured-torque leakage."""
import numpy as np
import pytest
import torch

from data_process.tool.train_xarm_tau_labels import make_split
from data_process.tool.xarm_tau_offline_experiment import infer
from model.xarm_tau_free import LearnedTorqueModel, predict_episode
from model.xarm_tau_sequence import SequenceTorqueRegressor


def episode(n=600):
    t=np.arange(n)*.01
    q=np.sin(t[:,None]+np.arange(7))*.2
    dq=np.cos(t[:,None]+np.arange(7))*.2
    return dict(name='test',t=t,timestamp_us=np.arange(n)*10000,
                q=q,dq=dq,q_cmd=q+dq*.05,delta_q=dq*.05,tau=q*10,
                valid=np.ones(n,dtype=bool))


@pytest.mark.parametrize('arch', ['lstm','bilstm'])
@pytest.mark.parametrize('mode,rate', [('zero5',50),('raw',100),('zero5',100),('zero10',50)])
def test_portable_teacher_matches_training_and_cannot_read_tau(tmp_path,arch,mode,rate):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    spec=dict(arch=arch,horizon=rate+(arch=='bilstm'),filter=mode,rate=rate,physics_prior=False)
    model=SequenceTorqueRegressor(spec).eval()
    norm=dict(x_mean=np.zeros(21,np.float32),x_std=np.ones(21,np.float32),
              y_mean=np.arange(7,dtype=np.float32),y_std=np.ones(7,np.float32))
    cp=dict(format_version=2,spec=spec,model=model.state_dict(),
            normalization={k:torch.tensor(v) for k,v in norm.items()})
    path=tmp_path/'model.pt';torch.save(cp,path)
    teacher=LearnedTorqueModel(path)
    e=episode()
    expected,target,valid=infer(model,e,spec,norm,'cpu')
    actual=teacher.predict(e['t'],e['q'],e['dq'],e['q_cmd'],e['tau'])
    np.testing.assert_allclose(actual['tau_free'],expected,atol=1e-6)
    np.testing.assert_allclose(actual['tau_measured'],target)
    np.testing.assert_array_equal(actual['valid_context'],valid)
    changed=teacher.predict(e['t'],e['q'],e['dq'],e['q_cmd'],e['tau']+100)
    np.testing.assert_array_equal(changed['tau_free'],actual['tau_free'])
    np.testing.assert_allclose(changed['tau_ext']-actual['tau_ext'],100,atol=.03)
    sparse=teacher.predict(e['t'],e['q'],e['dq'],e['q_cmd'],e['tau'],output_times=e['t'][::4])
    np.testing.assert_array_equal(sparse['tau_free'],actual['tau_free'][::4])
    e['valid'][300]=False
    segmented=predict_episode(teacher,e)
    assert not segmented['valid_context'][200:401].any()
    assert np.isnan(segmented['tau_ext'][300]).all()


def test_split_has_no_shared_rows():
    e=episode(73135)
    blocks,ids=make_split(e)
    assert len(ids['train'])==16 and len(ids['validation'])==len(ids['calibration'])==4
    row_sets={role:set(np.concatenate([b['timestamp_us'] for b in values])) for role,values in blocks.items()}
    assert not row_sets['train'] & row_sets['validation']
    assert not row_sets['train'] & row_sets['calibration']
    assert not row_sets['validation'] & row_sets['calibration']
    assert len(set.union(*row_sets.values()))==72000


def test_teacher_rejects_wrong_preprocessing(tmp_path):
    path=tmp_path/'model.pt'
    torch.save(dict(format_version=2,spec=dict(arch='lstm',filter='unknown',rate=50,horizon=50)),path)
    with pytest.raises(ValueError,match='contract'):
        LearnedTorqueModel(path)
