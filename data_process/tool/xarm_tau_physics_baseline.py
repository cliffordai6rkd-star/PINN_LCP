#!/usr/bin/env python3
"""Offline, regularized inverse-dynamics identification control experiment.

This is an unconstrained torque regressor, not a physically certified inertia
model. Fit coefficients use background recordings only; validation chooses ridge.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pinocchio as pin
from scipy.signal import butter,sosfiltfilt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from data_process.tool.xarm_tau_offline_experiment import read_episode,preprocess,metrics,save_json


def regressors(episode,model,mode):
    p=preprocess(episode,mode,50)
    q,dq=p['q'].astype(float),p['dq'].astype(float)
    ddq=np.gradient(dq,.02,axis=0)
    data=model.createData()
    matrices=[];nominal=[]
    for i in range(len(q)):
        y=pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i]).copy()
        # Per-joint viscous, smooth Coulomb, low-speed friction, and bias terms.
        fric=np.zeros((7,28))
        for j in range(7):
            fric[j,j*4:(j+1)*4]=[dq[i,j],np.tanh(dq[i,j]/.02),np.tanh(dq[i,j]/.002),1.]
        matrices.append(np.concatenate([y,fric],axis=1))
        nominal.append(pin.rnea(model,data,q[i],dq[i],ddq[i]).copy())
    valid=(p['t']>p['t'][0]+1)&(p['t']<p['t'][-1]-1)
    return np.asarray(matrices),np.asarray(nominal),p,valid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bg',type=Path,default=ROOT/'data/xarm_bg/bg_data')
    parser.add_argument('--urdf',type=Path,default=ROOT.parent/'xarm_ws/gello_teleop/models/xarm7_dynamics.urdf')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/xarm_tau_offline_20261004/physics')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=sorted(args.bg.glob('*.h5'))
    model=pin.buildModelFromUrdf(str(args.urdf))
    assert model.nq==model.nv==7
    validation=read_episode(files[2],'right')
    common=sosfiltfilt(butter(4,5,fs=100,output='sos'),validation['tau'],axis=0)
    results=[]
    for mode in ('zero10','zero5'):
        started=time.monotonic()
        vx,base,vp,valid=regressors(validation,model,mode)
        for mixed in (False,True):
            episodes=[read_episode(files[1],'right')]
            if mixed:episodes.extend([read_episode(files[1],'left'),read_episode(files[2],'left')])
            xtx=np.zeros((98,98));xty=np.zeros(98);rows=0
            for episode in episodes:
                x,nominal,p,mask=regressors(episode,model,mode)
                x=x[mask].reshape(-1,98);target=(p['tau'][mask]-nominal[mask]).reshape(-1)
                xtx+=x.T@x;xty+=x.T@target;rows+=len(target)
            scale=np.sqrt(np.diag(xtx)/rows).clip(1e-3)
            gram=xtx/scale[:,None]/scale[None,:]/rows
            rhs=xty/scale/rows
            best=None
            for ridge in (1e-5,1e-4,1e-3,1e-2,.1,1.,10.):
                coefficients=np.linalg.solve(gram+ridge*np.eye(98),rhs)/scale
                prediction=base+np.einsum('nji,i->nj',vx,coefficients)
                score=metrics(prediction[valid],vp['tau'][valid])
                if best is None or score['mse_nm2']<best[0]['mse_nm2']:
                    best=(score,ridge,coefficients,prediction)
            native,ridge,coefficients,prediction=best
            pred100=np.stack([np.interp(validation['t'],vp['t'],prediction[:,j]) for j in range(7)],axis=1)
            mask=(validation['t']>1)&(validation['t']<validation['t'][-1]-1)
            name=f'identified_dynamics_{"mixed" if mixed else "right"}_{mode}'
            result={'name':name,'filter':mode,'mixed':mixed,'ridge':ridge,'validation':native,
                    'validation_common5':metrics(pred100[mask],common[mask]),'seconds':time.monotonic()-started}
            results.append(result)
            np.savez_compressed(args.output/(name+'.npz'),coefficients=coefficients,tau_pred=pred100,time_s=validation['t'],valid_context=mask)
            save_json(args.output/'results.json',results)
            print(name,'ridge',ridge,'native_rmse',round(native['rmse_nm'],4),'common5_rmse',round(result['validation_common5']['rmse_nm'],4),flush=True)


if __name__=='__main__':main()
