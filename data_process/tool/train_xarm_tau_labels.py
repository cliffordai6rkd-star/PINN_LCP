"""Reproducible free-torque comparison with independent calibration and test data.

The first recording is split into disjoint 30-second blocks BEFORE filtering.
The second recording is test-only. Only the chosen arm is used; task torques
never supervise the regressor. All three models receive q, dq, delta_q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from data_process.tool.xarm_tau_offline_experiment import (
    Regressor, build_cache, eval_cache, infer, metrics, preprocess,
    read_episode, save_json, select_rows, summary, window_indexes,
)


def make_split(record, block_rows=3000, seed=42):
    n = len(record['t']) // block_rows
    if n < 12:
        raise ValueError('Need at least twelve 30-second background blocks')
    shuffled = np.random.default_rng(seed).permutation(n)
    reserved = max(2, n//6)
    ids = dict(validation=sorted(shuffled[:reserved].tolist()),
               calibration=sorted(shuffled[reserved:2*reserved].tolist()),
               train=sorted(shuffled[2*reserved:].tolist()))
    blocks = {role: [select_rows(record, i*block_rows, (i+1)*block_rows) for i in indexes]
              for role, indexes in ids.items()}
    return blocks, ids


def dynamics_features(block, physics, mode='zero5', rate=50):
    import pinocchio as pin
    p = preprocess(block, mode, rate)
    q, dq = p['q'].astype(float), p['dq'].astype(float)
    ddq = np.gradient(dq, 1/rate, axis=0)
    data = physics.createData()
    x, base = [], []
    for i in range(len(q)):
        dynamic = pin.computeJointTorqueRegressor(physics, data, q[i], dq[i], ddq[i]).copy()
        fric = np.zeros((7, 28))
        for j in range(7):
            fric[j, j*4:j*4+4] = [dq[i,j], np.tanh(dq[i,j]/.02), np.tanh(dq[i,j]/.002), 1.]
        x.append(np.concatenate((dynamic, fric), axis=1))
        base.append(pin.rnea(physics, data, q[i], dq[i], ddq[i]).copy())
    return np.asarray(x)[rate:-rate], (p['tau']-np.asarray(base))[rate:-rate]


def fit_physics(blocks, urdf, output, mode='zero5', rate=50):
    import pinocchio as pin
    physics = pin.buildModelFromUrdf(str(urdf))
    if physics.nq != 7 or physics.nv != 7:
        raise ValueError('Expected the xArm seven-joint dynamics model')
    gram = np.zeros((98,98)); rhs = np.zeros(98); rows = 0
    for block in blocks['train']:
        x,y = dynamics_features(block, physics, mode, rate)
        x=x.reshape(-1,98); y=y.reshape(-1)
        gram+=x.T@x; rhs+=x.T@y; rows+=len(y)
    scale=np.sqrt(np.diag(gram)/rows).clip(1e-3)
    gram=gram/scale[:,None]/scale[None,:]/rows; rhs=rhs/scale/rows
    validation=[dynamics_features(b,physics,mode,rate) for b in blocks['validation']]
    vx=np.concatenate([x for x,y in validation]); vy=np.concatenate([y for x,y in validation])
    candidates=[]; best=None
    for ridge in (1e-5,1e-4,1e-3,1e-2,.1,1.,10.):
        coeff=np.linalg.solve(gram+ridge*np.eye(98),rhs)/scale
        score=metrics(np.einsum('nji,i->nj',vx,coeff),vy)
        candidates.append(dict(ridge=ridge,validation=score))
        if best is None or score['mse_nm2']<best[0]: best=(score['mse_nm2'],coeff,ridge)
    path=output/'physics.npz'
    np.savez_compressed(path,coefficients=best[1])
    save_json(output/'physics_selection.json',dict(candidates=candidates,selected_ridge=best[2]))
    print('PHYSICS',best[2],np.sqrt(best[0]),flush=True)
    return path, best[1]


def train_model(blocks, spec, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=torch.device(args.device)
    cache=build_cache(blocks['train'],spec,device)
    normalization=cache[-1]
    validation=build_cache(blocks['validation'],spec,device,normalization)
    model=Regressor(spec).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5,min_lr=1e-5)
    x,y,indexes,_=cache
    scale=torch.tensor(normalization['y_std'],device=device)
    best=float('inf'); best_epoch=0; history=[]; started=time.monotonic()
    best_state=None
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        sampled=indexes[torch.randint(len(indexes),(args.samples_per_epoch,),device=device)]
        for batch in sampled.split(args.batch_size):
            prediction=model(x[window_indexes(batch,spec)])
            loss=((prediction-y[batch])*scale).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(loss.detach())
        val=eval_cache(model,validation,spec,normalization)
        scheduler.step(val['mse_nm2'])
        row=dict(epoch=epoch,train_mse_nm2=float(torch.stack(losses).mean()),
                 validation_mse_nm2=val['mse_nm2'],lr=optimizer.param_groups[0]['lr'],
                 elapsed_s=time.monotonic()-started)
        history.append(row)
        if val['mse_nm2']<best:
            best=val['mse_nm2']; best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        save_json(args.output/(spec['name']+'_history.json'),history)
        print('TRAIN',spec['name'],json.dumps(row),flush=True)
        if epoch-best_epoch>=args.patience: break
    model.load_state_dict(best_state)
    checkpoint=dict(model=best_state,spec=spec,normalization=normalization,
                    best_epoch=best_epoch,history=history,seed=args.seed)
    torch.save(checkpoint,args.output/(spec['name']+'_training.pt'))
    return model, checkpoint


def block_predictions(model,blocks,spec,norm,device):
    predictions=[]; targets=[]; scores=[]
    for block in blocks:
        pred,target,valid=infer(model,block,spec,norm,device)
        predictions.append(pred[valid]); targets.append(target[valid])
        scores.append(np.abs(target[valid]-pred[valid]).sum(1))
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(scores)


def export_checkpoint(cp,coefficients,urdf,threshold,path,split):
    spec={k:v for k,v in cp['spec'].items() if k not in ('physics_prior','urdf_path')}
    spec['physics_prior']=bool(cp['spec'].get('physics_prior'))
    artifact=dict(format_version=2,model=cp['model'],spec=spec,
                  normalization={k:torch.as_tensor(v) for k,v in cp['normalization'].items()},
                  coefficients=None if not spec['physics_prior'] else torch.as_tensor(coefficients),
                  urdf_xml=None if not spec['physics_prior'] else urdf.read_text(),
                  calibration=dict(contact_threshold=float(threshold),quantile=.995,
                      metric='tau_ext_l1',precontact_duration_s=1.,source='calibration blocks only'),
                  split=split,seed=cp['seed'],best_epoch=cp['best_epoch'],
                  contract=dict(inputs=['q','dq','delta_q'],target='contact-free measured tau',
                      preprocessing=dict(uniform_grid_hz=100,filter=spec['filter'],
                                         rate_hz=spec['rate'],filter_order=4),
                      edge_exclusion_s=1.,offline=True,arm='right'))
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(artifact,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--background',type=Path,default=ROOT/'data/xarm_bg')
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/tau_free_sequence/xarm_training')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--filter',choices=['raw','zero5','zero10'],default='zero5')
    p.add_argument('--rate',type=int,choices=[50,100],default=50)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--patience',type=int,default=15)
    p.add_argument('--samples-per-epoch',type=int,default=32768);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--resume-completed',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    files=sorted(args.background.glob('*.h5'))
    if len(files)!=2: raise ValueError('Specify a directory with exactly two background recordings')
    record,test=[read_episode(f,'right') for f in files]
    blocks,ids=make_split(record,seed=args.split_seed)
    split=dict(recording_1=str(files[0].resolve()),test_recording=str(files[1].resolve()),
               block_rows=3000,block_ids=ids,edge_exclusion_s=1.,arm='right',seed=args.split_seed,
               discarded_tail_rows=len(record['t'])%3000,
               source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    save_json(args.output/'split.json',split)
    save_json(args.output/'arguments.json',vars(args))
    save_json(args.output/'audit.json',dict(background=[summary(record),summary(test)],
        environment=dict(torch=str(torch.__version__),device=torch.cuda.get_device_name(0))))
    physics_path,coefficients=fit_physics(blocks,args.urdf,args.output,args.filter,args.rate)
    specs=[dict(name='lstm',arch='lstm',filter=args.filter,rate=args.rate,horizon=args.rate),
           dict(name='bilstm',arch='bilstm',filter=args.filter,rate=args.rate,horizon=args.rate+1),
           dict(name='physics_bilstm',arch='bilstm',filter=args.filter,rate=args.rate,horizon=args.rate+1,
                physics_prior=str(physics_path.resolve()),urdf_path=str(args.urdf.resolve()))]
    for spec in specs:
        spec['dropout']=args.dropout
    results=[]
    for spec in specs:
        existing=args.output/(spec['name']+'_training.pt')
        if args.resume_completed and existing.exists():
            cp=torch.load(existing,map_location='cpu',weights_only=False)
            model=Regressor(spec).to(args.device);model.load_state_dict(cp['model'])
        else:
            model,cp=train_model(blocks,spec,args)
        norm=cp['normalization']
        vp,vt,vs=block_predictions(model,blocks['validation'],spec,norm,args.device)
        cpred,ct,cs=block_predictions(model,blocks['calibration'],spec,norm,args.device)
        threshold=float(np.quantile(cs,.995))
        # Freeze model and threshold before reading test metrics.
        export_checkpoint(cp,coefficients,args.urdf,threshold,args.output/'deployment'/f"{spec['name']}.pt",split)
        tp,tt,valid=infer(model,test,spec,norm,args.device)
        ts=np.abs(tt-tp).sum(1)
        result=dict(name=spec['name'],spec=spec,best_epoch=cp['best_epoch'],
            parameter_count=sum(p.numel() for p in model.parameters()),
            seconds=cp['history'][-1]['elapsed_s'],validation=metrics(vp,vt),
            calibration=metrics(cpred,ct),contact_threshold=threshold,
            calibration_false_positive_fraction=float(np.mean(cs>threshold)),
            test=metrics(tp[valid],tt[valid]),test_false_positive_fraction=float(np.mean(ts[valid]>threshold)),
            test_valid_frames=int(valid.sum()),test_false_contact_frames=int(np.sum(ts[valid]>threshold)))
        results.append(result)
        np.savez_compressed(args.output/f"{spec['name']}_test.npz",time_s=test['t'],tau_pred=tp,tau_target=tt,
                            tau_ext=tt-tp,score=ts,valid_context=valid)
        save_json(args.output/'comparison.json',results)
        print('RESULT',spec['name'],json.dumps({k:result[k] for k in ('best_epoch','contact_threshold','test_false_positive_fraction')}),
              'RMSE',result['test']['rmse_nm'],flush=True)
        del model
        torch.cuda.empty_cache()
    winner=min(results,key=lambda r:r['validation']['mse_nm2'])
    source=args.output/'deployment'/f"{winner['name']}.pt"
    (args.output/'deployment/model.pt').write_bytes(source.read_bytes())
    save_json(args.output/'selection.json',dict(selected=winner['name'],
        criterion='Minimum validation MSE, independent of calibration/test/task data',
        checkpoint=str((args.output/'deployment/model.pt').resolve()),contact_threshold=winner['contact_threshold']))
    print('COMPLETE',winner['name'],flush=True)


if __name__=='__main__': main()
