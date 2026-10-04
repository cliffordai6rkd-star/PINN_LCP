#!/usr/bin/env python3
"""Audit xArm raw H5 and compare offline free-torque regressors without editing inputs.

Examples:
    python data_process/tool/xarm_tau_offline_experiment.py --audit-only
    python data_process/tool/xarm_tau_offline_experiment.py --epochs 30

Default training/validation use disjoint recording-1 time blocks, split BEFORE filtering.
With --split archived, reconstruct the supplied checkpoint's split (both arms in
recording 1 plus left recording 2 train; right recording 2 validates).
Windows never cross blocks or episodes.
Peeling torque is never used as free-space supervision or for model selection.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, resample_poly
from scipy.spatial import cKDTree
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_process.causal_data_filter import filter_episode_values
from model.tau_other_sequence import build_tau_other_sequence_model

KEYS = ('q', 'dq', 'delta_q', 'tau')


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value))
    path.write_text(json.dumps(payload, default=default, indent=2, allow_nan=False) + '\n')


def read_episode(path, arm='right'):
    with h5py.File(path, 'r') as f:
        t_us = f['teleop/timestamp_us'][:]
        if 'teleop/q_follower' in f:
            column = slice(0, 7) if arm == 'left' else slice(7, 14)
            columns = {k: f[f'teleop/{name}'][:, column] for k, name in
                       [('q', 'q_follower'), ('dq', 'dq_follower'), ('tau', 'tau_follower'), ('q_cmd', 'q_cmd')]}
            prefixes = {'q': 'q_follower_valid', 'dq': 'dq_valid_follower', 'tau': 'torque_valid_follower'}
            age_key, repeated_key = 'q_follower_age_us', 'q_follower_repeated'
            arm_index = 0 if arm == 'left' else 1
            stored_delta = f['teleop/delta_q'][:, column]
        else:
            columns = {k: f[f'teleop/{arm}_{k}_xarm'][:] for k in ('q', 'dq', 'tau', 'q_cmd')}
            prefixes = {'q': f'{arm}_q_valid_xarm', 'dq': f'{arm}_dq_valid_xarm', 'tau': f'{arm}_torque_valid_xarm'}
            age_key, repeated_key = f'{arm}_q_age_us_xarm', f'{arm}_q_repeated_xarm'
            arm_index = 0
            stored_delta = f[f'teleop/{arm}_delta_q_xarm'][:]
        columns['delta_q'] = columns['q_cmd'] - columns['q']
        valid = np.ones(len(t_us), dtype=bool)
        validity = {}
        for key, name in prefixes.items():
            a = f['teleop/' + name][:].reshape(len(t_us), -1)
            a = a[:, min(arm_index, a.shape[1] - 1)].astype(bool)
            validity[key] = float(a.mean())
            valid &= a
        for value in columns.values():
            valid &= np.isfinite(value).all(axis=1)
        age = f['teleop/' + age_key][:].reshape(len(t_us), -1)
        repeated = f['teleop/' + repeated_key][:].reshape(len(t_us), -1)
        config = yaml.safe_load(f['config_yaml'][()])
        meta = json.loads(f['metadata/episode_json'][()])
        extra = {
            'validity_fraction': validity, 'valid_rows_fraction': float(valid.mean()),
            'q_age_ms_p50_p99_max': np.percentile(age[:, min(arm_index, age.shape[1]-1)] / 1000, [50, 99, 100]),
            'q_repeated_fraction': float(repeated[:, min(arm_index, repeated.shape[1]-1)].mean()),
            'stored_delta_max_discrepancy': float(np.nanmax(np.abs(stored_delta - columns['delta_q']))),
            'control': config.get('control'), 'arm_config': config.get('arms', {}).get(arm),
            'firmware': meta.get('xarm_firmware'),
        }
        cameras = {key: f[f'cameras/{key}/timestamp_us'][:] for key in f.get('cameras', {})}
    return {'path': str(path.resolve()), 'name': path.stem + '_' + arm, 'arm': arm,
            't': (t_us - t_us[0]) * 1e-6, 'timestamp_us': t_us, 'valid': valid,
            'cameras': cameras, 'extra': extra, **columns}


def summary(episode):
    t = episode['t']
    stats = {'name': episode['name'], 'rows': len(t), 'duration_s': float(t[-1]),
             'dt_ms_p0_p50_p99_p100': np.percentile(np.diff(t)*1000, [0, 50, 99, 100]),
             'gaps_over_30ms': int(np.sum(np.diff(t) > .03)), **episode['extra'],
             'camera_frames': {k: len(v) for k, v in episode['cameras'].items()}}
    for key in KEYS:
        a = episode[key]
        stats[key] = {'p01': np.nanpercentile(a, 1, axis=0), 'p99': np.nanpercentile(a, 99, axis=0),
                      'mean': np.nanmean(a, axis=0), 'std': np.nanstd(a, axis=0),
                      'unchanged_row_fraction': float(np.mean(np.all(np.diff(a, axis=0) == 0, axis=1)))}
    grad = np.gradient(episode['q'], t, axis=0)
    stats['dq_gradient_correlation'] = [float(np.corrcoef(grad[:,j], episode['dq'][:,j])[0,1]) for j in range(7)]
    freq, power = welch(episode['tau'], fs=100, nperseg=min(2048, len(t)), axis=0)
    stats['torque_power_fraction_above_10hz'] = power[freq > 10].sum(axis=0) / np.maximum(power[1:].sum(axis=0), 1e-12)
    return stats


def audit(bg, peel, checkpoint, output):
    report = {'background': [summary(e) for e in bg], 'peeling': [summary(e) for e in peel]}
    right = [e for e in bg if e['arm'] == 'right' and len(e['t']) > 1000]
    reference = np.concatenate([np.concatenate([e[k] for k in ('q', 'dq', 'delta_q')], axis=1) for e in right])
    target = np.concatenate([np.concatenate([e[k] for k in ('q', 'dq', 'delta_q')], axis=1) for e in peel])
    mean, std = reference.mean(axis=0), reference.std(axis=0).clip(1e-6)
    reference_z, target_z = (reference-mean)/std, (target-mean)/std
    tree = cKDTree(reference_z[::5])
    nearest = tree.query(target_z[::5], workers=4)[0] / np.sqrt(reference.shape[1])
    report['coverage'] = {
        'right_bg_rows': len(reference), 'peeling_rows': len(target),
        'peeling_outside_bg_minmax_fraction_per_feature': np.mean((target < reference.min(axis=0)) | (target > reference.max(axis=0)), axis=0),
        'peeling_abs_z_p99_per_feature': np.percentile(np.abs(target_z), 99, axis=0),
        'nearest_standardized_rms_p50_p90_p99': np.percentile(nearest, [50,90,99]),
        'features_order': ['q[0:7]', 'dq[0:7]', 'delta_q[0:7]'],
    }
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    report['checkpoint'] = {'path': checkpoint, 'epoch': ckpt['epoch'], 'sample_rate_hz': ckpt.get('sample_rate_hz'),
                            'monitor_score': ckpt['monitor_score'], 'config': ckpt['config']}
    save_json(output/'audit.json', report)
    np.savez_compressed(output/'coverage.npz', nearest_standardized_rms=nearest)
    print('AUDIT', json.dumps({'bg_rows': [len(e['t']) for e in bg], 'peel_rows':len(target),
           'coverage_nearest_p50_p90_p99': np.percentile(nearest,[50,90,99]).tolist(),
           'peel_outside_bg_feature_fraction':report['coverage']['peeling_outside_bg_minmax_fraction_per_feature'].round(4).tolist()}), flush=True)
    return report


def select_rows(episode, start, stop):
    return {**episode, **{k: episode[k][start:stop].copy() for k in (*KEYS, 'q_cmd', 't', 'timestamp_us', 'valid')}}


def preprocess(episode, mode, rate):
    """Filter one isolated block; resample on its measured timeline first."""
    if not episode['valid'].all() or np.any(np.diff(episode['t']) > .03):
        raise ValueError(f'Invalid data/gap requires segmentation: {episode["name"]}')
    t = episode['t']
    grid = np.arange(t[0], t[-1] + 1e-8, .01)
    columns = {k: np.stack([np.interp(grid, t, episode[k][:, j]) for j in range(7)], axis=1) for k in KEYS}
    if mode == 'causal10':
        for key in ('dq', 'tau'):
            columns[key] = filter_episode_values(grid, columns[key], [{'type':'lowpass', 'cutoff_hz':10.}])
    elif mode in ('zero10', 'zero5'):
        sos = butter(4, 10 if mode == 'zero10' else 5, fs=100, output='sos')
        columns = {k:sosfiltfilt(sos, a, axis=0) for k,a in columns.items()}
    elif mode != 'raw':
        raise ValueError(mode)
    if rate == 50:
        columns = {k:resample_poly(a, 1, 2, axis=0, padtype='line') for k,a in columns.items()}
        grid = grid[::2]
    elif rate != 100:
        raise ValueError(rate)
    return {'t':grid, **{k:a.astype(np.float32) for k,a in columns.items()}}


def metrics(prediction, target):
    error = prediction.astype(np.float64) - target
    l1, l2 = np.abs(error).sum(axis=-1), np.linalg.norm(error, axis=-1)
    return {'mse_nm2':float(np.mean(error**2)), 'rmse_nm':float(np.sqrt(np.mean(error**2))),
            'mae_nm':float(np.mean(np.abs(error))), 'rmse_joint_nm':np.sqrt(np.mean(error**2,axis=0)),
            'bias_joint_nm':error.mean(axis=0), 'l1_p50_p95_p99_nm':np.percentile(l1,[50,95,99]),
            'l2_p50_p95_p99_nm':np.percentile(l2,[50,95,99])}


@torch.inference_mode()
def predict_original(episode, checkpoint, device):
    model = build_tau_other_sequence_model(checkpoint['config']).to(device).eval()
    model.load_state_dict(checkpoint['model'])
    # Mirror the checkpoint filters on the raw recorder timeline, independently per H5.
    columns = {k:episode[k].astype(np.float32) for k in KEYS}
    for key, spec in checkpoint['dataloader_filters'].items():
        if spec['enabled']:
            columns[key] = filter_episode_values(episode['t'], columns[key], spec['operations'])
    stats = checkpoint['normalizer']['stats']
    active = checkpoint['config']['model']['inputs']
    normalized = {k:torch.tensor((columns[k]-np.asarray(stats[k]['mean']))/(np.asarray(stats[k]['std'])+1e-6),device=device) for k in active}
    horizon = checkpoint['config']['dataloader']['horizon']
    predictions = []
    for start in range(horizon-1,len(episode['t']),512):
        ends = torch.arange(start,min(start+512,len(episode['t'])),device=device)
        indexes = ends[:,None] + torch.arange(1-horizon,1,device=device)[None,:]
        batch = {k:a[indexes] for k,a in normalized.items()}
        predictions.append(model(batch)['tau_other_pred'].cpu().numpy())
    prediction = np.concatenate(predictions)*(np.asarray(stats['tau']['std'])+1e-6) + np.asarray(stats['tau']['mean'])
    return prediction, columns['tau'][horizon-1:]


class Regressor(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        input_dim = (14 if spec.get('no_delta') else 21) + (7 if spec.get('ddq') else 0)
        if spec['arch'] == 'mlp':
            self.encoder = None
            self.head = nn.Sequential(nn.Linear(input_dim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,7))
        else:
            bidirectional = spec['arch'] == 'bilstm'
            self.encoder = nn.LSTM(input_dim,128,2,batch_first=True,dropout=.1,bidirectional=bidirectional)
            self.head = nn.Sequential(nn.Linear(256 if bidirectional else 128,256),nn.ReLU(),nn.Dropout(.1),nn.Linear(256,7))

    def forward(self, value):
        if self.encoder is None:
            return self.head(value[:,-1])
        value,_ = self.encoder(value)
        return self.head(value[:, value.shape[1]//2 if self.spec['arch']=='bilstm' else -1])


def model_features(processed, spec):
    values = [processed[k] for k in (('q','dq') if spec.get('no_delta') else ('q','dq','delta_q'))]
    if spec.get('ddq'):
        values.append(np.gradient(processed['dq'],1/spec['rate'],axis=0))
    return np.concatenate(values,axis=1).astype(np.float32)


@functools.lru_cache(maxsize=8)
def load_physics_prior(path):
    import pinocchio as pin
    model=pin.buildModelFromUrdf(str(ROOT.parent/'xarm_ws/gello_teleop/models/xarm7_dynamics.urdf'))
    coefficients=np.load(path)['coefficients']
    return model,coefficients


def physics_prior(processed,spec):
    if not spec.get('physics_prior'):
        return np.zeros_like(processed['tau'])
    import pinocchio as pin
    model,coefficients=load_physics_prior(spec['physics_prior'])
    data=model.createData()
    q,dq=processed['q'].astype(float),processed['dq'].astype(float)
    ddq=np.gradient(dq,1/spec['rate'],axis=0)
    predictions=[]
    for i in range(len(q)):
        reg=pin.computeJointTorqueRegressor(model,data,q[i],dq[i],ddq[i])
        dyn=reg@coefficients[:70]+pin.rnea(model,data,q[i],dq[i],ddq[i])
        friction=np.stack([dq[i],np.tanh(dq[i]/.02),np.tanh(dq[i]/.002),np.ones(7)],axis=1)
        predictions.append(dyn+(friction*coefficients[70:].reshape(7,4)).sum(axis=1))
    return np.asarray(predictions,dtype=np.float32)


def build_cache(blocks, spec, device, normalization=None):
    features, targets, valid, offset = [], [], [], 0
    for block in blocks:
        processed = preprocess(block,spec['filter'],spec['rate'])
        x, y = model_features(processed,spec),processed['tau']-physics_prior(processed,spec)
        # One second on BOTH sides also excludes filtering edge transients.
        margin = spec['rate']
        start = max(margin,spec['horizon']-1)
        stop = len(x)-margin
        valid.append(np.arange(start,stop)+offset)
        features.append(x);targets.append(y);offset+=len(x)
    x, y = np.concatenate(features),np.concatenate(targets)
    indexes = np.concatenate(valid)
    if normalization is None:
        normalization = {'x_mean':x[indexes].mean(0),'x_std':x[indexes].std(0).clip(1e-5),
                         'y_mean':y[indexes].mean(0),'y_std':y[indexes].std(0).clip(1e-3)}
    x = (x-normalization['x_mean'])/normalization['x_std']
    y = (y-normalization['y_mean'])/normalization['y_std']
    return torch.tensor(x,device=device),torch.tensor(y,device=device),torch.tensor(indexes,device=device),normalization


def window_indexes(indexes, spec):
    if spec['arch']=='bilstm':
        offsets = torch.arange(spec['horizon'],device=indexes.device)-spec['horizon']//2
    else:
        offsets = torch.arange(1-spec['horizon'],1,device=indexes.device)
    return indexes[:,None]+offsets[None,:]


@torch.inference_mode()
def eval_cache(model,cache,spec,normalization):
    x,y,indexes,_=cache
    model.eval()
    indexes=indexes[::max(1,spec['rate']//20)]
    predictions=[]
    for batch in indexes.split(512):
        predictions.append(model(x[window_indexes(batch,spec)]).cpu().numpy())
    pred=np.concatenate(predictions)*normalization['y_std']+normalization['y_mean']
    target=y[indexes].cpu().numpy()*normalization['y_std']+normalization['y_mean']
    return metrics(pred,target)


@torch.inference_mode()
def infer(model,episode,spec,normalization,device):
    processed=preprocess(episode,spec['filter'],spec['rate'])
    x=(model_features(processed,spec)-normalization['x_mean'])/normalization['x_std']
    x=torch.tensor(x,device=device)
    model.eval()
    # Endpoint padding is exposed by valid_context; never use it in scored metrics.
    predictions=[]
    for start in range(0,len(x),512):
        batch=torch.arange(start,min(start+512,len(x)),device=device)
        ix=window_indexes(batch,spec).clamp(0,len(x)-1)
        predictions.append(model(x[ix]).cpu().numpy())
    pred=np.concatenate(predictions)*normalization['y_std']+normalization['y_mean']
    pred+=physics_prior(processed,spec)
    pred100=np.stack([np.interp(episode['t'],processed['t'],pred[:,j]) for j in range(7)],axis=1)
    target100=np.stack([np.interp(episode['t'],processed['t'],processed['tau'][:,j]) for j in range(7)],axis=1)
    valid=(episode['t']>=episode['t'][0]+1)&(episode['t']<=episode['t'][-1]-1)
    return pred100,target100,valid


def run_experiments(args,bg,peel):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE',device,torch.__version__,flush=True)
    if not (args.output/'original_checkpoint_metrics.json').exists():
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        original={}
        for episode in bg:
            pred,target=predict_original(episode,checkpoint,device)
            original[episode['name']]=metrics(pred,target)
            print('ORIGINAL',episode['name'],original[episode['name']]['rmse_nm'],flush=True)
        save_json(args.output/'original_checkpoint_metrics.json',original)

    train_record=bg[3]  # right arm, long recording 0001
    test_record=bg[5]   # right arm, independent recording 0002
    block_size=3000
    nblocks=len(train_record['t'])//block_size
    rng=np.random.default_rng(42)
    val_blocks=set(rng.choice(nblocks,size=5,replace=False).tolist())
    train_blocks=[]; validation_blocks=[]; left_train_blocks=[]
    for index in range(nblocks):
        block=select_rows(train_record,index*block_size,(index+1)*block_size)
        (validation_blocks if index in val_blocks else train_blocks).append(block)
        if index not in val_blocks:
            left_train_blocks.append(select_rows(bg[2],index*block_size,(index+1)*block_size))
    split={'train_source':train_record['path'],'test_source':test_record['path'],
           'block_size_rows':block_size,'validation_blocks':sorted(val_blocks),'edge_exclusion_s':1,
           'description':'recording 0001 disjoint 30s blocks, split before filtering; recording 0002 held out from all new training and model selection',
           'archived_checkpoint_caveat':'normalizer statistics and reproduced validation MSE identify original train as recording 1 both arms plus recording 2 left, validation as recording 2 right; original sees simultaneous left recording 2, whereas blocked experiments exclude recording 2 entirely'}
    if args.split=='archived':
        train_blocks=[bg[3]]
        left_train_blocks=[bg[2],bg[4]]
        validation_blocks=[bg[5]]
        split={'train_sources':[e['name'] for e in [bg[2],bg[3],bg[4]]],
               'validation_source':bg[5]['name'],'edge_exclusion_s':1,
               'description':'reconstructed archived checkpoint split; right-only ablation trains right recording 1 only; this is validation for model selection, not an independent test',
               'seed':args.seed}
    save_json(args.output/'split.json',split)
    specs=[
        {'name':'next_mixed_causal10_100','arch':'lstm','filter':'causal10','rate':100,'horizon':50,'mixed':True},
        {'name':'next_right_causal10_100','arch':'lstm','filter':'causal10','rate':100,'horizon':50},
        {'name':'next_right_zero10_100','arch':'lstm','filter':'zero10','rate':100,'horizon':50},
        {'name':'next_right_zero10_50','arch':'lstm','filter':'zero10','rate':50,'horizon':25},
        {'name':'next_right_zero5_50','arch':'lstm','filter':'zero5','rate':50,'horizon':25},
        {'name':'bilstm_right_zero10_50','arch':'bilstm','filter':'zero10','rate':50,'horizon':51},
        {'name':'mlp_ddq_right_zero10_50','arch':'mlp','filter':'zero10','rate':50,'horizon':1,'ddq':True},
        {'name':'bilstm_right_zero10_50_no_delta','arch':'bilstm','filter':'zero10','rate':50,'horizon':51,'no_delta':True},
    ]
    if args.models and any(name.startswith('residual_') for name in args.models):
        for mode in ('zero10','zero5'):
            specs.append({'name':f'residual_bilstm_right_{mode}_50','arch':'bilstm','filter':mode,'rate':50,'horizon':51,
                          'physics_prior':str((args.physics_dir/f'identified_dynamics_right_{mode}.npz').resolve())})
    if args.split=='archived':
        for spec in specs:
            if spec['name']!='next_right_causal10_100':
                spec['name']=spec['name'].replace('_right_','_mixed_')
                spec['mixed']=True
    if args.models:
        specs=[s for s in specs if s['name'] in args.models]
        if not specs:
            raise ValueError('No experiment names matched --models')
    # Same 5 Hz target and same raw 100 Hz rows for cross-preprocessing comparison.
    test_common=sosfiltfilt(butter(4,5,fs=100,output='sos'),test_record['tau'],axis=0)
    results=[]
    for spec in specs:
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        train_cache=build_cache(train_blocks+(left_train_blocks if spec.get('mixed') else []),spec,device)
        normalization=train_cache[-1]
        val_cache=build_cache(validation_blocks,spec,device,normalization)
        model=Regressor(spec).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4,min_lr=1e-5)
        x,y,indexes,_=train_cache
        scale=torch.tensor(normalization['y_std'],device=device)
        best=float('inf');best_epoch=0;best_state=None;history=[];started=time.monotonic()
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            # Fixed sample budget makes 50/100 Hz and mixed/right comparisons comparable.
            sampled=indexes[torch.randint(len(indexes),(args.samples_per_epoch,),device=device)]
            for batch in sampled.split(512):
                prediction=model(x[window_indexes(batch,spec)])
                loss=((prediction-y[batch])*scale).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();losses.append(float(loss.detach()))
            val=eval_cache(model,val_cache,spec,normalization)
            scheduler.step(val['mse_nm2'])
            history.append({'epoch':epoch,'train_mse_nm2':float(np.mean(losses)),'validation_mse_nm2':val['mse_nm2']})
            if val['mse_nm2']<best:
                best=val['mse_nm2'];best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==1 or epoch%5==0:
                print('TRAIN',spec['name'],epoch,'train_mse',round(np.mean(losses),4),'val_mse',round(val['mse_nm2'],4),'seconds',round(time.monotonic()-started),flush=True)
            if epoch-best_epoch>=10:
                break
        model.load_state_dict(best_state)
        val=eval_cache(model,val_cache,spec,normalization)
        pred,target,valid=infer(model,test_record,spec,normalization,device)
        result={'spec':spec,'seed':args.seed,'evaluation_role':'validation' if args.split=='archived' else 'test',
                'best_epoch':best_epoch,'seconds':time.monotonic()-started,'validation':val,
                'test_native_target':metrics(pred[valid],target[valid]),
                'test_common_5hz_target':metrics(pred[valid],test_common[valid]),
                'test_raw_target':metrics(pred[valid],test_record['tau'][valid]),'history':history}
        results.append(result)
        torch.save({'model':best_state,'spec':spec,'normalization':normalization,'split':split,'result':result},args.output/(spec['name']+'.pt'))
        np.savez_compressed(args.output/(spec['name']+'_test.npz'),time_s=test_record['t'],tau_pred=pred,tau_raw=test_record['tau'],tau_target=target,valid_context=valid)
        save_json(args.output/'experiments.json',results)
        print('RESULT',spec['name'],'test_native_rmse',round(result['test_native_target']['rmse_nm'],4),
              'test_common5_rmse',round(result['test_common_5hz_target']['rmse_nm'],4),flush=True)
        del model,train_cache,val_cache,x,y,indexes
    print('EXPERIMENTS_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bg',type=Path,default=ROOT/'data/xarm_bg/bg_data')
    parser.add_argument('--peel',type=Path,default=ROOT.parent/'xarm_ws/runs/peel_cucumber_25hzcam')
    parser.add_argument('--checkpoint',type=Path,default=ROOT.parent/'xarm_ws/model/dp/pretrained_model-20260901T082955Z-1-001/bg/epoch_003_val_tau_mse_nm2_1.386437.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/xarm_tau_offline_20261004')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--split',choices=['blocked','archived'],default='blocked')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--samples-per-epoch',type=int,default=32768)
    parser.add_argument('--models',nargs='+')
    parser.add_argument('--physics-dir',type=Path,default=ROOT/'outputs/xarm_tau_offline_20261004/physics')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    bg=[read_episode(p,arm) for p in sorted(args.bg.glob('*.h5')) for arm in ('left','right')]
    peel=[read_episode(p) for p in sorted(args.peel.glob('*.h5'))]
    audit(bg,peel,args.checkpoint,args.output)
    if not args.audit_only:
        run_experiments(args,bg,peel)


if __name__=='__main__':
    main()
