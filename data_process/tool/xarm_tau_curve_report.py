#!/usr/bin/env python3
"""Export 100 Hz torque residual sidecars and camera-aligned phase review plots.

Labels are provisional: a/b come from background validation residual quantiles,
not contact ground truth. Raw H5 recordings are only opened read-only.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,sosfiltfilt
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from data_process.tool.xarm_tau_offline_experiment import (
    Regressor,infer,read_episode,save_json,predict_original,preprocess,physics_prior,metrics,
)


def phase_labels(t,score,low,high,precontact_s=1.,on_s=.08,off_s=.12):
    """Band labels plus debounced hysteresis + pre-onset labels; contact wins."""
    t,score=np.asarray(t),np.asarray(score)
    if t.ndim!=1 or score.shape!=t.shape or not len(t) or not np.isfinite(score).all():
        raise ValueError('Expected aligned finite one-dimensional time and score')
    if np.any(np.diff(t)<=0) or not (0<=low<high):
        raise ValueError('Increasing timestamps and 0 <= low < high required')
    band=np.where(score<low,0,np.where(score>high,2,1)).astype(np.int8)
    contact=np.zeros(len(t),dtype=bool)
    active=False;pending=None
    for i,s in enumerate(score):
        threshold_met=(s<=low) if active else (s>=high)
        if threshold_met:
            if pending is None:pending=i
            required=off_s if active else on_s
            if t[i]-t[pending]>=required-1e-9:
                active=not active
                # Offline confirmation can backdate the transition to its onset.
                contact[pending:i+1]=active
                pending=None
        else:pending=None
        contact[i]=active
    phases=np.where(contact,2,0).astype(np.int8)
    onsets=np.flatnonzero(contact & ~np.r_[False,contact[:-1]])
    for index in onsets:
        start=np.searchsorted(t,t[index]-precontact_s)
        phases[start:index][~contact[start:index]]=1
    return band,phases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--experiment-root',type=Path,default=ROOT/'outputs/xarm_tau_offline_20261004')
    parser.add_argument('--peel',type=Path,default=ROOT.parent/'xarm_ws/runs/peel_cucumber_25hzcam')
    parser.add_argument('--bg',type=Path,default=ROOT/'data/xarm_bg/bg_data')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--low-threshold',type=float,help='Override background P95 threshold a, in summed Nm')
    parser.add_argument('--high-threshold',type=float,help='Override background P99 threshold b, in summed Nm')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for directory in ('figures','predictions'): (args.output/directory).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    spec,norm=cp['spec'],cp['normalization']
    model=Regressor(spec).to(device);model.load_state_dict(cp['model']);model.eval()
    bgfiles=sorted(args.bg.glob('*.h5'))
    validation=read_episode(bgfiles[2])
    pred,target,valid=infer(model,validation,spec,norm,device)
    score=np.abs(target-pred).sum(axis=1)
    low,high=np.percentile(score[valid],[95,99])
    if args.low_threshold is not None:low=args.low_threshold
    if args.high_threshold is not None:high=args.high_threshold
    if not 0<=low<high:raise ValueError('Require 0 <= low threshold < high threshold')
    threshold_source='background validation P95/P99' if args.low_threshold is None and args.high_threshold is None else 'user overrides (remaining defaults: background validation P95/P99)'
    calibration={'checkpoint':str(args.checkpoint.resolve()),'spec':spec,
        'score':'sum(abs(tau_matched_filter - tau_free_prediction)), seven joints, Nm',
        'low_a':float(low),'high_b':float(high),'threshold_source':threshold_source,'quantiles':[.95,.99],
        'purpose':'provisional visual review only; thresholds fitted to background validation, not contact truth',
        'validation':metrics(pred[valid],target[valid]),'on_confirmation_s':.08,'off_confirmation_s':.12,'precontact_s':1.,
        'phase_band':{'0':'free_motion','1':'intermediate_torque (user alignment heuristic)','2':'contact_candidate','-1':'insufficient_context'},
        'phase_first':{'0':'free_motion','1':'pre_contact (one second before onset)','2':'contact_candidate','-1':'insufficient_context'}}
    save_json(args.output/'calibration.json',calibration)
    # Archive comparison uses its own filters and then the same 5Hz target for fairness.
    old_path=ROOT.parent/'xarm_ws/model/dp/pretrained_model-20260901T082955Z-1-001/bg/epoch_003_val_tau_mse_nm2_1.386437.pt'
    old_cp=torch.load(old_path,map_location='cpu',weights_only=False)
    oldpred,oldtarget=predict_original(validation,old_cp,device)
    common=sosfiltfilt(butter(4,5,fs=100,output='sos'),validation['tau'],axis=0)
    mask=valid[49:]
    calibration['old_common5']=metrics(oldpred[mask],common[49:][mask])
    calibration['new_common5']=metrics(pred[valid],common[valid])
    save_json(args.output/'calibration.json',calibration)
    fig,axes=plt.subplots(4,1,figsize=(15,10),sharex=True)
    for ax,j in zip(axes[:3],[1,3,6]):
        ax.plot(validation['t'],target[:,j],label='Measured, matched filter',lw=.8,color='.4')
        ax.plot(validation['t'],pred[:,j],label='New tau_free',lw=.8)
        ax.plot(validation['t'][49:],oldpred[:,j],label='Archived NEXT',lw=.7,alpha=.65)
        ax.set_ylabel(f'Joint {j+1} [Nm]');ax.grid(alpha=.2)
    axes[0].legend(ncol=3)
    axes[3].plot(validation['t'][49:],np.abs(oldtarget-oldpred).sum(1),label='Archived residual L1',alpha=.6,lw=.7)
    axes[3].plot(validation['t'],score,label='New residual L1',lw=.8)
    axes[3].axhline(low,color='orange',ls='--',label=f'a={low:.2f}')
    axes[3].axhline(high,color='red',ls='--',label=f'b={high:.2f}')
    axes[3].set_ylabel('Residual L1 [Nm]');axes[3].set_xlabel('Time [s]');axes[3].legend(ncol=4)
    fig.suptitle('Background validation: contact-free recording 0002, right arm')
    fig.tight_layout();fig.savefig(args.output/'figures/background_validation.png',dpi=150);plt.close(fig)
    bgq=np.concatenate([read_episode(p)['q'] for p in bgfiles[1:]])
    bounds=(bgq.min(0),bgq.max(0))
    summaries=[];q_target=[]
    for number,path in enumerate(sorted(args.peel.glob('*.h5'))):
        episode=read_episode(path)
        pred,target,valid=infer(model,episode,spec,norm,device)
        residual=target-pred;score=np.abs(residual).sum(1)
        band,phase=phase_labels(episode['t'],score,low,high)
        band[~valid]=-1;phase[~valid]=-1
        outside=np.any((episode['q']<bounds[0])|(episode['q']>bounds[1]),axis=1)
        q_target.append(episode['q'])
        payload={'timestamp_us':episode['timestamp_us'],'time_s':episode['t'],
                 'tau_free':pred.astype(np.float32),'tau_measured_raw':episode['tau'].astype(np.float32),
                 'tau_measured_matched':target.astype(np.float32),'tau_ext_estimate':residual.astype(np.float32),
                 'tau_ext_raw_minus_prediction':(episode['tau']-pred).astype(np.float32),
                 'tau_ext_l1':score.astype(np.float32),'tau_ext_l2':np.linalg.norm(residual,axis=1).astype(np.float32),
                 'phase_band':band,'phase_first':phase,'valid_context':valid,'outside_bg_q_bounds':outside}
        for camera,timestamps in episode['cameras'].items():
            ix=np.searchsorted(episode['timestamp_us'],timestamps,side='right')-1
            in_range=(ix>=0)&(timestamps<=episode['timestamp_us'][-1])
            ix=ix.clip(0,len(score)-1)
            camera_phase=phase[ix].copy();camera_phase[~in_range]=-1
            payload[f'{camera}_timestamp_us']=timestamps
            payload[f'{camera}_lowdim_index']=ix
            payload[f'{camera}_phase_first']=camera_phase
        np.savez_compressed(args.output/'predictions'/f'{path.stem}.npz',**payload)
        fractions={str(label):float(np.mean(phase[valid]==label)) for label in (0,1,2)}
        summaries.append({'episode':number,'file':path.name,'duration_s':episode['t'][-1],
            'score_p10_p50_p90':np.percentile(score[valid],[10,50,90]),'phase_first_fraction':fractions,
            'outside_bg_q_bounds_fraction':float(outside.mean())})
        fig,axes=plt.subplots(4,1,figsize=(13,9),sharex=True,gridspec_kw={'height_ratios':[2,2,2,1]})
        for j in range(7):
            axes[0].plot(episode['t'],residual[:,j],lw=.7,label=f'J{j+1}')
        axes[0].set_ylabel('Residual [Nm]');axes[0].legend(ncol=7,fontsize=8)
        axes[1].plot(episode['t'],score,label='Matched residual L1',lw=1)
        axes[1].axhline(low,color='orange',ls='--',label=f'a={low:.2f}')
        axes[1].axhline(high,color='red',ls='--',label=f'b={high:.2f}')
        axes[1].set_ylabel('L1 [Nm]');axes[1].legend(ncol=3,fontsize=8)
        axes[2].plot(episode['t'],np.linalg.norm(episode['dq'],axis=1),color='purple')
        axes[2].set_ylabel('Speed L2 [rad/s]')
        axes[3].step(episode['t'],phase,where='post',label='FIRST-style candidate')
        axes[3].step(episode['t'],band,where='post',alpha=.5,label='a/b band candidate')
        axes[3].set_yticks([-1,0,1,2],['edge','free','pre / middle','contact']);axes[3].set_xlabel('100 Hz lowdim time [s]')
        axes[3].legend(fontsize=8,loc='upper right')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f'Episode {number:02d} | {spec["name"]}\nProvisional thresholds; labels require video review')
        fig.tight_layout();fig.savefig(args.output/f'figures/episode_{number:02d}.png',dpi=130);plt.close(fig)
        if number in (0,10,25,47):
            with h5py.File(path,'r') as f:
                frames=f['cameras/right_wrist/frames']
                cam_t=(episode['cameras']['right_wrist']-episode['timestamp_us'][0])*1e-6
                indices=np.linspace(0,len(frames)-1,8,dtype=int)
                fig,axes=plt.subplots(2,4,figsize=(13,7))
                for ax,index in zip(axes.flat,indices):
                    ix=np.searchsorted(episode['t'],cam_t[index]).clip(0,len(score)-1)
                    ax.imshow(frames[index]);ax.axis('off');ax.set_title(f'{cam_t[index]:.2f}s | L1={score[ix]:.2f}',fontsize=10)
                fig.suptitle(f'Episode {number:02d}: camera evidence for manual phase review')
                fig.tight_layout();fig.savefig(args.output/f'figures/episode_{number:02d}_camera.png',dpi=130);plt.close(fig)
        print('CURVE',number,'L1 median',round(float(np.median(score[valid])),3),'candidate contact fraction',round(fractions['2'],3),flush=True)
    save_json(args.output/'episode_summary.json',summaries)
    q_target=np.concatenate(q_target)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,j in zip(axes,[2,6]):
        ax.hist(bgq[:,j],bins=80,density=True,alpha=.6,label='Background right arm')
        ax.hist(q_target[:,j],bins=80,density=True,alpha=.6,label='Peeling right arm')
        ax.set_xlabel(f'Joint {j+1} position [rad]');ax.set_ylabel('Density');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'figures/coverage.png',dpi=150);plt.close(fig)
    rows=''.join(f'<tr><td>{s["episode"]:02d}</td><td>{s["duration_s"]:.1f}</td><td>{s["score_p10_p50_p90"][1]:.2f}</td><td><a href="figures/episode_{s["episode"]:02d}.png">曲线</a></td><td><a href="predictions/{Path(s["file"]).stem}.npz">100 Hz 数值</a></td></tr>' for s in summaries)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>xArm 接触阶段曲线检查</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;line-height:1.6}}img{{max-width:100%}}td,th{{padding:6px 20px;text-align:left;border-bottom:1px solid #ddd}}code{{background:#eee}}</style>
<h1>xArm 自由空间力矩与接触阶段曲线检查</h1>
<p>模型：<code>{html.escape(spec['name'])}</code>。48 段低维数据按原始 100 Hz 时间戳输出，25 Hz 相机用时间戳关联。</p>
<p>候选阈值 a={low:.3f}、b={high:.3f} Nm，来源：{threshold_source}。它们只是自动检查的起点，不是已验证的接触阈值。原始 H5 未修改。</p>
<p>phase_band 按你的两阈值区间划分；phase_first 用滞回接触判断，再将接触前 1 秒标为 pre-contact。-1 表示缺少完整上下文。所有削黄瓜数据的关节 7 均超出背景数据范围；outside_bg_q_bounds 字段记录此项。</p>
<p>背景验证 RMSE：{calibration['validation']['rmse_nm']:.4f} Nm。统一 5 Hz 目标：原权重 {calibration['old_common5']['rmse_nm']:.4f}，当前模型 {calibration['new_common5']['rmse_nm']:.4f} Nm。这些是用于模型选择的验证结果，非独立测试。</p>
<h2>背景验证</h2><img src="figures/background_validation.png"><h2>数据覆盖</h2><img src="figures/coverage.png">
<h2>削黄瓜曲线与相机</h2>'''
    for index in (0,10,25,47):
        page+=f'<h3>Episode {index:02d}</h3><img src="figures/episode_{index:02d}.png"><img src="figures/episode_{index:02d}_camera.png">'
    page+=f'<h2>全部 48 段</h2><table><tr><th>Episode</th><th>秒</th><th>L1 中位数</th><th>曲线</th><th>输出</th></tr>{rows}</table></html>'
    (args.output/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
