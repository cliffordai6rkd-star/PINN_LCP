"""Label a LeRobot v3 task through both WM datasets and export review sidecars.

Uses the exact frozen-teacher preprocessing, cache, timestamps, validity masks,
and phase rule used during WM training. Does not edit source Parquet files.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from data_process.contact_world_model_dataset import ContactWorldModelDataset
from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from data_process.tool.xarm_tau_offline_experiment import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-checkpoints',type=Path,nargs='*',default=[])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cfg=yaml.safe_load(args.config.read_text())
    # This command generates dataset labels/windows, not a WM training run.
    cfg['dataloader']['tau_ext_generation']['device']='cpu'
    cfg['train']['output_dir']=str(args.output/'contact_wm')
    contact=ContactWorldModelDataset(cfg)
    latent_cfg=copy.deepcopy(cfg);latent_cfg['train']['output_dir']=str(args.output/'latent_wm')
    latent=LatentContactWorldModelDataset(latent_cfg)
    torch.testing.assert_close(contact.contact,latent.contact)
    assert contact.tau_label_report['label_contract_sha256']==latent.tau_label_report['label_contract_sha256']
    dataset=contact
    predictions=args.output/'predictions';predictions.mkdir(exist_ok=True)
    figures=args.output/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[];all_phase=[];all_residual=[];all_ts=[];all_episode=[]
    for episode in dataset.episodes:
        a,b=episode['dataset_from_index'],episode['dataset_to_index']
        number=int(episode.get('source_episode_index',episode['episode_index']))
        timestamps=dataset.high_timestamps[a:b].numpy();t=(timestamps-timestamps[0])*1e-9
        phase=dataset.contact[a:b,0].numpy().astype(np.int8)
        residual=dataset.high_tensors['tau_ext'][a:b].numpy()
        score=np.abs(residual).sum(1);valid=dataset.tau_label_valid[a:b].numpy()
        cache=Path(dataset.tau_label_report['episodes'][len(rows)]['cache_file'])
        with np.load(cache,allow_pickle=False) as archive:
            free=archive['tau_free'];measured=archive['tau_measured']
        np.savez_compressed(predictions/f'episode_{number:03d}.npz',timestamp_ns=timestamps,
            tau_free=free,tau_measured=measured,tau_ext=residual,tau_ext_l1=score,
            valid_context=valid,phase=phase)
        changes=np.flatnonzero(np.diff(np.r_[False,phase==2,False]))
        events=[dict(start_s=float(t[l]),end_s=float(t[r-1]+.01)) for l,r in zip(changes[::2],changes[1::2])]
        row=dict(episode=number,frames=b-a,duration_s=float(t[-1]),valid_frames=int(valid.sum()),
            phase_counts={str(k):int(np.sum(phase==k)) for k in (-1,0,1,2)},contact_events=events,
            score_p10_p50_p90=np.percentile(score[valid],[10,50,90]) if valid.any() else [])
        rows.append(row);all_phase.append(phase);all_residual.append(residual);all_ts.append(timestamps);all_episode.append(np.full(len(t),number))
        fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,2,1]})
        for j in range(7): axes[0].plot(t,residual[:,j],lw=.6,label=f'J{j+1}')
        axes[0].set_ylabel('tau_ext [Nm]');axes[0].legend(ncol=7,fontsize=8)
        axes[1].plot(t,score,color='#277da1',lw=.9)
        axes[1].axhline(dataset.contact_gate_config.contact_threshold,color='red',ls='--')
        axes[1].set_ylabel('Residual L1 [Nm]')
        axes[2].step(t,phase,where='post',color='#333')
        axes[2].set_yticks([-1,0,1,2],['invalid','free','alignment','contact'])
        axes[2].set_xlabel('Episode time [s]')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f'Erase board episode {number:03d} | automatic labels, no human ground truth')
        fig.tight_layout();fig.savefig(figures/f'episode_{number:03d}.png',dpi=120);plt.close(fig)
    np.savez_compressed(args.output/'labels_all.npz',timestamp_ns=np.concatenate(all_ts),
        episode_index=np.concatenate(all_episode),tau_ext=np.concatenate(all_residual),phase=np.concatenate(all_phase),
        valid_context=np.concatenate(all_phase)>=0)
    save_json(args.output/'episodes.json',rows)
    artifact=torch.load(cfg['dataloader']['tau_ext_generation']['checkpoint'],map_location='cpu',weights_only=True)
    summary=dict(source=cfg['train_data']['sources'],checkpoint=cfg['dataloader']['tau_ext_generation']['checkpoint'],
        checkpoint_sha256=dataset.tau_label_report['checkpoint_sha256'],teacher_spec=artifact['spec'],wm_rate_hz=100,
        threshold=dataset.contact_gate_config.contact_threshold,phase_counts=dataset.tau_label_report['phase_counts'],
        episodes=len(rows),rows=len(dataset.contact),contact_wm_windows=len(contact),latent_wm_windows=len(latent),
        both_wm_phases_identical=True,latent_cache_hits=latent.tau_label_report['cache_hits'],
        label_contract_sha256=dataset.tau_label_report['label_contract_sha256'],
        raw_data_modified=False,ground_truth='not available; automatic estimates only')
    save_json(args.output/'summary.json',summary)
    comparisons=[]
    for checkpoint in args.compare_checkpoints:
        artifact=torch.load(checkpoint,map_location='cpu',weights_only=True)
        other_cfg=copy.deepcopy(cfg)
        other_cfg['dataloader']['tau_ext_generation']['checkpoint']=str(checkpoint)
        other_cfg['contact_gate']['contact_threshold']=artifact['calibration']['contact_threshold']
        other_cfg['train']['output_dir']=str(args.output/'model_comparison'/checkpoint.stem)
        other=ContactWorldModelDataset(other_cfg)
        phases=other.contact[:,0].numpy().astype(np.int8)
        mask=dataset.tau_label_valid.numpy() & other.tau_label_valid.numpy()
        comparisons.append(dict(model=checkpoint.stem,checkpoint=str(checkpoint),
            contact_threshold=other.contact_gate_config.contact_threshold,
            phase_counts=other.tau_label_report['phase_counts'],
            disagreement_with_selected_fraction=float(np.mean(phases[mask]!=dataset.contact[:,0].numpy()[mask]))))
        np.savez_compressed(args.output/'model_comparison'/f'{checkpoint.stem}.npz',
            phase=phases,tau_ext=other.high_tensors['tau_ext'].numpy(),valid_context=other.tau_label_valid.numpy())
        del other
    if comparisons:
        save_json(args.output/'model_comparison.json',comparisons)
    selected=sorted(set([0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1]))
    table=''.join(f'<tr><td>{r["episode"]}</td><td>{r["duration_s"]:.1f}</td><td>{r["phase_counts"]["2"]}</td><td>{len(r["contact_events"])}</td><td><a href="figures/episode_{r["episode"]:03d}.png">curve</a></td><td><a href="predictions/episode_{r["episode"]:03d}.npz">NPZ</a></td></tr>' for r in rows)
    previews=''.join(f'<img src="figures/episode_{rows[i]["episode"]:03d}.png">' for i in selected)
    (args.output/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>擦板接触阶段标注</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;line-height:1.7;padding:20px}}img{{width:100%}}td,th{{padding:6px 15px;border-bottom:1px solid #ddd}}</style>
<h1>擦板接触阶段标注</h1><p>{len(rows)} 段、{len(dataset.contact)} 帧。阈值 {summary['threshold']:.3f} N·m（七轴残差 L1），接触前 1 秒为 alignment。-1 表示无完整上下文，0=free、1=alignment、2=contact。</p>
<p>Contact WM 与 Latent WM 实际加载后逐帧阶段完全一致，后者复用 {summary['latent_cache_hits']} 个缓存。已按导出元数据还原 dq/tau 的 20 Hz 因果滤波，再执行教师的离线预处理。源 Parquet 未修改。</p>
<p>当前无视频或人工接触真值，以下是自动候选标签，不能据此宣称接触准确率。下载 <a href="labels_all.npz">全部标签</a>，<a href="summary.json">统计与标注契约</a>。</p>{previews}
<table><tr><th>Episode</th><th>秒</th><th>接触帧</th><th>接触片段</th><th>曲线</th><th>数据</th></tr>{table}</table></html>''',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
