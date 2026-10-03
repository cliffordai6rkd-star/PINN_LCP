"""Small selected difficult windows and real-feedback replay, without training/robot I/O."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import torch
from data_process.contact_world_model_dataset import ContactWorldModelDataset
from model.pinn_model.checkpoint import model_from_checkpoint
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.contact_world_model_student import ContactWorldModelStudent
from train.nomalizer import Normalizer
from train.trainer.contact_world_model_distill_train import _equal_nested, _data_contract, _sha256
from train.carswm_execution_metrics import adjacent_prediction_metrics, execution_slice, ContactAccumulator


def load(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = model_from_checkpoint(payload)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).float().eval(), payload


@torch.inference_mode()
def run(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    student, payload = load(args.student_checkpoint, args.device)
    teacher, reference = load(args.teacher_checkpoint, args.device)
    if not isinstance(student, ContactWorldModelStudent) or not isinstance(teacher, ContactWorldModel) or isinstance(teacher, ContactWorldModelStudent):
        raise ValueError('requires an actual Student contract and original Teacher')
    if student.checkpoint_contract()['student']['teacher_checkpoint_sha256'] != _sha256(args.teacher_checkpoint):
        raise ValueError('reference is not the original supervising Teacher file')
    if _data_contract(student) != _data_contract(teacher) or not _equal_nested(payload['normalizer'], reference['normalizer']):
        raise ValueError('Teacher/Student data contract or normalizer mismatch')
    config = copy.deepcopy(payload['config'])
    config['dataloader'].setdefault('action_augmentation', {})['enabled'] = False
    n = payload['normalizer']
    normalizer = Normalizer(n['stats'], eps=float(n.get('eps', 1e-6)))
    dataset = ContactWorldModelDataset(config, normalizer=normalizer, compute_normalizer=False)
    mode = config['dataloader'].get('normalize_mode')
    normalized = config['dataloader'].get('normalize_lowdim_keys', [])
    def physical(key, value):
        return getattr(normalizer, f'{mode}_denormalize')(key, value.float()) if mode and key in normalized else value.float()
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn((1, args.num_samples, student.future_horizon, student.flow_dim), generator=generator).to(args.device)
    indices = list(args.sample_indices or [])
    replay = []
    if args.replay_count:
        raw0 = dataset.valid_indices[args.replay_start_index]
        lookup = {raw: i for i, raw in enumerate(dataset.valid_indices)}
        episode = dataset.raw_idx_to_episode[raw0]
        for k in range(args.replay_count):
            raw = raw0 + k * args.replay_stride
            if raw not in lookup or dataset.raw_idx_to_episode[raw] != episode:
                raise ValueError('replay anchors must be valid windows in one episode; choose a shorter replay')
            replay.append(lookup[raw])
        indices.extend(replay)
    if not indices:
        raise ValueError('choose --sample-indices or --replay-count; no automatic full-dataset high-noise evaluation')
    execution_slice(student.external_future_horizon, args.delay_steps, args.execute_steps)
    predictions, batches, rows = {}, {}, []
    for idx in dict.fromkeys(indices):
        raw_batch = dataset[idx]
        batch = {key: value[None].to(args.device) for key, value in raw_batch.items() if torch.is_tensor(value)}
        batches[idx] = batch
        # Public sample() rebuilds condition K/V on EVERY fresh feedback window.
        t = teacher.sample(batch, num_samples=args.num_samples, steps=64, solver='heun', source_noise=noise, cache_condition_kv=True)
        s = student.sample(batch, num_samples=args.num_samples, source_noise=noise, cache_condition_kv=True)
        predictions[idx] = {'teacher': t, 'student': s}
        row = {'dataset_index': idx, 'raw_anchor': int(raw_batch['sample_idx']), 'task_index': int(raw_batch['task_index'])}
        for stream in ('q', 'tau'):
            if stream not in student.predicted_state_streams:
                continue
            tp, sp = physical(stream, t[f'{stream}_pred']), physical(stream, s[f'{stream}_pred'])
            # All Teacher draws are compared with all Student draws. Worst
            # Teacher nearest-neighbour distance diagnoses rare-future omission,
            # without declaring any fixed physical pass threshold.
            distances = torch.cdist(tp.flatten(2), sp.flatten(2)) / (tp.shape[2] * tp.shape[3]) ** 0.5
            nearest = distances.amin(-1)
            row[f'{stream}_teacher_to_student_nearest_rms_mean'] = float(nearest.mean())
            row[f'{stream}_teacher_to_student_nearest_rms_max'] = float(nearest.max())
            row[f'{stream}_same_noise_mse'] = float((tp - sp).square().mean())
            if stream == 'tau':
                for name, values in [('teacher', tp), ('student', sp)]:
                    row[f'{name}_motor_tau_abs_p95_nm'] = float(torch.quantile(values.abs().flatten(), .95))
                    row[f'{name}_motor_tau_abs_peak_nm'] = float(values.abs().max())
        row['contact_same_noise_probability_mse'] = float((t['contact_probability'] - s['contact_probability']).square().mean())
        for name, result in [('teacher', t), ('student', s)]:
            row[f'{name}_contact_probability_peak'] = float(result['contact_probability'][..., -1].max())
            if (config.get('contact_gate') or {}).get('enabled'):
                acc = ContactAccumulator(student.contact_state_count)
                acc.update(result['contact_probability'], batch['contact_future'], batch['future_time'], batch.get('contact'))
                row.update({f'{name}_contact_marginal_{key}': value for key, value in acc.finalize().items()})
        rows.append(row)
    replay_rows = []
    for old_idx, new_idx in zip(replay, replay[1:]):
        old_batch, new_batch = batches[old_idx], batches[new_idx]
        delta = int(new_batch['sample_idx'] - old_batch['sample_idx'])
        overlap = student.external_future_horizon - delta
        if overlap <= 0 or not torch.equal(old_batch['future_timestamp_ns'][:, delta:], new_batch['future_timestamp_ns'][:, :overlap]):
            raise ValueError('replay overlap timestamps do not match; refusing index-only alignment')
        row = {'old_index': old_idx, 'new_index': new_idx, 'anchor_delta': delta,
               'delay_steps_assumption': args.delay_steps}
        for name, model in [('student', student), ('teacher', teacher)]:
            old = physical('q', predictions[old_idx][name]['q_pred'])
            new = physical('q', predictions[new_idx][name]['q_pred'])
            row.update({f'{name}_{key}': float(value.mean()) for key, value in adjacent_prediction_metrics(old, new, delta, args.delay_steps).items()})
            truth = physical('q', new_batch['q_future'])[:, None, :overlap]
            row[f'{name}_before_feedback_label_mse_rad2'] = float((old[:, :, delta:delta + overlap] - truth).square().mean())
            row[f'{name}_after_feedback_label_mse_rad2'] = float((new[:, :, :overlap] - truth).square().mean())
            current = physical('q', new_batch['q'][:, -1])
            row[f'{name}_new_chunk_join_displacement_rad'] = float((new[:, :, 0] - current[:, None]).abs().max())
            if 'dq' in new_batch:
                dq = physical('dq', new_batch['dq'][:, -1])
                first_dt = new_batch['future_time'][:, :1]
                row[f'{name}_new_chunk_join_dq_mse_rad2_s2'] = float(((new[:, :, 0] - current[:, None]) / first_dt[:, None] - dq[:, None]).square().mean())
        replay_rows.append(row)
    return {'student_checkpoint': str(args.student_checkpoint.resolve()),
            'teacher_checkpoint': str(args.teacher_checkpoint.resolve()),
            'student_steps': student.student_steps, 'precision': 'float32',
            'weights_ema': bool((payload.get('ema') or {}).get('enabled')),
            'seed': args.seed, 'num_samples': args.num_samples, 'source_noise': 'same noise across models and replay anchors',
            'task_sources': config.get('train_data', {}).get('sources'),
            'tau_semantics': 'motor total torque; not external torque/contact force',
            'execution_window_assumption': {'delay_steps': args.delay_steps, 'execute_steps': args.execute_steps},
            'windows': rows, 'real_feedback_reconditioning': replay_rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student-checkpoint', type=Path, required=True)
    p.add_argument('--teacher-checkpoint', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sample-indices', type=int, nargs='+')
    p.add_argument('--num-samples', type=int, default=32)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--replay-start-index', type=int, default=0)
    p.add_argument('--replay-count', type=int, default=0)
    p.add_argument('--replay-stride', type=int, default=8)
    p.add_argument('--delay-steps', type=int, default=0)
    p.add_argument('--execute-steps', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.num_samples < 2 or args.replay_count < 0 or args.replay_stride < 1:
        p.error('invalid sampling/replay counts')
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
