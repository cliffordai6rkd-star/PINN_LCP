"""Build/cache frozen xArm torque-teacher labels before either WM is trained."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from model.xarm_tau_free import LearnedTorqueModel, predict_episode

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def restore_export_filters(t, values, feature_filters):
    """Undo the known invertible causal export filter on a private teacher copy.

    WM observations retain their original causal filtering. This avoids applying
    the teacher's 5 Hz offline filter to already lagged 20 Hz dq/torque values.
    Only the recorded one-pole cascade contract is invertible here.
    """
    restored = {key: value.copy() for key, value in values.items()}
    for key, spec in feature_filters.items():
        if not spec.get('enabled', False):
            continue
        if (spec.get('contract') != 'causal_variable_dt_one_pole_cascade_v1'
                or not spec.get('causal') or not spec.get('history_only')):
            raise ValueError(f'Cannot restore source filter for teacher input {key}: {spec}')
        cutoff, order = float(spec['cutoff_hz']), int(spec['order'])
        if not np.isfinite(cutoff) or cutoff <= 0 or order < 1:
            raise ValueError('Invalid source filter cutoff/order')
        alpha = -np.expm1(-2*np.pi*cutoff*np.diff(t))
        if np.any(alpha <= 0):
            raise ValueError('Source timestamps must increase for filter restoration')
        value = restored[key].astype(np.float64)
        for _ in range(order):
            original = value.copy()
            original[1:] = (value[1:]-(1-alpha[:, None])*value[:-1])/alpha[:, None]
            value = original
        restored[key] = value.astype(np.float32)
    return restored


def _source_filters(dataset, source_index):
    if dataset.backend != 'lerobot':
        return {}
    root = Path(dataset.lerobot_source_specs[source_index]['root'])
    path = root/'meta/world_model_timeline.json'
    metadata = json.loads(path.read_text()) if path.is_file() else {}
    filters = metadata.get('feature_filters', {})
    result = {key: filters[dataset.high_keys[key]] for key in ('q', 'dq', 'delta_q', 'tau')
              if dataset.high_keys[key] in filters}
    # A declared prefiltered input without its export contract must not silently
    # receive a second, mismatched filter chain.
    for key, spec in (dataset.data_config.get('filters') or {}).items():
        if key in ('q', 'dq', 'delta_q', 'tau') and (spec.get('source_already_filtered')
                or spec.get('dataset_preprocessed_operations')) and key not in result:
            raise ValueError(f'Torque labels need meta/world_model_timeline.json feature_filters for {key}')
    return result


def _atomic_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem+'.', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate_tau_labels(dataset):
    """Return teacher outputs; source H5/Parquet and WM observations stay intact."""
    cfg = dataset.tau_generation_config
    if dataset.high_fps != 100:
        raise ValueError('The xArm tau teacher requires the original 100 Hz state timeline')
    checkpoint = Path(cfg['checkpoint']).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Torque-label checkpoint does not exist: {checkpoint}')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    code_sha = hashlib.sha256((ROOT/'model/xarm_tau_free.py').read_bytes()+(ROOT/'model/xarm_tau_sequence.py').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    cache = Path(cfg.get('cache_dir', ROOT/'outputs/cache/wm_tau_labels')).expanduser()
    count = len(dataset.high_timestamps)
    for key in ('q', 'dq', 'delta_q', 'tau'):
        if dataset.high_tensors[key].shape != (count, 7):
            raise ValueError(f'xArm torque-label input {key} must have shape [N,7]')
    result = {key: torch.full((count, 7), float('nan')) for key in ('tau_free', 'tau_measured', 'tau_ext')}
    result['valid_context'] = torch.zeros(count, dtype=torch.bool)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                  pipeline_sha256=code_sha, cache_dir=str(cache.resolve()),
                  teacher_inputs=['q', 'dq', 'delta_q'], filter='defined by exported checkpoint spec/contract',
                  episodes=[], cache_hits=0, cache_misses=0)
    teacher = None
    old_threads = torch.get_num_threads()
    threads = int(cfg.get('threads', 2))
    if threads < 1:
        raise ValueError('tau_ext_generation.threads must be positive')
    try:
        torch.set_num_threads(threads)
        for episode in tqdm(dataset.episodes, desc='prepare WM tau_ext labels', unit='episode'):
            start, end = int(episode['dataset_from_index']), int(episode['dataset_to_index'])
            timestamps = dataset.high_timestamps[start:end].cpu().numpy()
            t = (timestamps-timestamps[0])*1e-9
            values = {key: dataset.high_tensors[key][start:end].cpu().numpy()
                      for key in ('q', 'dq', 'delta_q', 'tau')}
            source_index = int(episode.get('source_index', 0))
            filters = _source_filters(dataset, source_index)
            digest = hashlib.sha256(json.dumps(dict(checkpoint=checkpoint_sha, code=code_sha,
                                                     filters=filters), sort_keys=True).encode())
            for value in (timestamps, *values.values()):
                digest.update(str((value.shape, str(value.dtype))).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            signature = digest.hexdigest()
            path = cache/(signature+'.npz')
            prediction = None
            if path.is_file() and not cfg.get('force_rebuild', False):
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        if str(saved['signature']) != signature:
                            raise ValueError('cache signature mismatch')
                        prediction = {key: saved[key].copy() for key in result}
                    for key, value in prediction.items():
                        shape = (end-start,) if key == 'valid_context' else (end-start, 7)
                        if value.shape != shape:
                            raise ValueError('cache shape mismatch')
                    if prediction['valid_context'].dtype != np.bool_:
                        raise ValueError('cache validity must be boolean')
                    if any(not np.isfinite(prediction[k][prediction['valid_context']]).all()
                           for k in ('tau_free', 'tau_measured', 'tau_ext')):
                        raise ValueError('cache contains invalid labeled torques')
                except (ValueError, KeyError, OSError, EOFError):
                    log.warning('Rebuilding invalid torque cache %s', path)
                    prediction = None
            if prediction is None:
                if teacher is None:
                    teacher = LearnedTorqueModel(checkpoint, device=cfg.get('device', 'cpu'))
                inputs = restore_export_filters(t, values, filters)
                valid = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in inputs.values()])
                prediction = predict_episode(teacher, dict(t=t, valid=valid, **inputs,
                                             q_cmd=inputs['q']+inputs['delta_q']))
                _atomic_npz(path, dict(signature=np.asarray(signature), timestamp_ns=timestamps,
                                      **{key: prediction[key] for key in result}))
                report['cache_misses'] += 1
            else:
                report['cache_hits'] += 1
            for key in result:
                result[key][start:end] = torch.as_tensor(prediction[key], dtype=result[key].dtype)
            report['episodes'].append(dict(source_index=source_index,
                episode_index=int(episode.get('source_episode_index', episode.get('episode_index', 0))),
                start=start, end=end, valid_rows=int(prediction['valid_context'].sum()),
                restored_source_filters=list(filters), signature=signature, cache_file=str(path.resolve())))
    finally:
        del teacher
        torch.set_num_threads(old_threads)
    report['valid_rows'] = int(result['valid_context'].sum())
    log.info('WM tau labels ready: %d valid / %d rows, %d cache hits / %d generated',
             report['valid_rows'], count, report['cache_hits'], report['cache_misses'])
    return result, report


def write_label_report(dataset):
    """Persist the label contract and frame counts beside the training run."""
    if not getattr(dataset, 'tau_label_report', None):
        return
    report = dataset.tau_label_report
    contact = dataset.contact[:, 0]
    report['phase_rule'] = dict(norm=dataset.contact_gate_config.metric,
                               contact_threshold=dataset.contact_gate_config.contact_threshold,
                               comparison='>', precontact_duration_s=dataset.contact_gate_config.precontact_duration_s,
                               classes={'-1': 'invalid_context', '0': 'free', '1': 'alignment', '2': 'contact'})
    report['phase_counts'] = {str(i): int((contact == i).sum()) for i in (-1, 0, 1, 2)}
    for episode in report['episodes']:
        phase = contact[episode['start']:episode['end']]
        episode['phase_counts'] = {str(i): int((phase == i).sum()) for i in (-1, 0, 1, 2)}
    # Resume can detect changes in the checkpoint, input data, and phase rule.
    report['label_contract_sha256'] = hashlib.sha256(json.dumps(
        dict(phase_rule=report['phase_rule'], episodes=[e['signature'] for e in report['episodes']]),
        sort_keys=True).encode()).hexdigest()
    output = (dataset.config.get('train') or {}).get('output_dir')
    if output:
        path = Path(output)/'tau_label_report.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='tau_labels.', suffix='.json', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(report, stream, indent=2)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
