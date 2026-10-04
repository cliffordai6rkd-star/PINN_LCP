import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location('v21_converter', Path(__file__).resolve().parents[1] / 'data_process/tool/lerobotv3_to_v21.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    features = {k: {'dtype': 'float32', 'shape': [7]} for k in c.ALIASES.values()}
    features.update({k: {'dtype': 'int64', 'shape': [1]} for k in ['index', 'frame_index', 'episode_index', 'task_index']})
    features['timestamp'] = {'dtype': 'float32', 'shape': [1]}
    features['observation.images.wrist'] = {'dtype': 'video', 'shape': [16, 16, 3], 'info': {'video.pix_fmt': 'yuv420p', 'video.codec': 'h264'}}
    c.write_json(root / 'meta/info.json', {
        'codebase_version': 'v3.0', 'fps': 5, 'total_episodes': 2, 'total_frames': 10,
        'total_tasks': 1, 'features': features, 'splits': {'train': '0:2'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
    })
    stats = {k: {'min': [0], 'max': [1], 'mean': [0.5], 'std': [0.5], 'count': [5]} for k in features}
    for key, feature in features.items():
        for name in ('min', 'max', 'mean', 'std'):
            value = stats[key][name][0]
            stats[key][name] = [[[value]] for _ in range(3)] if feature['dtype'] == 'video' else [value] * feature['shape'][0]
    c.write_json(root / 'meta/stats.json', stats)
    pq.write_table(pa.table({'task_index': [0], 'task': ['test']}), root / 'meta/tasks.parquet')
    episodes = []
    for i in range(2):
        e = {'episode_index': i, 'length': 5, 'tasks': ['test'], 'dataset_from_index': i*5, 'dataset_to_index': i*5+5,
             'data/chunk_index': 0, 'data/file_index': 0}
        for k, v in {'chunk_index': 0, 'file_index': 0, 'from_timestamp': float(i), 'to_timestamp': float(i+1)}.items():
            e[f'videos/observation.images.wrist/{k}'] = v
        for key, values in stats.items():
            for name, value in values.items():
                e[f'stats/{key}/{name}'] = value
        episodes.append(e)
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-000/file-000.parquet')
    rows = {'index': range(10), 'frame_index': list(range(5))*2, 'episode_index': [0]*5+[1]*5,
            'task_index': [0]*10, 'timestamp': list(np.arange(5)/5)*2}
    rows.update({k: [[float(i)]*7 for i in range(10)] for k in c.ALIASES.values()})
    (root / 'data/chunk-000').mkdir(parents=True)
    table = pa.table(rows)
    hf = {'info': {'features': {
        key: ({'_type': 'List', 'length': 7, 'feature': {'_type': 'Value', 'dtype': 'float64'}}
              if key in c.ALIASES.values() else {'_type': 'Value', 'dtype': str(table[key].type).replace('double', 'float64')})
        for key in rows
    }}}
    table = table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()})
    pq.write_table(table, root / 'data/chunk-000/file-000.parquet')
    video = root / 'videos/observation.images.wrist/chunk-000/file-000.mp4'
    video.parent.mkdir(parents=True)
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg unavailable')
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=16x16:r=5:d=1',
                    '-f', 'lavfi', '-i', 'color=c=blue:s=16x16:r=5:d=1',
                    '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0', '-r', '5', '-c:v', 'libx264', '-crf', '0', str(video)], check=True)
    return root


def decode(path):
    return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                          check=True, capture_output=True).stdout


def test_conversion_aliases_and_video_boundaries(source, tmp_path):
    output = tmp_path / 'output'
    c.convert(source, output)
    info = c.read_json(output / 'meta/info.json')
    assert info['codebase_version'] == 'v2.1'
    assert info['total_videos'] == 2
    original = pq.read_table(source / 'data/chunk-000/file-000.parquet')
    original_video = decode(source / 'videos/observation.images.wrist/chunk-000/file-000.mp4')
    for i in range(2):
        table = pq.read_table(output / c.DATA_PATH.format(episode_chunk=0, episode_index=i))
        assert table.select(original.column_names).equals(original.slice(i*5, 5))
        for alias, key in c.ALIASES.items():
            assert table[alias].equals(table[key])
        video = output / c.VIDEO_PATH.format(episode_chunk=0, episode_index=i, video_key='observation.images.wrist')
        assert decode(video) == original_video[i*5*16*16*3:(i+1)*5*16*16*3]
    es = json.loads((output / 'meta/episodes_stats.jsonl').read_text().splitlines()[1])
    assert es['stats']['action'] == es['stats']['action.ee_pose']
    with pytest.raises(FileExistsError):
        c.convert(source, output)


def test_invalid_rows_never_publish_partial_dataset(source, tmp_path):
    path = source / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).slice(0, 9)
    pq.write_table(table, path)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='missing rows'):
        c.convert(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.output.partial-*'))


def test_huggingface_alias_metadata():
    table = pa.table({k: [[1., 2.]] for k in c.ALIASES.values()})
    hf = {'info': {'features': {k: {'_type': 'List', 'feature': {'_type': 'Value', 'dtype': 'float32'}, 'length': 2} for k in c.ALIASES.values()}}}
    table = table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()})
    result = c.add_aliases(table)
    features = json.loads(result.schema.metadata[b'huggingface'])['info']['features']
    assert features['action'] == features['action.ee_pose']

    assert features['action']['_type'] == 'Sequence'


def rename_features(root, mapping):
    """Change source schema, including v3 statistics and HF parquet metadata."""
    info = c.read_json(root / 'meta/info.json')
    info['features'] = {mapping.get(k, k): v for k, v in info['features'].items()}
    c.write_json(root / 'meta/info.json', info)
    stats = c.read_json(root / 'meta/stats.json')
    c.write_json(root / 'meta/stats.json', {mapping.get(k, k): v for k, v in stats.items()})
    for path in (root / 'meta/episodes').rglob('*.parquet'):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            for old, new in mapping.items():
                for stat in ('min', 'max', 'mean', 'std', 'count'):
                    row[f'stats/{new}/{stat}'] = row.pop(f'stats/{old}/{stat}')
        pq.write_table(pa.Table.from_pylist(rows), path)
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    hf = json.loads(table.schema.metadata[b'huggingface'])
    hf['info']['features'] = {mapping.get(k, k): v for k, v in hf['info']['features'].items()}
    table = table.rename_columns([mapping.get(k, k) for k in table.column_names])
    table = table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()})
    pq.write_table(table, path)


@pytest.mark.parametrize('state_key,action_key', [
    ('observation.joint', 'action.joint'),
    ('observation.state', 'action'),
    ('observation.eepose', 'action.eepose'),
])
def test_convert_other_schemas(source, tmp_path, state_key, action_key):
    rename_features(source, {'observation.ee_pose': state_key, 'action.ee_pose': action_key})
    output = tmp_path / 'output'
    c.convert(source, output)
    table = pq.read_table(output / c.DATA_PATH.format(episode_chunk=0, episode_index=0))
    assert table['observation.state'].equals(table[state_key])
    assert table['action'].equals(table[action_key])
    stats = c.read_json(output / 'meta/stats.json')
    assert stats['action'] == stats[action_key]
    hf = json.loads(table.schema.metadata[b'huggingface'])['info']['features']
    assert hf['action']['_type'] == 'Sequence'


def test_explicit_action_source_and_no_videos(source, tmp_path):
    rename_features(source, {'observation.ee_pose': 'observation.joint'})
    info = c.read_json(source / 'meta/info.json')
    info['features'].pop('observation.images.wrist')
    info['features']['action.joint'] = {'dtype': 'float32', 'shape': [7]}
    c.write_json(source / 'meta/info.json', info)
    stats = c.read_json(source / 'meta/stats.json')
    stats['action.joint'] = stats['action.ee_pose']
    c.write_json(source / 'meta/stats.json', stats)
    for path in (source / 'meta/episodes').rglob('*.parquet'):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            for name in ('min', 'max', 'mean', 'std', 'count'):
                row[f'stats/action.joint/{name}'] = row[f'stats/action.ee_pose/{name}']
        pq.write_table(pa.Table.from_pylist(rows), path)
    path = source / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).append_column('action.joint', pa.array([[99.] * 7] * 10))
    hf = json.loads(table.schema.metadata[b'huggingface'])
    hf['info']['features']['action.joint'] = hf['info']['features']['action.ee_pose']
    pq.write_table(table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()}), path)

    output = tmp_path / 'output'
    c.convert(source, output, state_key='observation.joint', action_key='action.ee_pose')
    converted = pq.read_table(output / c.DATA_PATH.format(episode_chunk=0, episode_index=0))
    assert converted['action'].equals(converted['action.ee_pose'])
    assert not converted['action'].equals(converted['action.joint'])
    assert c.read_json(output / 'meta/info.json')['video_path'] is None
    assert not (output / 'videos').exists()


def test_alias_validation():
    features = {
        'observation.joint': {'dtype': 'float32', 'shape': [7]},
        'action.joint': {'dtype': 'float32', 'shape': [7]},
        'action.eepose': {'dtype': 'float32', 'shape': [7]},
    }
    assert c.resolve_aliases(features)['action'] == 'action.joint'
    assert c.resolve_aliases(features, action_key='action.eepose')['action'] == 'action.eepose'
    with pytest.raises(ValueError, match='Cannot resolve action'):
        c.resolve_aliases(features, action_key='missing')
    features['action'] = features['action.joint']
    with pytest.raises(ValueError, match='Refusing to replace'):
        c.resolve_aliases(features, action_key='action.joint')
    features['action']['shape'] = [10, 7]
    with pytest.raises(ValueError, match='per-frame vector'):
        c.resolve_aliases(features)


def test_cli_single_destination_and_batch_layout(source, tmp_path, capsys):
    single = tmp_path / 'single'
    c.main(['--input-dir', str(source), '--output-dir', str(single), '--dry-run'])
    assert f'{source} -> {single} (' in capsys.readouterr().out
    assert not single.exists()
    c.main(['--input-dir', str(source), '--output-dir', str(single)])
    assert (single / 'meta/info.json').exists()

    batch = tmp_path / 'batch'
    (batch / 'robot').mkdir(parents=True, exist_ok=True)
    source.rename(batch / 'robot' / 'task')
    output = tmp_path / 'converted'
    c.main(['--input-dir', str(batch), '--output-dir', str(output)])
    assert (output / 'robot/task/meta/info.json').exists()


def add_missing_auxiliary_signal(source):
    key = 'observation.tau_ext'
    info = c.read_json(source / 'meta/info.json')
    info['features'][key] = {'dtype': 'float32', 'shape': [1]}
    c.write_json(source / 'meta/info.json', info)
    stats = {'min': [float('nan')], 'max': [float('nan')], 'mean': [float('nan')],
             'std': [float('nan')], 'count': [5]}
    global_stats = c.read_json(source / 'meta/stats.json')
    global_stats[key] = stats
    c.write_json(source / 'meta/stats.json', global_stats, allow_nan=True)
    for path in (source / 'meta/episodes').rglob('*.parquet'):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row.update({f'stats/{key}/{name}': value for name, value in stats.items()})
        pq.write_table(pa.Table.from_pylist(rows), path)
    path = source / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).append_column(key, pa.array([float('nan')] * 10))
    hf = json.loads(table.schema.metadata[b'huggingface'])
    hf['info']['features'][key] = {'_type': 'Value', 'dtype': 'float64'}
    pq.write_table(table.replace_schema_metadata({b'huggingface': json.dumps(hf).encode()}), path)


def test_auxiliary_nan_is_preserved(source, tmp_path):
    add_missing_auxiliary_signal(source)
    output = tmp_path / 'output'
    c.convert(source, output)
    assert np.isnan(c.read_json(output / 'meta/stats.json')['observation.tau_ext']['mean']).all()
    episode_stats = json.loads((output / 'meta/episodes_stats.jsonl').read_text().splitlines()[0])
    assert np.isnan(episode_stats['stats']['observation.tau_ext']['std']).all()
    table = pq.read_table(output / c.DATA_PATH.format(episode_chunk=0, episode_index=0))
    assert np.isnan(table['observation.tau_ext'].to_numpy()).all()


def test_nonfinite_action_fails_before_video_conversion(source, tmp_path):
    stats = c.read_json(source / 'meta/stats.json')
    stats['action.ee_pose']['mean'][0] = float('nan')
    c.write_json(source / 'meta/stats.json', stats, allow_nan=True)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='Non-finite state/action statistics'):
        c.convert(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.output.partial-*'))


def test_openpi_pinned_lerobot_reader(source, tmp_path):
    legacy = pytest.importorskip('lerobot.common.datasets.lerobot_dataset')
    rename_features(source, {'observation.ee_pose': 'observation.joint', 'action.ee_pose': 'action.joint'})
    add_missing_auxiliary_signal(source)
    output = tmp_path / 'output'
    c.convert(source, output, state_key='observation.joint', action_key='action.joint')
    dataset = legacy.LeRobotDataset(
        repo_id='test-local-v21', root=output,
        delta_timestamps={'action': [i / 5 for i in range(3)]}, video_backend='pyav',
    )
    assert len(dataset) == 10
    for index in (0, 4, 5, 9):
        sample = dataset[index]
        assert sample['action'].shape == (3, 7)
        assert sample['observation.state'].shape == (7,)
        assert sample['observation.images.wrist'].shape == (3, 16, 16)
        assert sample['task'] == 'test'
        # Action windows pad at episode ends instead of crossing into the next one.
        end = 4 if index < 5 else 9
        np.testing.assert_array_equal(sample['action'][:, 0].numpy(), np.minimum(np.arange(index, index + 3), end))
