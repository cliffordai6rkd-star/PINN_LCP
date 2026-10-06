"""Recompute Nero contact prefixes from saved torque residuals without teacher inference."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats
from lerobot.datasets.utils import serialize_dict

from model.pinn_model.contact_gate import ContactGateConfig, contact_phase_labels_from_signal
from scripts.prepare_nero_wm_data import TASKS, replace_datasets

ROOT = Path(__file__).resolve().parents[1]
PHASE = "observation.contact_phase"


def relabel_task(directory, threshold, final_directory):
    info_path = directory/"meta/info.json"
    info = json.loads(info_path.read_text())
    report_path = directory/"meta/torque_label_report.json"
    report = json.loads(report_path.read_text())
    paths = sorted((directory/"data").rglob("*.parquet"))
    tables = [pq.read_table(path) for path in paths]
    table = pa.concat_tables(tables)
    times = np.asarray(table["timing.state_timestamp_ns"].to_pylist(), dtype=np.int64).reshape(-1)
    external = np.asarray(table["observation.tau_ext"].to_pylist(), dtype=np.float32)
    validity = np.asarray(table["observation.tau_label_valid"].to_pylist(), dtype=np.uint8).reshape(-1).astype(bool)
    episode_ids = np.asarray(table["episode_index"].to_pylist()).reshape(-1)
    bounds = np.r_[0, np.flatnonzero(np.diff(episode_ids))+1, len(times)]
    gate = ContactGateConfig(enabled=True, label_mode="three_phase", metric="tau_ext_l1",
                             contact_threshold=float(threshold), precontact_duration_s=1., state_rate_hz=100.)
    gate.validate()
    phases = np.full((len(times), 1), -1., dtype=np.float32)
    stats_by_episode = {}
    total_counts = {str(key): 0 for key in (-1, 0, 1, 2)}
    for start, end in zip(bounds[:-1], bounds[1:]):
        episode = int(episode_ids[start])
        local_times = torch.from_numpy((times[start:end]-times[start]).astype(np.float64)*1e-9)
        signal = torch.from_numpy(external[start:end]).abs().sum(-1)
        labels = contact_phase_labels_from_signal(signal, [(0, end-start)], gate,
            timestamps_s=local_times, valid_mask=torch.from_numpy(validity[start:end])).numpy()
        phases[start:end] = labels
        stats_by_episode[episode] = compute_episode_stats({PHASE: labels}, {PHASE: info["features"][PHASE]})[PHASE]
        counts = {str(key): int((labels == key).sum()) for key in (-1, 0, 1, 2)}
        report["episodes"][episode]["phase_counts"] = counts
        report["episodes"][episode]["label_sha256"] = hashlib.sha256(
            external[start:end].tobytes()+validity[start:end].astype(np.uint8).tobytes()+labels.tobytes()).hexdigest()
        for key, count in counts.items():
            total_counts[key] += count
    cursor = 0
    for path, original in zip(paths, tables):
        field = original.schema.field(PHASE)
        values = phases[cursor:cursor+len(original)]
        replacement = pa.array(values[:, 0] if pa.types.is_floating(field.type) else values.tolist(), type=field.type)
        updated = original.set_column(original.schema.get_field_index(PHASE), field, replacement)
        pq.write_table(updated, path)
        reread = pq.read_table(path)
        for key in original.column_names:
            if key != PHASE and not original[key].equals(reread[key]):
                raise ValueError(f"Relabel modified another column: {key}")
        np.testing.assert_array_equal(np.asarray(reread[PHASE].to_pylist()).reshape(-1), values[:, 0])
        cursor += len(original)
    for path in sorted((directory/"meta/episodes").rglob("*.parquet")):
        original = pq.read_table(path)
        ids = original["episode_index"].to_pylist()
        updated = original
        for key in original.column_names:
            prefix = f"stats/{PHASE}/"
            if key.startswith(prefix):
                statistic = key[len(prefix):]
                values = [stats_by_episode[int(episode)][statistic].tolist() for episode in ids]
                updated = updated.set_column(updated.schema.get_field_index(key), original.schema.field(key),
                                             pa.array(values, type=original.schema.field(key).type))
        pq.write_table(updated, path)
    stats_path = directory/"meta/stats.json"
    stats = json.loads(stats_path.read_text())
    aggregate = aggregate_stats([{PHASE: value} for value in stats_by_episode.values()])
    stats[PHASE] = serialize_dict(aggregate)[PHASE]
    stats_path.write_text(json.dumps(stats, indent=2)+"\n")
    report["phase_counts"] = total_counts
    report["phase_rule"]["contact_threshold"] = float(threshold)
    report["label_contract_sha256"] = hashlib.sha256(json.dumps({"teacher": report["teacher"],
        "phase_rule": report["phase_rule"], "episodes": report["episodes"]}, sort_keys=True).encode()).hexdigest()
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    manifest_path = directory/"meta/world_model_timeline.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contact_labels"] = report["phase_rule"]
    manifest["label_contract_sha256"] = report["label_contract_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    config_path = directory/"meta/wm_conversion_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tau_labels"]["contact_threshold"] = float(threshold)
    config["io"]["output"] = str(final_directory)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    validation_path = directory/"meta/wm_validation.json"
    summary = json.loads(validation_path.read_text())
    summary.update(phase_counts=total_counts, contact_threshold=float(threshold),
                   label_contract_sha256=report["label_contract_sha256"], relabel_validation="all_rows_passed_other_columns_unchanged")
    validation_path.write_text(json.dumps(summary, indent=2)+"\n")
    print(directory.name, "threshold", threshold, "phase counts", total_counts, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT/"data/nero_data")
    parser.add_argument("--threshold", type=float, default=1.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.data_root.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    stage = root.parent/f".nero_contact_relabel_{stamp}"
    backup = root.parent/f"nero_data_backup_before_contact_relabel_{stamp}"
    stage.mkdir()
    summaries = {}
    for _, _, name in TASKS:
        shutil.copytree(root/name, stage/name)
        summaries[name] = relabel_task(stage/name, args.threshold, root/name)
    preparation = json.loads((root/"wm_preparation_summary.json").read_text())
    preparation.update(tasks=summaries, contact_threshold=args.threshold, contact_relabel_backup=str(backup))
    replace_datasets(stage, root, backup, [name for _, _, name in TASKS])
    (root/"wm_preparation_summary.json").write_text(json.dumps(preparation, indent=2)+"\n")
    stage.rmdir()
    print("Previous labels backed up:", backup, flush=True)


if __name__ == "__main__":
    main()
