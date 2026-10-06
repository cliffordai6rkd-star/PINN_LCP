"""Build and validate four 100 Hz Nero WM datasets before replacing VA exports."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from data_process.tool.h5_2_lerobotev3 import H5Dataset, LeRobotV3Dataset, load_conversion_deps
from data_process.tool.h5_v3_wm import build_wm_conversion_spec, build_wm_episode_cache, read_wm_frame, write_wm_manifest
from model.nero_tau_free import NeroTorqueTeacher
from model.pinn_model.contact_gate import ContactGateConfig, contact_phase_labels_from_signal

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("insert_usb", "insert_usb", "insert_usb_lerobotv3"),
    ("push_button", "push_button", "push_button_lerobotv3"),
    ("cucumber_peeling", "cuccumber_peeling", "cuccumber_peeling_lerobotv3"),
    ("wipe_board", "wipe_board", "wipe_board_lerobotv3"),
)
LABEL_FEATURES = {
    "observation.tau_free": {"dtype": "float32", "shape": (7,)},
    "observation.tau_ext": {"dtype": "float32", "shape": (7,)},
    "observation.tau_label_valid": {"dtype": "uint8", "shape": (1,)},
    "observation.contact_phase": {"dtype": "float32", "shape": (1,)},
}


def save_lowdim_episode(writer, task):
    """Skip LeRobot's per-row image embedding for an image-free numeric table.

    The installed writer always calls embed_images, even for numeric-only data.
    Its image conversion is an identity here; stats and Parquet writing still
    use the ordinary writer. The override is process-local and restored at once.
    """
    module = importlib.import_module(type(writer.dataset).__module__)
    original = getattr(module, "embed_images", None)
    if original is None or writer.dataset.meta.image_keys or writer.dataset.meta.video_keys:
        return writer.save_episode(task=task)
    module.embed_images = lambda dataset: dataset
    try:
        writer.save_episode(task=task)
    finally:
        module.embed_images = original


def conversion_config(task, source, destination, checkpoint, threshold):
    config = yaml.safe_load((ROOT/f"config/shape_meta/swm/nero/{task}.yaml").read_text())
    config["io"].update(input=str(source), output=str(destination), repo_id=destination.name,
                        max_episodes=None, no_videos=True, push_to_hub=False)
    # Recorded labels belong to other models. This export derives all torque labels below.
    config["features"].pop("observation.tau_ext", None)
    for feature in config["features"].values():
        feature["lowpass"] = False
        feature.pop("cutoff_hz", None)
        feature.pop("order", None)
    config["features"]["observation.q_cmd"] = {
        "rate": "state", "dtype": "float32", "shape": [7],
        "h5_path": "teleop/q_cmd", "align": "index", "lowpass": False,
    }
    config["tau_labels"] = {
        "checkpoint": str(checkpoint), "metric": "tau_ext_l1", "contact_threshold": threshold,
        "precontact_duration_s": 1., "valid_key": "observation.tau_label_valid",
        "phase_key": "observation.contact_phase",
    }
    return config


def label_columns(teacher, h5, cache, gate):
    timestamp_ns = np.asarray(h5["teleop/timestamp_us"], dtype=np.int64).reshape(-1)*1000
    columns = [np.asarray(h5[f"teleop/{key}"])
               for key in ("q_follower", "dq_follower", "q_cmd", "tau_follower")]
    prediction = teacher.predict(timestamp_ns, *columns)
    retained = cache["timing"]["timing.state_timestamp_ns"][:, 0]
    indices = np.searchsorted(timestamp_ns, retained)
    np.testing.assert_array_equal(timestamp_ns[indices], retained)
    external = prediction["tau_ext"][indices]
    valid = prediction["valid_context"][indices]
    times = (retained-retained[0]).astype(np.float64)*1e-9
    phase = contact_phase_labels_from_signal(
        torch.from_numpy(np.abs(external).sum(-1)), [(0, len(retained))], gate,
        timestamps_s=torch.from_numpy(times), valid_mask=torch.from_numpy(valid),
    ).numpy()
    return {"observation.tau_free": prediction["tau_free"][indices],
            "observation.tau_ext": external,
            "observation.tau_label_valid": valid.astype(np.uint8)[:, None],
            "observation.contact_phase": phase}


def build_task(config, teacher, destination, *, max_episodes=None):
    h5py, _, dataset_class = load_conversion_deps()
    dataset = H5Dataset(Path(config["io"]["input"]), h5py=h5py, np=np, max_episodes=max_episodes)
    files = dataset.files()
    spec = build_wm_conversion_spec(config)
    spec["lerobot_features"].update(copy.deepcopy(LABEL_FEATURES))
    writer = LeRobotV3Dataset(dataset_class, repo_id=destination.name, root=destination,
                            fps=100, features=spec["lerobot_features"], no_videos=True)
    gate = ContactGateConfig(enabled=True, label_mode="three_phase", metric="tau_ext_l1",
                             contact_threshold=config["tau_labels"]["contact_threshold"],
                             precontact_duration_s=1., state_rate_hz=100.)
    report = {"task": config["task"], "source": config["io"]["input"],
              "teacher": teacher.contract, "state_fps": 100, "action_fps": 25,
              "phase_rule": {"metric": "sum_abs_tau_ext_nm", "contact_threshold": gate.contact_threshold,
                  "comparison": ">", "alignment": "one_second_preceding_each_contact_onset_contact_takes_priority",
                  "precontact_duration_s": 1., "classes": {"-1": "unknown_context", "0": "free", "1": "alignment", "2": "contact"}},
              "episodes": [], "phase_counts": {str(key): 0 for key in (-1, 0, 1, 2)}}
    try:
        for index, path in enumerate(files):
            with dataset.open_episode(path) as h5:
                cache = build_wm_episode_cache(dataset, h5, spec, path)
                labels = label_columns(teacher, h5, cache, gate)
                cache["resampled"].update(labels)
                rows = len(cache["state_timestamps"])
                for row in range(rows):
                    writer.add_frame(read_wm_frame(cache, row), task=spec["task"])
                counts = {str(key): int((labels["observation.contact_phase"] == key).sum()) for key in (-1, 0, 1, 2)}
                raw_rows = len(h5["teleop/timestamp_us"])
                dataset.clear_episode_cache(cache)
            save_lowdim_episode(writer, spec["task"])
            report["episodes"].append({"episode_index": index, "source_file": str(path.resolve()),
                                        "raw_rows": raw_rows, "retained_rows": rows, "phase_counts": counts})
            for key, count in counts.items():
                report["phase_counts"][key] += count
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(files):
                print(f"{spec['task']}: converted {index+1}/{len(files)} episodes", flush=True)
    finally:
        writer.finalize()
    manifest_path = write_wm_manifest(destination, spec)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(torque_labels=teacher.contract, contact_labels=report["phase_rule"],
                    tau_label_valid_key="observation.tau_label_valid", contact_phase_key="observation.contact_phase")
    report["label_contract_sha256"] = hashlib.sha256(json.dumps({
        "teacher": teacher.contract, "phase_rule": report["phase_rule"], "episodes": report["episodes"]}, sort_keys=True).encode()).hexdigest()
    manifest["label_contract_sha256"] = report["label_contract_sha256"]
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    (destination/"meta/torque_label_report.json").write_text(json.dumps(report, indent=2)+"\n")
    (destination/"meta/wm_conversion_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return report


def validate_task(destination, config, report):
    """Check every exported row against its original H5/timeline and phase rule."""
    h5py, _, _ = load_conversion_deps()
    info = json.loads((destination/"meta/info.json").read_text())
    if info["fps"] != 100 or info["total_episodes"] != len(report["episodes"]):
        raise ValueError("Converted episode/rate metadata mismatch")
    if any(spec["dtype"] in {"video", "image"} for spec in info["features"].values()):
        raise ValueError("WM export must contain low-dimensional state/action features")
    for key in LABEL_FEATURES:
        if key not in info["features"]:
            raise ValueError(f"Missing label feature {key}")
    tables = [pq.read_table(path) for path in sorted((destination/"data").rglob("*.parquet"))]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    if len(table) != info["total_frames"] or len(table) != sum(ep["retained_rows"] for ep in report["episodes"]):
        raise ValueError("Converted frame count mismatch")
    numeric = {key: np.asarray(table[key].to_pylist()) for key in info["features"]}
    for key, values in numeric.items():
        if values.ndim == 1 and info["features"][key]["shape"] == [1]:
            numeric[key] = values[:, None]
    if any(not np.isfinite(value).all() for value in numeric.values()):
        raise ValueError("Nonfinite converted data")
    dataset = H5Dataset(Path(config["io"]["input"]), h5py=h5py, np=np)
    spec = build_wm_conversion_spec(config)
    gate = ContactGateConfig(enabled=True, label_mode="three_phase", metric="tau_ext_l1",
                             contact_threshold=config["tau_labels"]["contact_threshold"], precontact_duration_s=1.)
    cursor = 0
    for episode in report["episodes"]:
        end = cursor+episode["retained_rows"]
        with h5py.File(episode["source_file"], "r") as h5:
            expected = build_wm_episode_cache(dataset, h5, spec, Path(episode["source_file"]))
            for key, values in {**expected["resampled"], **expected["timing"]}.items():
                np.testing.assert_allclose(numeric[key][cursor:end], values, rtol=0, atol=0)
        if not (numeric["episode_index"][cursor:end] == episode["episode_index"]).all():
            raise ValueError("Converted episode boundaries mismatch")
        times = numeric["timing.state_timestamp_ns"][cursor:end].reshape(-1).astype(np.int64)
        valid = numeric["observation.tau_label_valid"][cursor:end].reshape(-1).astype(bool)
        external = numeric["observation.tau_ext"][cursor:end].astype(np.float32)
        np.testing.assert_allclose(external[valid],
            (numeric["observation.torque"][cursor:end].astype(np.float32)-numeric["observation.tau_free"][cursor:end].astype(np.float32))[valid], rtol=0, atol=0)
        phases = contact_phase_labels_from_signal(torch.from_numpy(np.abs(external).sum(-1)), [(0, len(times))], gate,
            timestamps_s=torch.from_numpy((times-times[0]).astype(np.float64)*1e-9), valid_mask=torch.from_numpy(valid)).numpy()
        np.testing.assert_array_equal(numeric["observation.contact_phase"][cursor:end], phases)
        # Exported q/cmd round independently; delta is subtracted at source precision.
        np.testing.assert_allclose(numeric["observation.q_cmd"][cursor:end].astype(np.float32)-numeric["observation.joint"][cursor:end].astype(np.float32),
                                   numeric["observation.delta_q"][cursor:end].astype(np.float32), rtol=0, atol=1e-6)
        cursor = end
    summary = {"episodes": info["total_episodes"], "frames": info["total_frames"],
               "fps": info["fps"], "action_fps": 25, "phase_counts": report["phase_counts"],
               "label_contract_sha256": report["label_contract_sha256"], "validation": "all_rows_passed"}
    (destination/"meta/wm_validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Validated {destination.name}: {summary}", flush=True)
    return summary


def replace_datasets(stage, output, backup, names):
    """Roll back all installed directories if any rename fails."""
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=False)
    moved, installed = [], []
    try:
        for name in names:
            if (output/name).exists():
                (output/name).rename(backup/name)
                moved.append(name)
            (stage/name).rename(output/name)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            (output/name).rename(stage/name)
        for name in reversed(moved):
            (backup/name).rename(output/name)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT.parent/"nero_ws/runs")
    parser.add_argument("--output-root", type=Path, default=ROOT/"data/nero_data")
    parser.add_argument("--teacher", type=Path, default=ROOT/"outputs/tau_free_sequence/nero/epoch_124_val_tau_mse_nm2_0.005756.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--contact-threshold", type=float, default=1.)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    output = args.output_root.resolve()
    stage = (args.stage_root or output.parent/f".nero_wm_staging_{stamp}").resolve()
    backup = (args.backup_root or output.parent/f"nero_data_backup_25hz_{stamp}").resolve()
    if stage.exists() or backup.exists():
        raise FileExistsError("Staging/backup directory already exists; select fresh paths")
    stage.mkdir(parents=True)
    teacher = NeroTorqueTeacher(args.teacher, device=args.device)
    summaries = {}
    for task, raw_name, name in TASKS:
        source = args.input_root.resolve()/raw_name
        destination = stage/name
        config = conversion_config(task, source, destination, teacher.path, args.contact_threshold)
        report = build_task(config, teacher, destination, max_episodes=args.max_episodes)
        summaries[name] = validate_task(destination, config, report)
    manifest = {"staging": str(stage), "destination": str(output), "backup": str(backup) if args.replace else None,
                "teacher": teacher.contract, "tasks": summaries, "replaced": args.replace}
    (stage/"wm_preparation_summary.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if args.replace:
        replace_datasets(stage, output, backup, [name for _, _, name in TASKS])
        shutil.copy2(stage/"wm_preparation_summary.json", output/"wm_preparation_summary.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
