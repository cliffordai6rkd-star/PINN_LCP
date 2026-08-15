"""Build a LeRobot v3 world-model dataset on the raw lowdim rows.

State features are copied by raw index from ``teleop/timestamp_us``; this
converter never creates a new high-rate grid. Actions intentionally match the
DP conversion contract:

1. every recorded wrist-camera timestamp is an action anchor (no new grid);
2. every anchor selects the first teleop action row at or after it (``next``);
3. that sampled action is held on raw lowdim rows until the next camera anchor.

Raw state rows without a valid camera-anchor ``next`` action are omitted.
Retained rows keep their original indices and timestamps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from data_process.tool.h5_2_lerobotev3 import (
    H5Dataset,
    LeRobotV3Dataset,
    config_bool,
    config_int,
    config_path,
    config_str,
    load_conversion_deps,
    load_h5py,
    load_shape_meta,
    normalize_feature_spec,
    normalize_fps,
    normalize_h5_sources,
)


WM_TIMELINE_MODE = "raw_lowdim_action_hold"
WM_MANIFEST_NAME = "world_model_timeline.json"
GENERATED_TIMING_FEATURES = {
    "timing.state_timestamp_ns": {"dtype": "int64", "shape": (1,)},
    "timing.action_anchor_timestamp_ns": {"dtype": "int64", "shape": (1,)},
    "timing.action_source_timestamp_ns": {"dtype": "int64", "shape": (1,)},
    "timing.action_update": {"dtype": "uint8", "shape": (1,)},
    "timing.action_index": {"dtype": "int64", "shape": (1,)},
    "timing.action_phase_ns": {"dtype": "int64", "shape": (1,)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy raw lowdim H5 rows to LeRobot v3 and hold DP-compatible "
            "camera-rate actions on those rows."
        )
    )
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None, help="Override io.input.")
    parser.add_argument("--output", type=Path, default=None, help="Override io.output.")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def _positive_finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return result


def normalize_wm_timeline(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("shape_meta.timeline must be a mapping.")
    unknown = set(value) - {
        "mode",
        "state_timestamp_path",
        "action_anchor_timestamp_path",
        "action_fps",
        "max_action_gap_s",
    }
    if unknown:
        raise ValueError(f"shape_meta.timeline has unknown options: {sorted(unknown)}")
    mode = str(value.get("mode", "")).strip().lower()
    if mode != WM_TIMELINE_MODE:
        raise ValueError(f"timeline.mode must be {WM_TIMELINE_MODE!r}.")
    state_timestamp_path = value.get("state_timestamp_path")
    anchor_timestamp_path = value.get("action_anchor_timestamp_path")
    if not isinstance(state_timestamp_path, str) or not state_timestamp_path:
        raise ValueError("timeline.state_timestamp_path is required.")
    if not isinstance(anchor_timestamp_path, str) or not anchor_timestamp_path:
        raise ValueError("timeline.action_anchor_timestamp_path is required.")
    return {
        "mode": mode,
        "state_timestamp_path": state_timestamp_path,
        "action_anchor_timestamp_path": anchor_timestamp_path,
        "action_fps": normalize_fps(value.get("action_fps", 25)),
        "max_action_gap_s": _positive_finite(
            value.get("max_action_gap_s", 0.02), "max_action_gap_s"
        ),
    }


def _output_feature_spec(raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    feature_spec = {
        key: value
        for key, value in raw_spec.items()
        if key
        not in {
            "rate",
            "h5_path",
            "h5_paths",
            "sources",
            "timestamp_path",
            "align",
            "resample",
            "max_gap_s",
            "allow_stale",
            "transform",
            "combine",
        }
    }
    normalize_feature_spec(feature_spec)
    dtype = str(feature_spec.get("dtype", feature_spec.get("type", ""))).lower()
    if dtype in {"image", "video"}:
        raise ValueError("h5_v3_wm only supports low-dimensional features.")
    return feature_spec


def build_wm_conversion_spec(shape_meta: Mapping[str, Any]) -> dict[str, Any]:
    fps = normalize_fps(shape_meta.get("fps"))
    timeline = normalize_wm_timeline(shape_meta.get("timeline"))
    raw_features = shape_meta.get("features")
    if not isinstance(raw_features, Mapping) or not raw_features:
        raise ValueError("shape_meta must contain a non-empty features mapping.")

    mappings: list[dict[str, Any]] = []
    lerobot_features: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in raw_features.items():
        key = str(raw_key)
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Feature {key!r} spec must be a mapping.")
        rate = str(raw_spec.get("rate", "")).strip().lower()
        if rate not in {"state", "action"}:
            raise ValueError(f"Feature {key!r} rate must be 'state' or 'action'.")
        sources = normalize_h5_sources(
            key, raw_spec, dual_rate=(rate == "action")
        )
        required_method = "index" if rate == "state" else "next"
        if any(source["method"] != required_method for source in sources):
            contract = "align='index'" if rate == "state" else "resample='next'"
            raise ValueError(f"Feature {key!r} must use {contract}.")
        if rate == "state":
            wrong_timestamps = {
                source["timestamp_path"]
                for source in sources
                if source["timestamp_path"]
                not in {None, timeline["state_timestamp_path"]}
            }
            if wrong_timestamps:
                raise ValueError(
                    f"State feature {key!r} must share "
                    f"{timeline['state_timestamp_path']!r}."
                )

        feature_spec = _output_feature_spec(raw_spec)
        lerobot_features[key] = feature_spec
        mappings.append(
            {
                "lerobot_key": key,
                "rate": rate,
                "sources": sources,
                "h5_paths": [source["h5_path"] for source in sources],
                "transform": raw_spec.get("transform"),
                "combine": raw_spec.get("combine"),
                "max_gap_s": raw_spec.get("max_gap_s"),
                "feature_spec": feature_spec,
            }
        )

    state_mappings = [mapping for mapping in mappings if mapping["rate"] == "state"]
    action_mappings = [mapping for mapping in mappings if mapping["rate"] == "action"]
    if not state_mappings or not action_mappings:
        raise ValueError("At least one state and one action feature are required.")
    action_timestamp_paths = {
        source["timestamp_path"]
        for mapping in action_mappings
        for source in mapping["sources"]
    }
    if None in action_timestamp_paths or len(action_timestamp_paths) != 1:
        raise ValueError("All action features must share one timestamp_path.")
    duplicates = sorted(set(lerobot_features) & set(GENERATED_TIMING_FEATURES))
    if duplicates:
        raise ValueError(f"Generated timing keys must not be declared: {duplicates}")
    lerobot_features.update(GENERATED_TIMING_FEATURES)
    return {
        "task": str(shape_meta.get("task", "world_model")),
        "fps": fps,
        "timeline": timeline,
        "mappings": mappings,
        "state_mappings": state_mappings,
        "action_mappings": action_mappings,
        "action_source_timestamp_path": next(iter(action_timestamp_paths)),
        "lerobot_features": lerobot_features,
    }


def _raw_timestamps_to_ns(h5_dataset: H5Dataset, values, timestamp_path: str):
    np = h5_dataset.np
    raw = np.asarray(values).reshape(-1)
    scale_to_ns = h5_dataset._timestamp_seconds_scale(timestamp_path) / 1.0e-9
    rounded_scale = round(scale_to_ns)
    if np.issubdtype(raw.dtype, np.integer) and math.isclose(
        scale_to_ns, rounded_scale, rel_tol=0.0, abs_tol=1.0e-12
    ):
        return raw.astype(np.int64) * int(rounded_scale)
    return np.rint(raw.astype(np.float64) * scale_to_ns).astype(np.int64)


def _dp_action_anchors(h5_dataset: H5Dataset, raw_camera_timestamps, path: str, fps: int):
    """Return the exact recorded wrist timestamps used by camera_rows DP."""

    del fps
    values = h5_dataset.np.asarray(raw_camera_timestamps).reshape(-1)
    # Validate through the common parser, but never modify the timestamps.
    h5_dataset._timestamps_seconds(values, path)
    return values.copy()


def build_wm_episode_cache(
    h5_dataset: H5Dataset,
    h5_file: Any,
    spec: Mapping[str, Any],
    h5_path: Path,
) -> dict[str, Any]:
    np = h5_dataset.np
    if np is None:
        raise RuntimeError("h5_v3_wm conversion requires numpy.")
    timeline = spec["timeline"]
    state_timestamp_path = timeline["state_timestamp_path"]
    camera_timestamp_path = timeline["action_anchor_timestamp_path"]
    dataset_paths = {state_timestamp_path, camera_timestamp_path}
    for mapping in spec["mappings"]:
        for source in mapping["sources"]:
            dataset_paths.add(source["h5_path"])
            if source["timestamp_path"] is not None:
                dataset_paths.add(source["timestamp_path"])
    datasets = {
        path: h5_dataset._dataset(h5_file, path, h5_path) for path in dataset_paths
    }

    raw_state_timestamps = datasets[state_timestamp_path][:]
    raw_camera_timestamps = datasets[camera_timestamp_path][:]
    raw_state_seconds = h5_dataset._timestamps_seconds(
        raw_state_timestamps, state_timestamp_path
    )
    all_action_anchor_raw = _dp_action_anchors(
        h5_dataset,
        raw_camera_timestamps,
        camera_timestamp_path,
        int(timeline["action_fps"]),
    )
    all_action_anchor_seconds = h5_dataset._timestamps_seconds(
        all_action_anchor_raw, camera_timestamp_path
    )
    action_source_path = spec["action_source_timestamp_path"]
    action_source_seconds = h5_dataset._timestamps_seconds(
        datasets[action_source_path][:], action_source_path
    )
    next_indices = np.searchsorted(
        action_source_seconds, all_action_anchor_seconds, side="left"
    )
    anchor_valid = next_indices < len(action_source_seconds)
    safe_next = next_indices.clip(0, len(action_source_seconds) - 1)
    anchor_valid &= (
        action_source_seconds[safe_next] - all_action_anchor_seconds
        <= float(timeline["max_action_gap_s"]) + 1.0e-12
    )
    action_anchor_raw = all_action_anchor_raw[anchor_valid]
    action_anchor_seconds = all_action_anchor_seconds[anchor_valid]
    if action_anchor_seconds.size == 0:
        raise ValueError(f"No camera anchor has a valid next action in {h5_path}.")

    # A raw lowdim row belongs to the latest recorded camera anchor at/before
    # it. If that camera anchor has no legal next action, the row has no label
    # and is excluded rather than padded or marked for downstream filtering.
    raw_camera_index = np.searchsorted(
        all_action_anchor_seconds, raw_state_seconds, side="right"
    ) - 1
    state_has_label = raw_camera_index >= 0
    safe_camera_index = raw_camera_index.clip(0, len(anchor_valid) - 1)
    state_has_label &= anchor_valid[safe_camera_index]
    state_indices = np.flatnonzero(state_has_label).astype(np.int64, copy=False)
    if state_indices.size == 0:
        raise ValueError(f"No raw state row has a valid action label in {h5_path}.")
    state_seconds = raw_state_seconds[state_indices]
    state_count = len(state_seconds)

    timestamp_seconds: dict[str, Any] = {
        state_timestamp_path: raw_state_seconds,
        camera_timestamp_path: h5_dataset._timestamps_seconds(
            raw_camera_timestamps, camera_timestamp_path
        ),
    }
    for mapping in spec["mappings"]:
        for source in mapping["sources"]:
            timestamp_path = source["timestamp_path"]
            if timestamp_path is not None and timestamp_path not in timestamp_seconds:
                timestamp_seconds[timestamp_path] = h5_dataset._timestamps_seconds(
                    datasets[timestamp_path][:], timestamp_path
                )
            expected_rows = (
                len(raw_state_seconds)
                if mapping["rate"] == "state"
                else len(timestamp_seconds[timestamp_path])
            )
            if len(datasets[source["h5_path"]]) != expected_rows:
                raise ValueError(
                    f"Feature {source['h5_path']!r} has "
                    f"{len(datasets[source['h5_path']])} rows, expected "
                    f"{expected_rows} in {h5_path}."
                )

    cache: dict[str, Any] = {
        "datasets": datasets,
        "timestamp_seconds": timestamp_seconds,
        "state_timestamps": state_seconds,
        "action_anchors": action_anchor_seconds,
        "resampled": {},
        "wm_raw_lowdim_rows": True,
    }
    for mapping in spec["state_mappings"]:
        cache["resampled"][mapping["lerobot_key"]] = (
            h5_dataset._sample_snapshot_mapping(mapping, state_indices, cache)
        )

    for mapping in spec["action_mappings"]:
        for source in mapping["sources"]:
            max_gap_s = source.get("max_gap_s")
            if max_gap_s is None:
                max_gap_s = mapping.get("max_gap_s")
            if max_gap_s is None:
                max_gap_s = timeline["max_action_gap_s"]
            h5_dataset._validate_resample_targets(
                timestamp_seconds[source["timestamp_path"]],
                action_anchor_seconds,
                "next",
                float(max_gap_s),
                mapping["lerobot_key"],
                h5_path,
            )
        cache["resampled"][mapping["lerobot_key"]] = (
            h5_dataset._resample_mapping(mapping, action_anchor_seconds, cache)
        )

    valid_anchor_index = np.full(len(anchor_valid), -1, dtype=np.int64)
    valid_anchor_index[anchor_valid] = np.arange(
        int(anchor_valid.sum()), dtype=np.int64
    )
    held_action_index = valid_anchor_index[raw_camera_index[state_indices]]
    if np.any(held_action_index < 0):
        raise RuntimeError("Internal error: retained state row has no action label.")
    for mapping in spec["action_mappings"]:
        key = mapping["lerobot_key"]
        cache["resampled"][key] = cache["resampled"][key][held_action_index]

    source_path = spec["action_source_timestamp_path"]
    selected_source_indices = h5_dataset._point_sample_indices(
        timestamp_seconds[source_path], action_anchor_seconds, "next"
    )
    source_ns = _raw_timestamps_to_ns(
        h5_dataset, datasets[source_path][:], source_path
    )[selected_source_indices]
    raw_state_ns = _raw_timestamps_to_ns(
        h5_dataset, raw_state_timestamps, state_timestamp_path
    )
    state_ns = raw_state_ns[state_indices]
    anchor_ns = _raw_timestamps_to_ns(
        h5_dataset, action_anchor_raw, camera_timestamp_path
    )
    held_anchor_ns = anchor_ns[held_action_index]
    held_source_ns = source_ns[held_action_index]

    action_update = np.zeros(state_count, dtype=np.uint8)
    action_update[0] = 1
    action_update[1:] = (
        held_action_index[1:] != held_action_index[:-1]
    ).astype(np.uint8)
    cache["timing"] = {
        "timing.state_timestamp_ns": state_ns[:, None],
        "timing.action_anchor_timestamp_ns": held_anchor_ns[:, None],
        "timing.action_source_timestamp_ns": held_source_ns[:, None],
        "timing.action_update": action_update[:, None],
        "timing.action_index": held_action_index.astype(np.int64)[:, None],
        "timing.action_phase_ns": (state_ns - held_anchor_ns)[:, None],
    }
    return cache


def read_wm_frame(cache: Mapping[str, Any], frame_idx: int) -> dict[str, Any]:
    frame = {key: values[frame_idx] for key, values in cache["resampled"].items()}
    frame.update({key: values[frame_idx] for key, values in cache["timing"].items()})
    return frame


def write_wm_manifest(output_path: Path, spec: Mapping[str, Any]) -> Path:
    path = output_path / "meta" / WM_MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    timeline = spec["timeline"]
    manifest = {
        "schema_version": 2,
        "mode": WM_TIMELINE_MODE,
        "row_sampling": "action_labeled_raw_state_index",
        "nominal_lerobot_fps": spec["fps"],
        "state_timestamp_path": timeline["state_timestamp_path"],
        "action_anchor_timestamp_path": timeline[
            "action_anchor_timestamp_path"
        ],
        "action_fps": timeline["action_fps"],
        "action_anchor_grid": "recorded_camera_rows",
        "action_sampling": "next",
        "action_upsampling": "previous_anchor_hold",
        "unlabeled_state_rows": "drop",
        "action_update_key": "timing.action_update",
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return path


def run_inspect(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    dataset = H5Dataset(
        config_path(config, "input", override=getattr(args, "input", None)),
        h5py=load_h5py(),
        max_episodes=config_int(config, "max_episodes"),
    )
    dataset.inspect()


def run_conversion(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    spec = build_wm_conversion_spec(config)
    h5py, np, LeRobotDataset = load_conversion_deps()
    output_path = config_path(config, "output", override=getattr(args, "output", None))
    dataset = H5Dataset(
        config_path(config, "input", override=getattr(args, "input", None)),
        h5py=h5py,
        np=np,
        max_episodes=config_int(config, "max_episodes"),
    )
    writer = LeRobotV3Dataset(
        LeRobotDataset,
        repo_id=config_str(config, "repo_id", "local/world_model_raw_rows"),
        root=output_path,
        fps=spec["fps"],
        features=spec["lerobot_features"],
        no_videos=config_bool(config, "no_videos", True),
    )
    episode_iter = tqdm(dataset.files(), desc="world-model episodes", unit="episode")
    try:
        for h5_path in episode_iter:
            episode_iter.set_postfix_str(h5_path.name)
            with dataset.open_episode(h5_path) as h5_file:
                cache = build_wm_episode_cache(dataset, h5_file, spec, h5_path)
                try:
                    for frame_idx in tqdm(
                        range(len(cache["state_timestamps"])),
                        desc=f"raw lowdim rows {h5_path.name}",
                        unit="frame",
                        leave=False,
                    ):
                        writer.add_frame(
                            read_wm_frame(cache, frame_idx), task=spec["task"]
                        )
                finally:
                    dataset.clear_episode_cache(cache)
            writer.save_episode(task=spec["task"])
    finally:
        writer.finalize()

    print(f"world-model timeline manifest: {write_wm_manifest(output_path, spec)}")
    if config_bool(config, "push_to_hub"):
        writer.push_to_hub()


def main() -> None:
    args = parse_args()
    if args.inspect_only:
        run_inspect(args)
    else:
        run_conversion(args)


if __name__ == "__main__":
    main()
