
# python dataset/tool/lerobot_add_feature.py \
#   --input-root data/train_episode/wrench_background/wrench_bg_lerobotv3 \
#   --input-repo-id wrench_bg_lerobotv3 \
#   --output-root data/train_episode/wrench_background/wrench_bg_lerobotv3_dv \
#   --output-repo-id wrench_bg_lerobotv3_dv \
#   --features acceleration ee_velocity ee_acceleration \
#   --overwrite


from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm


DEFAULT_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}

FEATURE_CHOICES = ("acceleration", "ee_velocity", "ee_acceleration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot dataset with selected derivative features added."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--input-repo-id", required=True, help="Source LeRobot repo id.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--output-repo-id", required=True, help="Output LeRobot repo id.")
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_CHOICES,
        default=list(FEATURE_CHOICES),
        help="Features to add. Default: add all supported derivative features.",
    )
    parser.add_argument("--velocity-key", default="observation.velocity")
    parser.add_argument("--acceleration-key", default="observation.acceleration")
    parser.add_argument("--ee-pose-key", default="action.ee_pose")
    parser.add_argument("--ee-velocity-key", default="observation.ee_velocity")
    parser.add_argument("--ee-acceleration-key", default="observation.ee_acceleration")
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--min-dt", type=float, default=1e-6, help="Lower bound for timestamp deltas.")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument(
        "--keep-videos",
        action="store_true",
        help="Copy video features too. By default this script writes a lowdim-only dataset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove output-root before writing.")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def filter_supported_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def feature_to_plain_dict(feature: Any) -> dict[str, Any]:
    if isinstance(feature, dict):
        return dict(feature)
    if hasattr(feature, "to_dict"):
        return dict(feature.to_dict())
    if hasattr(feature, "__dict__"):
        return dict(feature.__dict__)
    raise TypeError(f"Unsupported feature spec type: {type(feature).__name__}")


def is_video_feature(feature: Any) -> bool:
    spec = feature_to_plain_dict(feature)
    dtype = str(spec.get("dtype", "")).lower()
    return dtype in {"video", "image"}


def source_features_for_create(
    source_dataset: Any,
    *,
    selected_features: set[str],
    velocity_key: str,
    acceleration_key: str,
    ee_pose_key: str,
    ee_velocity_key: str,
    ee_acceleration_key: str,
    keep_videos: bool,
) -> dict[str, Any]:
    features = {}
    for key, spec in source_dataset.features.items():
        if key in DEFAULT_FEATURE_KEYS:
            continue
        if not keep_videos and is_video_feature(spec):
            continue
        features[key] = feature_to_plain_dict(spec)

    if "acceleration" in selected_features:
        features[acceleration_key] = derivative_feature_spec(source_dataset, velocity_key)
    if "ee_velocity" in selected_features:
        features[ee_velocity_key] = ee_twist_feature_spec(source_dataset, ee_pose_key)
    if "ee_acceleration" in selected_features:
        features[ee_acceleration_key] = ee_twist_feature_spec(source_dataset, ee_pose_key)
    return features


def derivative_feature_spec(source_dataset: Any, source_key: str) -> dict[str, Any]:
    if source_key not in source_dataset.features:
        raise KeyError(f"missing source feature spec: {source_key}")
    spec = feature_to_plain_dict(source_dataset.features[source_key])
    if "shape" not in spec:
        raise KeyError(f"missing shape in source feature spec: {source_key}")
    return {
        "dtype": "float32",
        "shape": tuple(spec["shape"]),
    }


def ee_twist_feature_spec(source_dataset: Any, source_key: str) -> dict[str, Any]:
    if source_key not in source_dataset.features:
        raise KeyError(f"missing source feature spec: {source_key}")
    spec = feature_to_plain_dict(source_dataset.features[source_key])
    shape = tuple(spec.get("shape", ()))
    if shape != (4, 4):
        raise ValueError(f"{source_key} must have shape (4, 4) to compute ee twist, got {shape}.")
    return {
        "dtype": "float32",
        "shape": (6,),
    }


def create_dataset(
    LeRobotDataset: Any,
    *,
    repo_id: str,
    root: Path,
    fps: int,
    features: dict[str, Any],
    use_videos: bool,
    video_backend: str,
) -> Any:
    kwargs = {
        "repo_id": repo_id,
        "root": root,
        "fps": fps,
        "features": features,
        "use_videos": use_videos,
        "video_backend": video_backend,
    }
    return LeRobotDataset.create(**filter_supported_kwargs(LeRobotDataset.create, kwargs))


def add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    frame = dict(frame)
    frame.setdefault("task", task)
    kwargs = {"frame": frame, "task": task}
    try:
        dataset.add_frame(**filter_supported_kwargs(dataset.add_frame, kwargs))
    except TypeError:
        dataset.add_frame(frame)


def save_episode(dataset: Any, task: str) -> None:
    kwargs = {"task": task}
    try:
        dataset.save_episode(**filter_supported_kwargs(dataset.save_episode, kwargs))
    except TypeError:
        dataset.save_episode()


def frame_for_output(frame: Mapping[str, Any], output_feature_keys: set[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in frame.items()
        if key not in DEFAULT_FEATURE_KEYS and key in output_feature_keys
    }


def torch_dtype(dtype_name: str) -> Any:
    import torch

    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return dtypes.get(str(dtype_name).lower())


def coerce_value_to_feature(value: Any, feature: Mapping[str, Any]) -> Any:
    import torch

    dtype = torch_dtype(str(feature.get("dtype", "")))
    shape = tuple(feature.get("shape", ()))
    if dtype is None:
        return value

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(dtype=dtype)

    if shape and tuple(value.shape) != shape:
        if value.numel() != int(torch.tensor(shape).prod().item()):
            raise ValueError(f"Cannot reshape value from {tuple(value.shape)} to feature shape {shape}.")
        value = value.reshape(shape)

    return value


def coerce_frame_to_features(frame: dict[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    coerced = {}
    for key, value in frame.items():
        if key in features:
            coerced[key] = coerce_value_to_feature(value, features[key])
        else:
            coerced[key] = value
    return coerced


def get_task(source_dataset: Any, episode: Mapping[str, Any], fallback: str = "default") -> str:
    task = episode.get("task")
    if task is not None:
        return str(task)

    task_index = episode.get("tasks")
    if isinstance(task_index, list) and task_index:
        task_index = task_index[0]
    if task_index is None:
        task_index = episode.get("task_index")

    try:
        tasks = source_dataset.meta.tasks
        if tasks is not None and task_index is not None:
            return str(tasks[int(task_index)]["task"])
    except Exception:
        pass

    return fallback


def compute_acceleration(
    velocity: Any,
    previous_velocity: Any | None,
    dt: float,
) -> Any:
    import torch

    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity)
    velocity = velocity.to(dtype=torch.float32)
    if previous_velocity is None:
        return torch.zeros_like(velocity)
    if not torch.is_tensor(previous_velocity):
        previous_velocity = torch.as_tensor(previous_velocity)
    return (velocity - previous_velocity.to(dtype=torch.float32)) / dt


def scalar_timestamp(value: Any) -> float:
    import torch

    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def compute_episode_accelerations(
    velocities: list[Any],
    *,
    timestamps: list[Any],
    min_dt: float,
) -> list[Any]:
    import torch

    if not velocities:
        return []
    if len(velocities) != len(timestamps):
        raise ValueError("velocities and timestamps must have the same length.")

    accelerations = [torch.zeros_like(torch.as_tensor(velocities[0], dtype=torch.float32))]
    for idx in range(1, len(velocities)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        accelerations.append(compute_acceleration(velocities[idx], velocities[idx - 1], dt))

    if len(accelerations) > 1:
        accelerations[0] = accelerations[1].clone()
    return accelerations


def ee_pose_value(ee_pose: Any) -> Any:
    import torch

    if not torch.is_tensor(ee_pose):
        ee_pose = torch.as_tensor(ee_pose)
    ee_pose = ee_pose.to(dtype=torch.float32)
    if tuple(ee_pose.shape) != (4, 4):
        raise ValueError(f"ee pose must have shape (4, 4), got {tuple(ee_pose.shape)}.")
    return ee_pose


def so3_log(rotation: Any) -> Any:
    import torch

    rotation = torch.as_tensor(rotation, dtype=torch.float32)
    trace = torch.trace(rotation)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )

    sin_theta = torch.sin(theta)
    if torch.abs(sin_theta) < 1e-6:
        return 0.5 * vee
    return theta / (2.0 * sin_theta) * vee


def ee_velocity_from_poses(current_pose: Any, previous_pose: Any, dt: float) -> Any:
    import torch

    current_pose = ee_pose_value(current_pose)
    previous_pose = ee_pose_value(previous_pose)
    linear_velocity = (current_pose[:3, 3] - previous_pose[:3, 3]) / dt

    # Spatial angular velocity in the same frame as the pose translation.
    rotation_delta = current_pose[:3, :3] @ previous_pose[:3, :3].transpose(0, 1)
    angular_velocity = so3_log(rotation_delta) / dt
    return torch.cat((linear_velocity, angular_velocity), dim=0)


def compute_episode_ee_velocities(
    ee_poses: list[Any],
    *,
    timestamps: list[Any],
    min_dt: float,
) -> list[Any]:
    import torch

    if not ee_poses:
        return []
    if len(ee_poses) != len(timestamps):
        raise ValueError("ee_poses and timestamps must have the same length.")

    velocities = [torch.zeros(6, dtype=torch.float32)]
    for idx in range(1, len(ee_poses)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        velocities.append(ee_velocity_from_poses(ee_poses[idx], ee_poses[idx - 1], dt))

    if len(velocities) > 1:
        velocities[0] = velocities[1].clone()
    return velocities


def compute_episode_derivative(values: list[Any], *, timestamps: list[Any], min_dt: float) -> list[Any]:
    import torch

    if not values:
        return []
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have the same length.")

    derivatives = [torch.zeros_like(torch.as_tensor(values[0], dtype=torch.float32))]
    for idx in range(1, len(values)):
        dt = scalar_timestamp(timestamps[idx]) - scalar_timestamp(timestamps[idx - 1])
        if dt < min_dt:
            dt = min_dt
        current = torch.as_tensor(values[idx], dtype=torch.float32)
        previous = torch.as_tensor(values[idx - 1], dtype=torch.float32)
        derivatives.append((current - previous) / dt)

    if len(derivatives) > 1:
        derivatives[0] = derivatives[1].clone()
    return derivatives


def main() -> None:
    args = parse_args()

    if args.min_dt <= 0:
        raise ValueError("--min-dt must be positive.")
    selected_features = set(args.features)

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output root already exists: {args.output_root}")
        shutil.rmtree(args.output_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_dataset = LeRobotDataset(
        repo_id=args.input_repo_id,
        root=args.input_root,
        video_backend=args.video_backend,
    )

    features = source_features_for_create(
        source_dataset,
        selected_features=selected_features,
        velocity_key=args.velocity_key,
        acceleration_key=args.acceleration_key,
        ee_pose_key=args.ee_pose_key,
        ee_velocity_key=args.ee_velocity_key,
        ee_acceleration_key=args.ee_acceleration_key,
        keep_videos=args.keep_videos,
    )
    output_feature_keys = set(features.keys())
    use_videos = args.keep_videos and bool(getattr(source_dataset.meta, "video_keys", []))
    output_dataset = create_dataset(
        LeRobotDataset,
        repo_id=args.output_repo_id,
        root=args.output_root,
        fps=int(source_dataset.fps),
        features=features,
        use_videos=use_videos,
        video_backend=args.video_backend,
    )

    episodes = list(source_dataset.meta.episodes)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    episode_iter = tqdm(episodes, desc="episodes", unit="episode")
    for episode in episode_iter:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        task = get_task(source_dataset, episode)
        source_frames = [
            source_dataset[raw_idx] if args.keep_videos else source_dataset.hf_dataset[raw_idx]
            for raw_idx in range(start, end)
        ]
        velocities = []
        ee_poses = []
        timestamps = []
        for source_frame in source_frames:
            if "acceleration" in selected_features and args.velocity_key not in source_frame:
                raise KeyError(f"missing velocity key in source frame: {args.velocity_key}")
            if (
                {"ee_velocity", "ee_acceleration"} & selected_features
                and args.ee_pose_key not in source_frame
            ):
                raise KeyError(f"missing ee pose key in source frame: {args.ee_pose_key}")
            if args.timestamp_key not in source_frame:
                raise KeyError(f"missing timestamp key in source frame: {args.timestamp_key}")
            if "acceleration" in selected_features:
                velocities.append(source_frame[args.velocity_key])
            if {"ee_velocity", "ee_acceleration"} & selected_features:
                ee_poses.append(ee_pose_value(source_frame[args.ee_pose_key]))
            timestamps.append(source_frame[args.timestamp_key])

        # Derivatives are episode-local: each ee_pose/velocity sample is paired
        # with the timestamp from the same episode row, then differenced in order.
        # EE velocity is a 6D twist: [vx, vy, vz, wx, wy, wz].
        computed_features: dict[str, list[Any]] = {}
        if "acceleration" in selected_features:
            computed_features[args.acceleration_key] = compute_episode_accelerations(
                velocities,
                timestamps=timestamps,
                min_dt=args.min_dt,
            )
        if "ee_velocity" in selected_features or "ee_acceleration" in selected_features:
            ee_velocities = compute_episode_ee_velocities(
                ee_poses,
                timestamps=timestamps,
                min_dt=args.min_dt,
            )
            if "ee_velocity" in selected_features:
                computed_features[args.ee_velocity_key] = ee_velocities
            if "ee_acceleration" in selected_features:
                computed_features[args.ee_acceleration_key] = compute_episode_derivative(
                    ee_velocities,
                    timestamps=timestamps,
                    min_dt=args.min_dt,
                )

        for frame_idx, source_frame in tqdm(
            enumerate(source_frames),
            total=len(source_frames),
            desc=f"frames {start}:{end}",
            leave=False,
            unit="frame",
        ):
            frame = frame_for_output(source_frame, output_feature_keys)
            for key, values in computed_features.items():
                frame[key] = values[frame_idx]
            frame = coerce_frame_to_features(frame, features)
            add_frame(output_dataset, frame, task)

        save_episode(output_dataset, task)

    print(f"wrote dataset: root={args.output_root} repo_id={args.output_repo_id}")
    print(
        "added keys: "
        f"{', '.join(computed_key_names(args, selected_features))}, "
        f"timestamp_key={args.timestamp_key}"
    )


def computed_key_names(args: argparse.Namespace, selected_features: set[str]) -> list[str]:
    keys = []
    if "acceleration" in selected_features:
        keys.append(args.acceleration_key)
    if "ee_velocity" in selected_features:
        keys.append(args.ee_velocity_key)
    if "ee_acceleration" in selected_features:
        keys.append(args.ee_acceleration_key)
    return keys


if __name__ == "__main__":
    main()
