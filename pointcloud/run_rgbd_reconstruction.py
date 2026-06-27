from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = Path("pointcloud/config/rgbd_reconstruction/ft_test_data.yaml")
DEFAULT_CALIBRATION = Path("pointcloud/calibration_v2/camera_extrinsics.json")
DEFAULT_EE_CALIBRATION = Path("pointcloud/calibration_v2/ee_cam_d435/data_ee_cam.json")
DEFAULT_THIRD_PERSON_CALIBRATION = Path(
    "pointcloud/calibration_v2/third_person_cam_d455/data_third_person_cam.json"
)
DEFAULT_DATASET = Path("data/train_episode/Ft_test_data")
DEFAULT_AUGMENTED_JSON = "data_with_rgbd_extrinsics.json"
DEFAULT_SOURCE_JSON = "data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Thin orchestration layer for the RGB-D point-cloud reconstruction flow: "
            "export camera extrinsics, write per-frame RGB-D extrinsics, then reconstruct."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base YAML/JSON reconstruction config.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the underlying scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and effective config path without executing them.",
    )
    parser.add_argument(
        "--keep-effective-config",
        action="store_true",
        help="Save the effective config next to the output point cloud.",
    )

    calibration = parser.add_argument_group("calibration export")
    calibration.add_argument(
        "--calibration-mode",
        choices=("auto", "always", "skip"),
        default="auto",
        help="'auto' exports only when --calibration is missing.",
    )
    calibration.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Camera extrinsics JSON used by the RGB-D augmentation step.",
    )
    calibration.add_argument(
        "--ee-calibration",
        type=Path,
        default=DEFAULT_EE_CALIBRATION,
        help="Eye-in-hand calibration JSON for the wrist camera.",
    )
    calibration.add_argument(
        "--third-person-calibration",
        type=Path,
        default=DEFAULT_THIRD_PERSON_CALIBRATION,
        help="Eye-to-hand calibration JSON for the fixed camera.",
    )
    calibration.add_argument(
        "--calibration-method",
        default="tsai",
        choices=("tsai", "park", "horaud", "andreff", "daniilidis"),
        help="OpenCV calibrateHandEye method for calibration export.",
    )
    calibration.add_argument(
        "--write-npy",
        action="store_true",
        help="Also write T_ee_camera.npy / T_base_camera.npy during calibration export.",
    )

    augmentation = parser.add_argument_group("episode RGB-D extrinsics")
    augmentation.add_argument(
        "--skip-extrinsics",
        action="store_true",
        help="Skip writing data_with_rgbd_extrinsics.json.",
    )
    augmentation.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset folder containing the source episode JSON, colors, and depths.",
    )
    augmentation.add_argument(
        "--source-json",
        default=DEFAULT_SOURCE_JSON,
        help="Source episode JSON name inside --dataset-dir.",
    )
    augmentation.add_argument(
        "--augmented-json",
        default=None,
        help=(
            "Output augmented episode JSON name. Defaults to dataset.input_json from "
            "--config, then data_with_rgbd_extrinsics.json."
        ),
    )
    augmentation.add_argument(
        "--in-place",
        action="store_true",
        help="Let add_rgbd_extrinsics.py overwrite --source-json after creating a backup.",
    )
    augmentation.add_argument(
        "--robot-key",
        default="single",
        help="Robot key under ee_states used for wrist-camera pose.",
    )
    augmentation.add_argument(
        "--ee-pose-order",
        default="xyzw",
        choices=("xyzw", "wxyz"),
        help="Quaternion order in ee_states.<robot-key>.pose after xyz.",
    )
    augmentation.add_argument(
        "--depth-scale-m-per-unit",
        type=float,
        default=None,
        help="Depth PNG scale. When set, also overrides projection.depth_scale_m_per_unit.",
    )
    augmentation.add_argument(
        "--depth-aligned-to",
        default=None,
        help="Depth alignment frame recorded in the augmented JSON.",
    )
    augmentation.add_argument(
        "--no-file-check",
        action="store_true",
        help="Do not check that color/depth files referenced by the episode exist.",
    )

    reconstruction = parser.add_argument_group("reconstruction overrides")
    reconstruction.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Stop after the calibration and augmentation steps.",
    )
    reconstruction.add_argument("--output-dir", type=Path, default=None)
    reconstruction.add_argument("--output-name", default=None)
    reconstruction.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names, e.g. --cameras ee_cam third_person_cam, or 'all'.",
    )
    reconstruction.add_argument("--start-frame", type=int, default=None)
    reconstruction.add_argument("--end-frame", type=int, default=None)
    reconstruction.add_argument("--frame-step", type=int, default=None)
    reconstruction.add_argument("--max-frames", type=int, default=None)
    reconstruction.add_argument("--stride", type=int, default=None)
    reconstruction.add_argument("--depth-min", type=float, default=None)
    reconstruction.add_argument("--depth-max", type=float, default=None)
    reconstruction.add_argument("--voxel-size", type=float, default=None)
    reconstruction.add_argument("--max-points", type=int, default=None)
    reconstruction.add_argument(
        "--save-per-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override output.save_per_frame.",
    )
    reconstruction.add_argument(
        "--ascii",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override output.ascii.",
    )
    reconstruction.add_argument(
        "--view",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override visualization.view.",
    )
    reconstruction.add_argument(
        "--view-existing",
        type=Path,
        default=None,
        help="Only open an existing .ply through reconstruct_rgbd_episode.py.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ModuleNotFoundError:
            payload = load_simple_yaml(path)
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping/object: {path}")
    return payload


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple section/key YAML shape used by pointcloud configs."""
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line {line_number} in {path}: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        if indent == 0:
            if raw_value == "":
                current_section = {}
                root[key] = current_section
            else:
                root[key] = parse_simple_yaml_scalar(raw_value)
                current_section = None
            continue

        if indent == 2 and current_section is not None:
            current_section[key] = parse_simple_yaml_scalar(raw_value)
            continue

        raise ValueError(
            f"Unsupported YAML nesting at line {line_number} in {path}: {raw_line!r}. "
            "Install PyYAML for richer YAML configs."
        )
    return root


def parse_simple_yaml_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    lowered = value.lower()
    if lowered in {"null", "none", "~", ""}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if value.startswith(("[", "{", '"')) or value in {"[]", "{}"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def write_json_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def repo_relative(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    repo_candidate = REPO_ROOT / path
    if cwd_candidate.exists() and not repo_candidate.exists():
        return cwd_candidate
    return path


def exists_from_repo(path: Path) -> bool:
    if path.is_absolute():
        return path.exists()
    return (REPO_ROOT / path).exists()


def fs_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def path_for_cli(path: Path | str) -> str:
    return Path(path).as_posix()


def section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        value = {}
        config[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be a mapping/object")
    return value


def config_value(config: dict[str, Any], group: str, key: str, default: Any) -> Any:
    group_value = config.get(group)
    if isinstance(group_value, dict) and key in group_value:
        return group_value[key]
    return default


def split_cameras(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    cameras: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                cameras.append(item)
    if not cameras:
        return None
    if len(cameras) == 1 and cameras[0].lower() in {"all", "none", "null"}:
        return []
    return cameras


def set_if_not_none(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = path_for_cli(value) if isinstance(value, Path) else value


def build_effective_config(args: argparse.Namespace, base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    dataset_dir = effective_dataset_dir(args, base_config)
    augmented_json = effective_augmented_json(args, base_config)
    reconstruction_input_json = args.source_json if args.in_place else augmented_json

    dataset = section(config, "dataset")
    dataset["dir"] = path_for_cli(dataset_dir)
    dataset["input_json"] = reconstruction_input_json

    output = section(config, "output")
    set_if_not_none(output, "dir", args.output_dir)
    set_if_not_none(output, "name", args.output_name)
    set_if_not_none(output, "save_per_frame", args.save_per_frame)
    set_if_not_none(output, "ascii", args.ascii)

    frames = section(config, "frames")
    set_if_not_none(frames, "start", args.start_frame)
    set_if_not_none(frames, "end", args.end_frame)
    set_if_not_none(frames, "step", args.frame_step)
    set_if_not_none(frames, "max", args.max_frames)

    projection = section(config, "projection")
    cameras = split_cameras(args.cameras)
    if cameras is not None:
        projection["cameras"] = None if not cameras else cameras
    set_if_not_none(projection, "stride", args.stride)
    set_if_not_none(projection, "depth_min", args.depth_min)
    set_if_not_none(projection, "depth_max", args.depth_max)
    set_if_not_none(projection, "depth_scale_m_per_unit", args.depth_scale_m_per_unit)

    downsample = section(config, "downsample")
    set_if_not_none(downsample, "voxel_size", args.voxel_size)
    set_if_not_none(downsample, "max_points", args.max_points)

    visualization = section(config, "visualization")
    set_if_not_none(visualization, "view", args.view)
    set_if_not_none(visualization, "view_existing", args.view_existing)

    return config


def effective_dataset_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.dataset_dir is not None:
        return repo_relative(args.dataset_dir)
    return repo_relative(Path(config_value(config, "dataset", "dir", DEFAULT_DATASET)))


def effective_augmented_json(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.augmented_json:
        return args.augmented_json
    value = config_value(config, "dataset", "input_json", DEFAULT_AUGMENTED_JSON)
    return str(value or DEFAULT_AUGMENTED_JSON)


def effective_output_dir(config: dict[str, Any]) -> Path:
    return repo_relative(Path(config_value(config, "output", "dir", "outputs/rgbd_pointcloud")))


def should_export_calibration(args: argparse.Namespace) -> bool:
    calibration = repo_relative(args.calibration)
    if args.calibration_mode == "always":
        return True
    if args.calibration_mode == "skip":
        return False
    return not exists_from_repo(calibration)


def command_line(parts: list[str]) -> str:
    return " ".join(parts)


def run_step(name: str, command: list[str], *, dry_run: bool) -> None:
    print(f"\n==> {name}")
    print(command_line(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def calibration_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "pointcloud/calibration_v2/export_camera_extrinsics.py",
        "--ee-calibration",
        path_for_cli(repo_relative(args.ee_calibration)),
        "--third-person-calibration",
        path_for_cli(repo_relative(args.third_person_calibration)),
        "--output",
        path_for_cli(repo_relative(args.calibration)),
        "--method",
        args.calibration_method,
    ]
    if args.write_npy:
        command.append("--write-npy")
    return command


def augmentation_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    command = [
        args.python,
        "pointcloud/tools/add_rgbd_extrinsics.py",
        "--dataset-dir",
        path_for_cli(effective_dataset_dir(args, config)),
        "--calibration",
        path_for_cli(repo_relative(args.calibration)),
        "--input-json",
        args.source_json,
        "--output-json",
        effective_augmented_json(args, config),
        "--robot-key",
        args.robot_key,
        "--ee-pose-order",
        args.ee_pose_order,
    ]
    if args.depth_scale_m_per_unit is not None:
        command.extend(["--depth-scale-m-per-unit", str(args.depth_scale_m_per_unit)])
    if args.depth_aligned_to is not None:
        command.extend(["--depth-aligned-to", args.depth_aligned_to])
    if args.in_place:
        command.append("--in-place")
    if args.no_file_check:
        command.append("--no-file-check")
    return command


def reconstruction_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    return [
        args.python,
        "pointcloud/tools/reconstruct_rgbd_episode.py",
        "--config",
        path_for_cli(config_path),
    ]


def output_hint(config: dict[str, Any]) -> tuple[Path, Path]:
    output_dir = effective_output_dir(config)
    output_name = str(config_value(config, "output", "name", "episode_fused.ply"))
    return output_dir / output_name, output_dir / "summary.json"


def run_pipeline(args: argparse.Namespace, effective_config: dict[str, Any], config_path: Path) -> None:
    if should_export_calibration(args):
        run_step("Export camera extrinsics", calibration_command(args), dry_run=args.dry_run)
    else:
        print(f"\n==> Export camera extrinsics")
        print(f"Skipped; using existing calibration JSON: {path_for_cli(repo_relative(args.calibration))}")

    if not args.skip_extrinsics:
        run_step(
            "Write per-frame RGB-D extrinsics",
            augmentation_command(args, effective_config),
            dry_run=args.dry_run,
        )
    else:
        print("\n==> Write per-frame RGB-D extrinsics")
        print("Skipped by --skip-extrinsics.")

    if not args.skip_reconstruction:
        run_step(
            "Reconstruct fused point cloud",
            reconstruction_command(args, config_path),
            dry_run=args.dry_run,
        )
    else:
        print("\n==> Reconstruct fused point cloud")
        print("Skipped by --skip-reconstruction.")


def main() -> None:
    args = parse_args()
    args.config = repo_relative(args.config)
    base_config = load_config(fs_path(args.config))
    effective_config = build_effective_config(args, base_config)

    output_ply, summary_path = output_hint(effective_config)
    if args.keep_effective_config:
        config_path = effective_output_dir(effective_config) / "effective_rgbd_reconstruction.json"
        write_json_config(fs_path(config_path), effective_config)
        run_pipeline(args, effective_config, config_path)
    else:
        with tempfile.TemporaryDirectory(prefix="rgbd_reconstruction_") as temp_dir:
            config_path = Path(temp_dir) / "effective_rgbd_reconstruction.json"
            write_json_config(config_path, effective_config)
            print(f"Effective config: {config_path}")
            run_pipeline(args, effective_config, config_path)

    if args.dry_run:
        print("\nDry run complete.")
        return

    if not args.skip_reconstruction:
        print("\nRGB-D reconstruction flow complete.")
        print(f"Point cloud: {output_ply}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
