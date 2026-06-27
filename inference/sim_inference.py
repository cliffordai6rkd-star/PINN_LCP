"""Run dp_hirol image+FT policy inference and replay predicted EE poses in MuJoCo.

Default paths first honor environment variables, then try common Docker/host
locations discovered on this machine:
  - checkpoint: /home/hirol/code/data/0627/latest.ckpt
  - image dataset: /home/hirol/code/dp_hirol-main/data/zml_data/push_cube_ada_29ep_lerobot_v3_img
  - FT dataset: /home/hirol/code/data/push_cube_ada_29ep_lerobot_v3_ft

The policy predicts action.ee_pose_7d as [x, y, z, qx, qy, qz, qw]. This script
converts each predicted EE pose to a FR3 arm q via Pinocchio IK, then sends the q
sequence to the existing MuJoCo replay interface.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_path(env_name: str, *candidates: str | Path) -> Path:
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


DEFAULT_DP_ROOT = default_path(
    "DP_HIROL_ROOT",
    "/workspace/dp_hirol-main",
    "/workspace/code/dp_hirol-main",
    "/home/hirol/code/dp_hirol-main",
)
DEFAULT_CKPT = default_path(
    "DP_CKPT",
    "/workspace/data/0627/latest.ckpt",
    "/home/hirol/code/data/0627/latest.ckpt",
)
DEFAULT_IMAGE_DATASET = default_path(
    "DP_IMAGE_DATASET",
    "/workspace/data/push_cube_ada_29ep_lerobot_v3_img",
    DEFAULT_DP_ROOT / "data/zml_data/push_cube_ada_29ep_lerobot_v3_img",
    "/home/hirol/code/dp_hirol-main/data/zml_data/push_cube_ada_29ep_lerobot_v3_img",
)
DEFAULT_FT_DATASET = default_path(
    "DP_FT_DATASET",
    "/workspace/data/push_cube_ada_29ep_lerobot_v3_ft",
    "/home/hirol/code/data/push_cube_ada_29ep_lerobot_v3_ft",
)
DEFAULT_DINO_MODEL = default_path(
    "DP_DINO_MODEL",
    "/workspace/data/models/dinov2-base",
    DEFAULT_DP_ROOT / "data/models/dinov2-base",
    "/home/hirol/code/dp_hirol-main/data/models/dinov2-base",
)
DEFAULT_SIM_CONFIG = default_path(
    "SIM_CONFIG",
    REPO_ROOT / "config/sim_cfg/replay_test.yaml",
)
DEFAULT_CONFIG = default_path(
    "SIM_INFERENCE_CONFIG",
    REPO_ROOT / "inference/sim_inference.yaml",
)


log = logging.getLogger("sim_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer EE pose with a dp_hirol checkpoint, solve IK, and replay in MuJoCo."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config controlling paths, inference, IK, and playback settings.",
    )
    return parser.parse_args()


def default_runtime_config() -> dict:
    return {
        "paths": {
            "ckpt": DEFAULT_CKPT,
            "dp_root": DEFAULT_DP_ROOT,
            "image_dataset": DEFAULT_IMAGE_DATASET,
            "ft_dataset": DEFAULT_FT_DATASET,
            "image_model": DEFAULT_DINO_MODEL,
            "sim_config": DEFAULT_SIM_CONFIG,
            "save_q": None,
            "save_ee": None,
        },
        "inference": {
            "episode_index": 0,
            "episode_indices": None,
            "start_frame": 0,
            "max_frames": 120,
            "policy_stride": 1,
            "append_mode": "first",
            "sampler": "checkpoint",
            "state_dict": "ema_model",
            "device": "auto",
            "num_inference_steps": None,
            "sampler_step_kwargs": {},
            "seed": 42,
        },
        "data": {
            "force_local_lerobot_reader": True,
        },
        "execution": {
            "mode": "online",
            "no_play": False,
            "unchecked_ik": False,
            "ik_failure_policy": "clip",
            "use_dataset_initial_q": True,
            "fix_gripper": True,
            "fixed_gripper_ctrl": None,
            "fixed_gripper_q": None,
            "offline_playback_fps": 50.0,
        },
        "logging": {
            "level": "INFO",
        },
    }


def deep_update(base: dict, updates: dict | None) -> dict:
    if not updates:
        return base
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def config_path(value, default: Path | None = None) -> Path | None:
    if value is None or value == "":
        return default

    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path

    for candidate in (REPO_ROOT / path, REPO_ROOT.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate

    return REPO_ROOT / path


def validate_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def normalize_sampler(value) -> str:
    if value is None or value == "":
        return "checkpoint"
    sampler = str(value).strip().lower()
    if sampler in {"auto", "ckpt"}:
        sampler = "checkpoint"
    return validate_choice(
        "inference.sampler",
        sampler,
        ("checkpoint", "ddim", "ddpm"),
    )


def normalize_append_mode(value) -> str:
    mode = str(value).strip().lower()
    aliases = {
        "aggregate": "mean",
        "avg": "mean",
        "chunk_avg": "mean",
        "chunk_mean": "mean",
    }
    mode = aliases.get(mode, mode)
    return validate_choice(
        "inference.append_mode",
        mode,
        ("first", "chunk", "mean"),
    )


def parse_episode_config(inference: dict) -> tuple[int, object]:
    raw_episode_index = inference.get("episode_index", 0)
    raw_episode_indices = inference.get("episode_indices")

    if raw_episode_indices is None:
        if isinstance(raw_episode_index, str) and raw_episode_index.lower() == "all":
            return 0, "all"
        if isinstance(raw_episode_index, (list, tuple)):
            return 0, raw_episode_index

    return int(raw_episode_index), raw_episode_indices


def load_runtime_config(path: Path) -> argparse.Namespace:
    path = require_path(path, "sim inference config")
    raw = load_yaml(path) or {}
    cfg = deep_update(default_runtime_config(), raw)

    paths = cfg["paths"]
    inference = cfg["inference"]
    data_cfg = cfg["data"]
    execution = cfg["execution"]
    logging_cfg = cfg["logging"]
    episode_index, episode_indices = parse_episode_config(inference)
    sampler_step_kwargs = (
        inference.get("sampler_step_kwargs", inference.get("scheduler_step_kwargs", {}))
        or {}
    )
    if not isinstance(sampler_step_kwargs, dict):
        raise ValueError(
            "inference.sampler_step_kwargs must be a mapping, "
            f"got {type(sampler_step_kwargs).__name__}"
        )

    args = argparse.Namespace(
        config=path,
        ckpt=config_path(paths.get("ckpt"), DEFAULT_CKPT),
        dp_root=config_path(paths.get("dp_root"), DEFAULT_DP_ROOT),
        image_dataset=config_path(paths.get("image_dataset"), DEFAULT_IMAGE_DATASET),
        ft_dataset=config_path(paths.get("ft_dataset"), DEFAULT_FT_DATASET),
        image_model=config_path(paths.get("image_model"), DEFAULT_DINO_MODEL),
        sim_config=config_path(paths.get("sim_config"), DEFAULT_SIM_CONFIG),
        save_q=config_path(paths.get("save_q"), None),
        save_ee=config_path(paths.get("save_ee"), None),
        episode_index=episode_index,
        episode_indices=episode_indices,
        start_frame=int(inference.get("start_frame", 0)),
        max_frames=(
            None
            if inference.get("max_frames") is None
            else int(inference.get("max_frames"))
        ),
        policy_stride=int(inference.get("policy_stride", 1)),
        append_mode=normalize_append_mode(inference.get("append_mode", "first")),
        sampler=normalize_sampler(inference.get("sampler", "checkpoint")),
        state_dict=validate_choice(
            "inference.state_dict",
            str(inference.get("state_dict", "ema_model")),
            ("ema_model", "model"),
        ),
        device=validate_choice(
            "inference.device",
            str(inference.get("device", "auto")),
            ("auto", "cuda", "cpu"),
        ),
        num_inference_steps=inference.get("num_inference_steps"),
        sampler_step_kwargs=dict(sampler_step_kwargs),
        seed=int(inference.get("seed", 42)),
        force_local_lerobot_reader=bool(data_cfg.get("force_local_lerobot_reader", True)),
        execution_mode=validate_choice(
            "execution.mode",
            str(execution.get("mode", "online")),
            ("online", "offline"),
        ),
        no_play=bool(execution.get("no_play", False)),
        unchecked_ik=bool(execution.get("unchecked_ik", False)),
        ik_failure_policy=validate_choice(
            "execution.ik_failure_policy",
            str(execution.get("ik_failure_policy", "clip")),
            ("clip", "hold"),
        ),
        use_dataset_initial_q=bool(execution.get("use_dataset_initial_q", True)),
        fix_gripper=bool(execution.get("fix_gripper", True)),
        fixed_gripper_ctrl=execution.get("fixed_gripper_ctrl"),
        fixed_gripper_q=execution.get("fixed_gripper_q"),
        offline_playback_fps=(
            None
            if execution.get("offline_playback_fps") is None
            else float(execution.get("offline_playback_fps"))
        ),
        log_level=str(logging_cfg.get("level", "INFO")),
    )
    if args.num_inference_steps is not None:
        args.num_inference_steps = int(args.num_inference_steps)
    if args.offline_playback_fps is not None and args.offline_playback_fps <= 0:
        raise ValueError(
            f"execution.offline_playback_fps must be positive or null, got {args.offline_playback_fps}"
        )
    return args


def add_import_roots(dp_root: Path) -> None:
    for root in (REPO_ROOT, dp_root.expanduser().resolve()):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def require_path(path: str | Path, name: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def progress_iter(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def resolve_repo_path(path_like) -> str:
    if path_like is None:
        return path_like
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dp_checkpoint(path: Path):
    import dill
    import torch

    with path.open("rb") as f:
        try:
            return torch.load(
                f,
                map_location="cpu",
                pickle_module=dill,
                weights_only=False,
            )
        except TypeError:
            f.seek(0)
            return torch.load(f, map_location="cpu", pickle_module=dill)


def configure_ckpt_cfg(cfg, args: argparse.Namespace):
    """Patch training-container paths in the checkpoint config to local paths."""
    from omegaconf import OmegaConf

    cfg = copy.deepcopy(cfg)
    image_dataset = str(args.image_dataset.expanduser().resolve())
    ft_dataset = str(args.ft_dataset.expanduser().resolve())
    image_model = str(args.image_model.expanduser().resolve())

    for key, value in (
        ("dataset_path", image_dataset),
        ("ft_dataset_path", ft_dataset),
        ("image_model_name", image_model),
    ):
        if key in cfg:
            cfg[key] = value

    if OmegaConf.select(cfg, "task.dataset") is not None:
        cfg.task.dataset.dataset_path = image_dataset
        cfg.task.dataset.ft_dataset_path = ft_dataset
        cfg.task.dataset.preload_images = False
        cfg.task.dataset.load_result_add = "ram"
        cfg.task.dataset.local_files_only = True

    if OmegaConf.select(cfg, "policy.obs_encoder") is not None:
        cfg.policy.obs_encoder.image_model_name = image_model
        cfg.policy.obs_encoder.image_local_files_only = True

    if args.num_inference_steps is not None:
        cfg.policy.num_inference_steps = int(args.num_inference_steps)

    return cfg


def configure_policy_sampler(policy, args: argparse.Namespace) -> None:
    import inspect

    sampler = args.sampler
    if sampler == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
    elif sampler == "ddpm":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        policy.noise_scheduler = DDPMScheduler.from_config(policy.noise_scheduler.config)

    existing_kwargs = dict(getattr(policy, "kwargs", {}) or {})
    merged_kwargs = {**existing_kwargs, **args.sampler_step_kwargs}
    scheduler_step = policy.noise_scheduler.step
    signature = inspect.signature(scheduler_step)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        filtered_kwargs = merged_kwargs
        dropped_kwargs = {}
    else:
        valid_keys = set(signature.parameters)
        filtered_kwargs = {
            key: value
            for key, value in merged_kwargs.items()
            if key in valid_keys
        }
        dropped_kwargs = {
            key: value
            for key, value in merged_kwargs.items()
            if key not in valid_keys
        }

    if hasattr(policy, "kwargs"):
        policy.kwargs = filtered_kwargs

    scheduler_name = type(policy.noise_scheduler).__name__
    log.info(
        "Using %s sampler (%s), num_inference_steps=%s, sampler_step_kwargs=%s",
        sampler,
        scheduler_name,
        getattr(policy, "num_inference_steps", None),
        filtered_kwargs,
    )
    if dropped_kwargs:
        log.warning(
            "Dropped sampler_step_kwargs unsupported by %s.step: %s",
            scheduler_name,
            sorted(dropped_kwargs),
        )


def build_policy_and_dataset(args: argparse.Namespace):
    import hydra
    import torch

    ckpt = load_dp_checkpoint(args.ckpt)
    if not {"cfg", "state_dicts"}.issubset(ckpt.keys()):
        raise KeyError(
            f"Expected dp_hirol checkpoint keys ['cfg', 'state_dicts'], got {list(ckpt.keys())}"
        )

    cfg = configure_ckpt_cfg(ckpt["cfg"], args)
    log.info("Instantiating policy from checkpoint cfg...")
    policy = hydra.utils.instantiate(cfg.policy)

    state_key = args.state_dict
    if state_key not in ckpt["state_dicts"]:
        log.warning("state_dict %r not found, falling back to 'model'", state_key)
        state_key = "model"
    policy.load_state_dict(ckpt["state_dicts"][state_key], strict=True)
    configure_policy_sampler(policy, args)
    policy.eval()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    policy.to(device)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_cfg = copy.deepcopy(cfg.task.dataset)
    dataset_cfg.val_ratio = 0.0
    dataset_cfg.max_train_episodes = None
    dataset_cfg.preload_images = False
    dataset_cfg.load_result_add = "ram"
    if args.force_local_lerobot_reader:
        import diffusion_policy.common.lerobot_v3_io as lerobot_v3_io

        lerobot_v3_io._try_import_lerobot_dataset = lambda: None
        log.info("Forcing dp_hirol local parquet/PyAV LeRobot v3 reader.")

    log.info("Opening image and FT LeRobot v3 datasets...")
    dataset = hydra.utils.instantiate(dataset_cfg)
    return policy, dataset, cfg, device


def episode_frame_indices(dataset, episode_index: int, start_frame: int, max_frames: int | None):
    if episode_index < 0 or episode_index >= len(dataset.episode_ranges):
        raise IndexError(
            f"episode_index={episode_index} out of range [0, {len(dataset.episode_ranges) - 1}]"
        )
    episode_range = dataset.episode_ranges[episode_index]
    start = episode_range.start + max(0, int(start_frame))
    stop = episode_range.stop
    if max_frames is not None and max_frames > 0:
        stop = min(stop, start + int(max_frames))
    if start >= stop:
        raise ValueError(
            f"Empty episode slice: episode={episode_index}, start={start}, stop={stop}"
        )
    return np.arange(start, stop, dtype=np.int64)


def selected_episode_indices(dataset, args: argparse.Namespace) -> list[int]:
    raw_indices = args.episode_indices
    if raw_indices is None:
        indices = [int(args.episode_index)]
    elif isinstance(raw_indices, str) and raw_indices.lower() == "all":
        indices = list(range(len(dataset.episode_ranges)))
    elif isinstance(raw_indices, (list, tuple)):
        indices = [int(idx) for idx in raw_indices]
    else:
        indices = [int(raw_indices)]

    num_episodes = len(dataset.episode_ranges)
    for idx in indices:
        if idx < 0 or idx >= num_episodes:
            raise IndexError(
                f"episode index {idx} out of range [0, {num_episodes - 1}]"
            )
    return indices


def obs_indices_for_frame(dataset, frame_idx: int) -> np.ndarray:
    obs_steps = int(dataset.n_obs_steps or 1)
    episode_idx = int(dataset.episode_index[frame_idx])
    episode_start = dataset.episode_ranges[episode_idx].start
    return np.asarray(
        [
            max(episode_start, frame_idx - obs_steps + 1 + offset)
            for offset in range(obs_steps)
        ],
        dtype=np.int64,
    )


def build_obs_at_frame(dataset, frame_idx: int):
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.dataset.hirol_lerobot_v3_dataset import _safe_torch_from_numpy

    obs_indices = obs_indices_for_frame(dataset, frame_idx)
    obs_dict = {}
    frame_cache = {}

    for key in dataset.rgb_keys:
        expected_shape = tuple(dataset.shape_meta["obs"][key]["shape"])
        feature_name = dataset.image_feature_map[key]
        images = np.stack(
            [
                dataset._load_frame_feature(
                    frame_idx=int(obs_idx),
                    feature_name=feature_name,
                    expected_shape=expected_shape,
                    frame_cache=frame_cache,
                )
                for obs_idx in obs_indices
            ],
            axis=0,
        )
        obs_dict[key] = images.astype(np.float32, copy=False)

    for key in dataset.lowdim_keys:
        obs_dict[key] = dataset.lowdim_data[key][obs_indices, ...].astype(np.float32, copy=False)

    if dataset.use_ft:
        ft_data, ft_mask = dataset._sample_ft_sequence(obs_indices)
        obs_dict[dataset.ft_obs_key] = ft_data
        obs_dict[dataset.ft_mask_key] = ft_mask

    obs_tensors = dict_apply(obs_dict, _safe_torch_from_numpy)
    return {key: value.unsqueeze(0) for key, value in obs_tensors.items()}


def move_obs_to_device(obs_dict, device: str):
    import torch
    from diffusion_policy.common.pytorch_util import dict_apply

    return dict_apply(
        obs_dict,
        lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x,
    )


def predict_action_for_frame(policy, dataset, frame_idx: int, device: str) -> np.ndarray:
    import torch

    obs = build_obs_at_frame(dataset, int(frame_idx))
    obs = move_obs_to_device(obs, device)
    with torch.inference_mode():
        out = policy.predict_action(obs)
    action = out["action"][0].detach().cpu().numpy()
    if action.ndim != 2 or action.shape[-1] != 7:
        raise ValueError(f"Expected policy action shape [T, 7], got {action.shape}")
    return action.astype(np.float64, copy=False)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Invalid quaternion: {quat}")
    return quat / norm


def average_quat_xyzw(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    reference = normalize_quat_xyzw(quats[0])
    aligned = []
    for quat in quats:
        quat = normalize_quat_xyzw(quat)
        if float(np.dot(reference, quat)) < 0.0:
            quat = -quat
        aligned.append(quat)
    return normalize_quat_xyzw(np.mean(np.asarray(aligned), axis=0))


def aggregate_action_chunk(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[-1] != 7:
        raise ValueError(f"Expected action chunk shape [T, 7], got {action.shape}")
    position = np.mean(action[:, :3], axis=0)
    quat = average_quat_xyzw(action[:, 3:7])
    return np.concatenate([position, quat], axis=0)


def action_poses_from_chunk(action: np.ndarray, append_mode: str) -> np.ndarray:
    if append_mode == "first":
        return action[:1]
    if append_mode == "chunk":
        return action
    if append_mode == "mean":
        return aggregate_action_chunk(action)[None, :]
    raise ValueError(f"Unsupported append_mode: {append_mode}")


def action_pose_for_online(action: np.ndarray, append_mode: str) -> np.ndarray:
    if append_mode == "mean":
        return aggregate_action_chunk(action)
    return action[0]


def predict_ee_poses(policy, dataset, frame_indices: Iterable[int], device: str, args: argparse.Namespace):
    ee_poses = []
    policy_stride = max(1, int(args.policy_stride))
    selected_frames = list(frame_indices)[::policy_stride]
    episode_idx = int(dataset.episode_index[int(selected_frames[0])]) if selected_frames else -1
    episode_start = int(dataset.episode_ranges[episode_idx].start) if selected_frames else 0
    iterator = progress_iter(
        enumerate(selected_frames, start=1),
        total=len(selected_frames),
        desc=f"predict ep{episode_idx}",
    )
    has_progress_bar = hasattr(iterator, "set_postfix")

    for count, frame_idx in iterator:
        rel_frame = int(frame_idx) - episode_start
        if has_progress_bar:
            iterator.set_postfix(ep_frame=rel_frame)
        action = predict_action_for_frame(policy, dataset, int(frame_idx), device)

        ee_poses.extend(action_poses_from_chunk(action, args.append_mode))

        if not has_progress_bar and count % 10 == 0:
            log.info(
                "Predicted %d/%d policy windows, episode frame=%d",
                count,
                len(selected_frames),
                rel_frame,
            )

    if not ee_poses:
        raise RuntimeError("No EE poses were predicted.")
    return np.asarray(ee_poses, dtype=np.float64)

def read_initial_joint(dataset, frame_idx: int) -> np.ndarray | None:
    try:
        value = dataset.lerobot_dataset.get_column("observation.joint")[frame_idx]
    except Exception as exc:
        log.warning("Could not read observation.joint for IK seed: %s", exc)
        return None
    return np.asarray(value, dtype=np.float64).reshape(-1)


def make_sim_config(path: Path) -> dict:
    config = load_yaml(path)
    for key in ("model_path", "output_path", "ik_urdf_path"):
        if key in config and config[key] is not None:
            config[key] = resolve_repo_path(config[key])
    return config


def fixed_gripper_q(sim_config: dict, args: argparse.Namespace) -> np.ndarray | None:
    if not args.fix_gripper:
        return None
    q = args.fixed_gripper_q
    if q is None:
        q = sim_config.get(
            "q_reset_gripper",
            sim_config.get("ik_default_gripper_q", [-0.04, 0.04]),
        )
    return np.asarray(q, dtype=np.float64).reshape(-1)


def fixed_gripper_ctrl(sim_config: dict, args: argparse.Namespace) -> float | None:
    if not args.fix_gripper:
        return None
    value = args.fixed_gripper_ctrl
    if value is None:
        value = sim_config.get("teleop_gripper_open_ctrl")
    if value is None:
        value = sim_config.get("gripper_ctrl_range", [-0.11, 0.0])[0]
    return float(value)


def apply_fixed_gripper(sim, sim_config: dict, args: argparse.Namespace, set_q: bool = False) -> None:
    if not args.fix_gripper:
        return

    if set_q:
        q = fixed_gripper_q(sim_config, args)
        if q is not None:
            joint_names = sim_config.get(
                "ik_gripper_joint_names",
                ["gripper_left_joint", "gripper_right_joint"],
            )
            sim.set_joint_positions(q, joint_names=joint_names)

    ctrl = fixed_gripper_ctrl(sim_config, args)
    if ctrl is not None:
        sim.command_gripper(ctrl, step=False)


def solve_q_sequence(ee_poses: np.ndarray, dataset, first_frame_idx: int, sim_config: dict, args: argparse.Namespace):
    from sim_mujoco.ik_controller import PinocchioIKController

    ik = PinocchioIKController(sim_config)
    q_seed = None
    if args.use_dataset_initial_q:
        initial_arm_q = read_initial_joint(dataset, first_frame_idx)
        if initial_arm_q is not None and initial_arm_q.shape[0] == len(ik.arm_q_indices):
            q_seed = ik.compose_q(arm_q=initial_arm_q)
            log.info("Using dataset observation.joint at frame %d as IK seed.", first_frame_idx)

    if q_seed is None:
        q_seed = ik.compose_q(arm_q=sim_config.get("q_reset"))

    q_seq = []
    failed = 0
    for idx, pose in enumerate(ee_poses):
        q_arm, q_next_seed, ok, reasons, used_limited_q = ee_pose_to_arm_q(
            ik=ik,
            ee_pose=pose,
            q_seed=q_seed,
            unchecked_ik=args.unchecked_ik,
            failure_policy=args.ik_failure_policy,
        )
        if not ok:
            failed += 1
            log.warning("IK check failed at predicted step %d: %s", idx, "; ".join(reasons))
            if used_limited_q:
                log.info("Using clipped joint step at predicted step %d.", idx)

        q_seed = q_next_seed
        q_seq.append(q_arm)

    if failed:
        log.warning("IK produced %d checked failures over %d poses.", failed, len(ee_poses))
    return np.asarray(q_seq, dtype=np.float64)


def initial_full_q(dataset, first_frame_idx: int, sim_config: dict, ik, use_dataset_initial_q: bool):
    q_seed = None
    if use_dataset_initial_q:
        initial_arm_q = read_initial_joint(dataset, first_frame_idx)
        if initial_arm_q is not None and initial_arm_q.shape[0] == len(ik.arm_q_indices):
            q_seed = ik.compose_q(arm_q=initial_arm_q)
            log.info("Using dataset observation.joint at frame %d as initial q.", first_frame_idx)

    if q_seed is None:
        q_seed = ik.compose_q(arm_q=sim_config.get("q_reset"))
        log.info("Using sim_config q_reset as initial q.")
    return q_seed


def limit_q_step(ik, q_candidate: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
    arm_ref = ik.extract_arm_q(q_reference)
    arm_candidate = ik.extract_arm_q(q_candidate)
    delta = arm_candidate - arm_ref

    max_delta = float(ik.max_joint_delta)
    if max_delta > 0:
        delta = np.clip(delta, -max_delta, max_delta)

    max_norm = float(ik.max_joint_delta_norm)
    delta_norm = float(np.linalg.norm(delta))
    if max_norm > 0 and delta_norm > max_norm:
        delta = delta * (max_norm / max(delta_norm, 1e-12))

    gripper_q = ik.extract_gripper_q(q_reference) if ik.gripper_q_indices else None
    return ik.compose_q(arm_q=arm_ref + delta, gripper_q=gripper_q)


def ee_pose_to_arm_q(
    ik,
    ee_pose: np.ndarray,
    q_seed: np.ndarray,
    unchecked_ik: bool,
    failure_policy: str,
):
    position = np.asarray(ee_pose[:3], dtype=np.float64)
    quat_xyzw = normalize_quat_xyzw(ee_pose[3:7])
    q_candidate, info = ik.solve_pose(
        position=position,
        quaternion=quat_xyzw,
        quat_order="xyzw",
        q_seed=q_seed,
        return_info=True,
    )
    ok, reasons = ik.check_solution(q_candidate, q_reference=q_seed, info=info)
    if ok or unchecked_ik:
        return ik.extract_arm_q(q_candidate), q_candidate, ok, reasons, False

    if failure_policy == "clip":
        q_limited = limit_q_step(ik, q_candidate=q_candidate, q_reference=q_seed)
        return ik.extract_arm_q(q_limited), q_limited, False, reasons, True

    return ik.extract_arm_q(q_seed), q_seed, False, reasons, False


def run_online_follow(policy, dataset, frame_indices: np.ndarray, device: str, sim_config: dict, args: argparse.Namespace):
    import mujoco
    import mujoco.viewer

    from sim_mujoco.ik_controller import PinocchioIKController
    from sim_mujoco.mujocosim_inteface import MujocoSim_interface_fr3

    ik = PinocchioIKController(sim_config)
    sim = MujocoSim_interface_fr3(sim_config)
    sim.load_model()
    sim.print_model_info()
    sim.save_compiled_mjcf()

    q_seed = initial_full_q(
        dataset=dataset,
        first_frame_idx=int(frame_indices[0]),
        sim_config=sim_config,
        ik=ik,
        use_dataset_initial_q=args.use_dataset_initial_q,
    )
    sim.reset_arm_to_q(ik.extract_arm_q(q_seed))
    apply_fixed_gripper(sim, sim_config, args, set_q=True)

    fps = float(dataset.lerobot_dataset.fps) / max(1, int(args.policy_stride))
    control_dt = 1.0 / fps
    sim_dt = float(sim.model.opt.timestep)
    steps_per_frame = max(1, int(round(control_dt / sim_dt)))
    actual_dt = steps_per_frame * sim_dt
    selected_frames = frame_indices[:: max(1, int(args.policy_stride))]

    log.info(
        "Online follow: fps=%.2f, control_dt=%.4f, sim_dt=%.4f, steps_per_frame=%d",
        fps,
        control_dt,
        sim_dt,
        steps_per_frame,
    )

    ee_poses = []
    q_seq = []
    failed = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        for step_idx, frame_idx in enumerate(selected_frames):
            if not viewer.is_running():
                break

            action = predict_action_for_frame(policy, dataset, int(frame_idx), device)
            ee_pose = action_pose_for_online(action, args.append_mode)
            q_arm, q_next_seed, ok, reasons, used_limited_q = ee_pose_to_arm_q(
                ik=ik,
                ee_pose=ee_pose,
                q_seed=q_seed,
                unchecked_ik=args.unchecked_ik,
                failure_policy=args.ik_failure_policy,
            )
            if not ok:
                failed += 1
                log.warning(
                    "IK check failed at frame %d: %s",
                    int(frame_idx),
                    "; ".join(reasons),
                )
                if used_limited_q:
                    log.info("Using clipped joint step at frame %d.", int(frame_idx))
            q_seed = q_next_seed

            ee_poses.append(ee_pose)
            q_seq.append(q_arm)

            for _ in range(steps_per_frame):
                apply_fixed_gripper(sim, sim_config, args, set_q=False)
                sim.command_joint_pos(q_arm)

            apply_fixed_gripper(sim, sim_config, args, set_q=True)
            viewer.sync()
            if sim.quick_replay is False:
                time.sleep(actual_dt)

            if (step_idx + 1) % 10 == 0:
                log.info("Followed %d/%d data frames", step_idx + 1, len(selected_frames))

    if failed:
        log.warning("Online IK produced %d checked failures over %d frames.", failed, len(q_seq))
    log.info("Online follow finished %d/%d selected frames.", len(q_seq), len(selected_frames))
    return np.asarray(ee_poses, dtype=np.float64), np.asarray(q_seq, dtype=np.float64)


def play_q_sequence(q_seq: np.ndarray, sim_config: dict, fps: float, args: argparse.Namespace) -> None:
    from sim_mujoco.mujocosim_inteface import MujocoSim_interface_fr3

    viewer = MujocoSim_interface_fr3(sim_config)
    viewer.quick_replay = False
    viewer.load_model()
    viewer.print_model_info()
    viewer.save_compiled_mjcf()
    apply_fixed_gripper(viewer, sim_config, args, set_q=True)
    log.info("Offline replay at %.2f Hz (%d q targets).", fps, len(q_seq))
    viewer.play_joint_sequences([q_seq], dt=1.0 / float(fps), pause_between_episodes=False)


def save_array(path: Path | None, value: np.ndarray, name: str) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    log.info("Saved %s to %s", name, path)


def save_episode_arrays(
    path: Path | None,
    episode_outputs: list[tuple[int, np.ndarray, np.ndarray]],
    value_index: int,
    name: str,
) -> None:
    if path is None:
        return
    if len(episode_outputs) == 1:
        save_array(path, episode_outputs[0][value_index], name)
        return

    path = path.expanduser().resolve()
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            f"episode_{output[0]:03d}": output[value_index]
            for output in episode_outputs
        },
    )
    log.info("Saved %s for %d episodes to %s", name, len(episode_outputs), path)


def main() -> None:
    cli_args = parse_args()
    args = load_runtime_config(cli_args.config)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Loaded sim inference config from %s", args.config)

    args.dp_root = require_path(args.dp_root, "dp_hirol-main root")
    args.ckpt = require_path(args.ckpt, "checkpoint")
    args.image_dataset = require_path(args.image_dataset, "image LeRobot dataset")
    args.ft_dataset = require_path(args.ft_dataset, "FT LeRobot dataset")
    args.image_model = require_path(args.image_model, "DINO image model")
    args.sim_config = require_path(args.sim_config, "MuJoCo sim config")

    add_import_roots(args.dp_root)

    policy, dataset, _cfg, device = build_policy_and_dataset(args)
    sim_config = make_sim_config(args.sim_config)
    episode_outputs = []
    episode_indices = selected_episode_indices(dataset, args)
    log.info("Selected episode indices: %s", episode_indices)

    for loop_idx, episode_index in enumerate(episode_indices, start=1):
        frame_indices = episode_frame_indices(
            dataset,
            episode_index=episode_index,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        episode_start = int(dataset.episode_ranges[episode_index].start)
        episode_frame_start = int(frame_indices[0]) - episode_start
        episode_frame_end = int(frame_indices[-1]) - episode_start

        log.info(
            (
                "Running policy on episode=%d (%d/%d), "
                "episode_frames=%d..%d (%d frames), device=%s"
            ),
            episode_index,
            loop_idx,
            len(episode_indices),
            episode_frame_start,
            episode_frame_end,
            len(frame_indices),
            device,
        )
        if args.execution_mode == "online" and not args.no_play:
            ee_poses, q_seq = run_online_follow(policy, dataset, frame_indices, device, sim_config, args)
            expected_steps = len(frame_indices[:: max(1, int(args.policy_stride))])
        else:
            ee_poses = predict_ee_poses(policy, dataset, frame_indices, device, args)
            q_seq = solve_q_sequence(
                ee_poses=ee_poses,
                dataset=dataset,
                first_frame_idx=int(frame_indices[0]),
                sim_config=sim_config,
                args=args,
            )
            expected_steps = len(q_seq)

        episode_outputs.append((episode_index, ee_poses, q_seq))
        log.info("Prepared q sequence for episode %d with shape %s", episode_index, q_seq.shape)

        if not args.no_play and args.execution_mode == "offline":
            if args.offline_playback_fps is None:
                fps = float(getattr(dataset, "lerobot_dataset").fps)
                if args.append_mode in {"first", "mean"}:
                    fps = fps / max(1, int(args.policy_stride))
            else:
                fps = float(args.offline_playback_fps)
            play_q_sequence(q_seq, sim_config=sim_config, fps=fps, args=args)

        if args.execution_mode == "online" and not args.no_play and len(q_seq) < expected_steps:
            log.info("Viewer closed before episode finished; stopping episode loop.")
            break

    save_episode_arrays(args.save_ee, episode_outputs, 1, "predicted EE poses")
    save_episode_arrays(args.save_q, episode_outputs, 2, "IK q sequence")


if __name__ == "__main__":
    main()
