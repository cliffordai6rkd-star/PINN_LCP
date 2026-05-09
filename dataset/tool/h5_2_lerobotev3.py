
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

EXAMPLE_SHAPE_META = {
    "fps": 30,
    "task": "your_task_name",
    "features": {
        "observation.images.wrist": {
            "type": "image",
            "shape": [224, 224, 3],
            "h5_path": "TODO/path/to/wrist_image_sequence",
        },
        "observation.state": {
            "type": "float32",
            "shape": ["TODO_state_dim"],
            "h5_paths": [
                "TODO/path/to/state_part_1",
                "TODO/path/to/state_part_2",
            ],
        },
        "action": {
            "type": "float32",
            "shape": ["TODO_action_dim"],
            "h5_path": "TODO/path/to/action_sequence",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert standalone .h5/.hdf5 episode files to LeRobot v3."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("tool/config/shape_meta.yaml"),
        help="Path to the conversion config YAML/JSON.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the H5 tree and exit.",
    )
    parser.add_argument(
        "--print-example-shape-meta",
        action="store_true",
        help="Print a minimal shape_meta template and exit.",
    )
    return parser.parse_args()

# shape_meta：读取“字段说明书”
def load_shape_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"shape_meta file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit("YAML shape_meta needs PyYAML. Use .json or install pyyaml.") from exc
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"shape_meta must be a mapping, got: {type(data).__name__}")
    return data


def load_h5py() -> Any:
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: h5py. Install it before reading H5 files.") from exc
    return h5py


def load_conversion_deps() -> tuple[Any, Any, Any]:
    h5py = load_h5py()
    try:
        import numpy as np  # type: ignore
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install numpy and lerobot before conversion."
        ) from exc
    return h5py, np, LeRobotDataset


def build_conversion_spec(shape_meta: Mapping[str, Any]) -> dict[str, Any]:
  
    raw_features = shape_meta.get("features")
    if not isinstance(raw_features, Mapping):
        raise ValueError("shape_meta must contain a 'features' mapping.")

    mappings = []
    lerobot_features = {}

    for lerobot_key, raw_spec in raw_features.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Feature {lerobot_key!r} spec must be a mapping.")

        h5_paths = normalize_h5_paths(str(lerobot_key), raw_spec)
        timestamp_path = raw_spec.get("timestamp_path")
        align = raw_spec.get("align", "index")
        window_size = int(raw_spec.get("window_size", 1))
        validate_mapping(str(lerobot_key), h5_paths, timestamp_path, align, window_size)

        feature_spec = {
            key: value
            for key, value in raw_spec.items()
            if key not in ("h5_path", "h5_paths", "timestamp_path", "align", "window_size")}
        normalize_feature_spec(feature_spec)

        lerobot_features[str(lerobot_key)] = feature_spec
        mappings.append(
            {
                "lerobot_key": str(lerobot_key),
                "h5_paths": h5_paths,
                "timestamp_path": timestamp_path,
                "align": align,
                "window_size": window_size,
                "feature_spec": feature_spec,
            }
        )

    return {
        "io": shape_meta.get("io", {}),
        "fps": int(shape_meta.get("fps", 30)),
        "task": str(shape_meta.get("task", "default_task")),
        "mappings": mappings,
        "lerobot_features": lerobot_features,
        "master_timestamp_path": shape_meta["master_timestamp_path"]
    }


def config_path(config: Mapping[str, Any], key: str, required: bool = True) -> Path | None:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")

    value = io_config.get(key)
    if value is None:
        if required:
            raise ValueError(f"config needs io.{key}")
        return None
    return Path(value)


def normalize_feature_spec(feature_spec: dict[str, Any]) -> None:
    """Normalize config feature metadata to what LeRobot validates against."""

    shape = feature_spec.get("shape")
    if isinstance(shape, list):
        feature_spec["shape"] = tuple(shape)


def config_bool(config: Mapping[str, Any], key: str, default: bool = False) -> bool:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")
    return bool(io_config.get(key, default))


def config_int(config: Mapping[str, Any], key: str, default: int | None = None) -> int | None:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")

    value = io_config.get(key, default)
    if value is None:
        return None
    return int(value)


def config_str(config: Mapping[str, Any], key: str, default: str) -> str:
    io_config = config.get("io", {})
    if not isinstance(io_config, Mapping):
        raise ValueError("config field 'io' must be a mapping.")
    return str(io_config.get(key, default))


def normalize_h5_paths(lerobot_key: str, raw_spec: Mapping[str, Any]) -> list[str]:
    # 一对一映射写法：
    #     h5_path: teleop/q_follower

    # 聚合映射写法：
    #     h5_paths:
    #       - teleop/q_follower
    #       - teleop/gripper_state
    h5_path = raw_spec.get("h5_path")
    h5_paths = raw_spec.get("h5_paths")

    if h5_path is not None and h5_paths is not None:
        raise ValueError(f"Feature {lerobot_key!r} can use h5_path or h5_paths, not both.")

    if h5_paths is None:
        if not isinstance(h5_path, str) or not h5_path:
            raise ValueError(f"Feature {lerobot_key!r} needs h5_path or h5_paths.")
        return [h5_path]

    if not isinstance(h5_paths, list) or not h5_paths:
        raise ValueError(f"Feature {lerobot_key!r} h5_paths must be a non-empty list.")

    for path in h5_paths:
        if not isinstance(path, str) or not path:
            raise ValueError(f"Feature {lerobot_key!r} h5_paths must only contain strings.")
    return h5_paths


def validate_mapping(
    lerobot_key: str,
    h5_paths: list[str],
    timestamp_path: Any,
    align: Any,
    window_size: int,
) -> None:
    if align not in ("index", "nearest", "nearest_past_window"):
        raise ValueError(f"Feature {lerobot_key!r} has unknown align mode: {align!r}")

    if align in ("nearest", "nearest_past_window"):
        if not isinstance(timestamp_path, str) or not timestamp_path:
            raise ValueError(f"Feature {lerobot_key!r} align={align!r} needs timestamp_path.")

    if align == "nearest_past_window":
        if len(h5_paths) != 1:
            raise ValueError(f"Feature {lerobot_key!r} nearest_past_window supports one h5_path only.")
        if window_size <= 0:
            raise ValueError(f"Feature {lerobot_key!r} window_size must be positive.")

class H5Dataset:
    def __init__(
        self,
        input_path: Path,
        *,
        h5py: Any,
        np: Any | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.h5py = h5py
        self.np = np
        self.max_episodes = max_episodes

    def files(self) -> list[Path]:
        if self.input_path.is_file():
            h5_files = [self.input_path]
        else:
            h5_files = sorted(self.input_path.glob("*.h5")) + sorted(self.input_path.glob("*.hdf5"))

        if not h5_files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found under {self.input_path}")

        if self.max_episodes is not None:
            h5_files = h5_files[: self.max_episodes]
        return h5_files

    def inspect(self):
        # 打印 H5 树结构
        for h5_path in self.files():
            # print(f"\n# {h5_path}", flush=True)
            # print("before open", flush=True)
            with self.h5py.File(h5_path, "r") as h5_file:
                # print("after open", flush=True)
                self._print_node(h5_file)
                # print("after print node", flush=True)

    def open_episode(self, h5_path: Path):
        # 用法：
        #     with h5_dataset.open_episode(path) as h5_file:

        return self.h5py.File(h5_path, "r")

    def build_episode_cache(self, h5_file, mappings, master_timestamp_path, h5_path):
        """Cache dataset handles and timestamp arrays for one opened episode."""

        dataset_cache = {}
        timestamp_cache = {}

        all_dataset_paths = {master_timestamp_path}
        all_timestamp_paths = {master_timestamp_path}
        for mapping in mappings:
            all_dataset_paths.update(mapping["h5_paths"])
            timestamp_path = mapping.get("timestamp_path")
            if timestamp_path:
                all_timestamp_paths.add(timestamp_path)

        for field_path in all_dataset_paths | all_timestamp_paths:
            dataset_cache[field_path] = self._dataset(h5_file, field_path, h5_path)

        for timestamp_path in all_timestamp_paths:
            timestamp_cache[timestamp_path] = dataset_cache[timestamp_path][:]

        return {
            "datasets": dataset_cache,
            "timestamps": timestamp_cache,
        }

    def clear_episode_cache(self, cache) -> None:
        """Release per-episode cached arrays and H5 dataset handles."""

        if not cache:
            return
        cache.get("datasets", {}).clear()
        cache.get("timestamps", {}).clear()
        cache.clear()

    def episode_length(self, h5_file, master_timestamp_path, h5_path, cache=None):
        # 主timestamp长度即为episode长度
        master_ts = self._timestamp_array(h5_file, master_timestamp_path, h5_path, cache)
        return int(master_ts.shape[0])

    def read_frame(self, h5_file, frame_idx, mappings, h5_path, master_timestamp_path, cache=None):
        # 从 H5 里读出一帧，返回 LeRobotDataset.add_frame 需要的字典
        master_ts = self._timestamp_array(h5_file, master_timestamp_path, h5_path, cache)
        target_t = master_ts[frame_idx]

        frame = {}

        for mapping in mappings:
            key = mapping["lerobot_key"]
            frame[key] = self._read_mapped_value(
                h5_file=h5_file,
                mapping=mapping,
                frame_idx=frame_idx,
                target_t=target_t,
                h5_path=h5_path,
                cache=cache,
            )

        return frame


    def _nearest_past_window_indices(self, timestamps, target_t, window_size):
        nearest_idx = int(self.np.argmin(self.np.abs(timestamps - target_t)))
        indices = self.np.arange(nearest_idx - window_size + 1, nearest_idx + 1)
        return self.np.clip(indices, 0, len(timestamps) - 1)

    def _read_nearest_past_window(self, h5_file, h5_field_path, timestamp_path, target_t, h5_path, window_size, cache=None):
        timestamps = self._timestamp_array(h5_file, timestamp_path, h5_path, cache)
        values = self._dataset_cached(h5_file, h5_field_path, h5_path, cache)

        indices = self._nearest_past_window_indices(timestamps, target_t, window_size)
        # h5py fancy indexing requires strictly increasing indices, but left-padding
        # intentionally creates duplicates such as [0, 0, 0, 0]. Read one by one.
        window = [values[int(index)] for index in indices]
        return self.np.asarray(window).astype("float32")




    def _read_mapped_value(self, h5_file, mapping, frame_idx, target_t, h5_path, cache=None):
        align = mapping.get("align", "index")

        if align == "index":
            return self._read_paths_at_index(
                h5_file, mapping["h5_paths"], frame_idx, h5_path, mapping["feature_spec"], cache
            )
    
        if align == "nearest":
            source_idx = self._nearest_index(
                h5_file, mapping["timestamp_path"], target_t, h5_path, cache
            )
            return self._read_paths_at_index(
                h5_file, mapping["h5_paths"], source_idx, h5_path, mapping["feature_spec"], cache
            )
    
        if align == "nearest_past_window":
            return self._read_nearest_past_window(
                h5_file=h5_file,
                h5_field_path=mapping["h5_paths"][0],
                timestamp_path=mapping["timestamp_path"],
                target_t=target_t,
                window_size=mapping["window_size"],
                h5_path=h5_path,
                cache=cache,
            )
    
        raise ValueError(f"Unknown align mode: {align}")
    

    def _nearest_index(self, h5_file, timestamp_path, target_t, h5_path, cache=None):
        timestamps = self._timestamp_array(h5_file, timestamp_path, h5_path, cache)
        return int(self.np.argmin(self.np.abs(timestamps - target_t)))

    def _read_paths_at_index(self, h5_file, h5_paths, source_idx, h5_path, feature_spec, cache=None):
        values = [
            self._read_value(
                h5_file=h5_file,
                h5_field_path=h5_field_path,
                frame_idx=source_idx,
                h5_path=h5_path,
                cache=cache,
            )
            for h5_field_path in h5_paths]
    
        if len(values) == 1:
            return self._coerce_single_value(values[0], feature_spec)
    
        return self._concat_values(values)


    def _read_value(self, h5_file: Any, h5_field_path: str, frame_idx: int, h5_path: Path, cache=None) -> Any:
        # 从一个 H5 dataset 里取一个值。
        # 标量 dataset：直接取 dataset[()]；
        # 时序 dataset：取 dataset[frame_idx]。
        # 后面如果需要对图片转 RGB、换通道、拼 state、裁剪 action，
        # 可以从这里拆出 `_read_image_value()` / `_read_state_value()

        dataset = self._dataset_cached(h5_file, h5_field_path, h5_path, cache)
        if len(dataset.shape) == 0:
            value = dataset[()]
        else:
            value = dataset[frame_idx]
        return self._to_lerobot_value(value)

    def _coerce_single_value(self, value, feature_spec):
        if self.np is None:
            return value

        if not isinstance(feature_spec, Mapping):
            return value

        dtype_name = feature_spec.get("dtype")
        if dtype_name in ("image", "video"):
            return value

        array = self.np.asarray(value)
        shape = feature_spec.get("shape")
        if (shape == [1] or shape == (1,)) and array.shape == ():
            array = array.reshape(1)

        return array.astype(self._dtype_from_feature(feature_spec), copy=False)

    def _concat_values(self, values: list[Any]) -> Any:
        """把多个低维 H5 字段聚合成一个一维向量。

        例子：
            teleop/q_follower      shape=(7,)
            teleop/gripper_state   shape=()
            -> observation.state   shape=(8,)

        注意：这个函数默认把多维数组 flatten 后再拼接，所以只建议用于
        state/action 这类低维字段，不建议用于图片。
        """

        if self.np is None:
            raise RuntimeError("Aggregating h5_paths requires numpy.")

        arrays = []
        for value in values:
            array = self.np.asarray(value)
            if array.shape == ():
                array = array.reshape(1)
            elif array.ndim > 1:
                array = array.reshape(-1)
            arrays.append(array)

        return self.np.concatenate(arrays, axis=0).astype("float32")

    def _dtype_from_feature(self, feature_spec):
        dtype_name = feature_spec.get("dtype", feature_spec.get("type", "float32"))
        try:
            return self.np.dtype(dtype_name)
        except TypeError:
            return self.np.float32

    def _dataset(self, h5_file, h5_field_path, h5_path) -> Any:
        if h5_field_path not in h5_file:
            raise KeyError(f"{h5_field_path!r} not found in {h5_path}")

        dataset = h5_file[h5_field_path]
        if not isinstance(dataset, self.h5py.Dataset):
            raise TypeError(f"{h5_field_path!r} is not a dataset in {h5_path}")
        return dataset

    def _dataset_cached(self, h5_file, h5_field_path, h5_path, cache=None):
        if cache is not None and h5_field_path in cache["datasets"]:
            return cache["datasets"][h5_field_path]
        return self._dataset(h5_file, h5_field_path, h5_path)

    def _timestamp_array(self, h5_file, timestamp_path, h5_path, cache=None):
        if cache is not None and timestamp_path in cache["timestamps"]:
            return cache["timestamps"][timestamp_path]
        return self._dataset(h5_file, timestamp_path, h5_path)[:]

    def _to_lerobot_value(self, value: Any) -> Any:
        if self.np is None:
            return value

        value = self.np.asarray(value)
        if value.shape == ():
            return value.item()
        return value

    def _print_node(self, node: Any, prefix: str = "") -> None:
        for key in sorted(node.keys()):
            item = node[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, self.h5py.Dataset):
                print(f"{path}: dataset shape={item.shape} dtype={item.dtype}{self._format_attrs(item.attrs)}")
            elif isinstance(item, self.h5py.Group):
                print(f"{path}/: group{self._format_attrs(item.attrs)}")
                self._print_node(item, path)

    @staticmethod
    def _format_attrs(attrs: Any) -> str:
        if len(attrs) == 0:
            return ""
        parts = [f"{key}={attrs[key]!r}" for key in sorted(attrs.keys())]
        return " attrs={" + ", ".join(parts) + "}"



class LeRobotV3Dataset:
    def __init__(
        self,
        LeRobotDataset,
        *,
        repo_id: str,
        root: Path,
        fps: int,
        features: Mapping[str, Any],
        no_videos: bool,
    ) -> None:
        create_kwargs = {
            "repo_id": repo_id,
            "root": root,
            "fps": fps,
            "features": dict(features),
            "use_videos": not no_videos,
            "video": not no_videos,
        }
        supported_kwargs = filter_supported_kwargs(LeRobotDataset.create, create_kwargs)
        self.dataset = LeRobotDataset.create(**supported_kwargs)

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        frame = dict(frame)
        frame.setdefault("task", task)
        kwargs = {"frame": frame, "task": task}
        try:
            self.dataset.add_frame(**filter_supported_kwargs(self.dataset.add_frame, kwargs))
        except TypeError:
            self.dataset.add_frame(frame)

    def save_episode(self, task: str) -> None:
        kwargs = {"task": task}
        try:
            self.dataset.save_episode(**filter_supported_kwargs(self.dataset.save_episode, kwargs))
        except TypeError:
            self.dataset.save_episode()

    def push_to_hub(self) -> None:
        if not hasattr(self.dataset, "push_to_hub"):
            raise AttributeError("Installed LeRobotDataset has no push_to_hub().")
        self.dataset.push_to_hub()


def filter_supported_kwargs(callable_obj: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """兼容不同 LeRobot 版本：只传当前函数支持的参数。"""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}



# 主流程：H5Dataset 读，LeRobotV3Dataset 写

def run_inspect(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    input_path = config_path(config, "input")
    h5py = load_h5py()
    h5_dataset = H5Dataset(
        input_path,
        h5py=h5py,
        max_episodes=config_int(config, "max_episodes"),
    )
    h5_dataset.inspect()

def run_conversion(args: argparse.Namespace) -> None:
    config = load_shape_meta(args.config)
    spec = build_conversion_spec(config)
    h5py, np, LeRobotDataset = load_conversion_deps()

    h5_dataset = H5Dataset(
        config_path(config, "input"),
        h5py=h5py,
        np=np,
        max_episodes=config_int(config, "max_episodes"),
    )
    lerobot_dataset = LeRobotV3Dataset(
        LeRobotDataset,
        repo_id=config_str(config, "repo_id", "local/h5_to_lerobot_v3"),
        root=config_path(config, "output"),
        fps=spec["fps"],
        features=spec["lerobot_features"],
        no_videos=config_bool(config, "no_videos"),
    )

    h5_files = h5_dataset.files()
    episode_iter = tqdm(h5_files, desc="episodes", unit="episode")
    for h5_path in episode_iter:
        episode_iter.set_postfix_str(h5_path.name)
        with h5_dataset.open_episode(h5_path) as h5_file:
            cache = None
            try:
                cache = h5_dataset.build_episode_cache(
                    h5_file,
                    spec["mappings"],
                    spec["master_timestamp_path"],
                    h5_path,
                )
                episode_length = h5_dataset.episode_length(
                    h5_file,
                    spec["master_timestamp_path"],
                    h5_path,
                    cache,
                )
                frame_iter = tqdm(
                    range(episode_length),
                    desc=f"frames {h5_path.name}",
                    unit="frame",
                    leave=False,
                )
                for frame_idx in frame_iter:
                    frame = h5_dataset.read_frame(
                        h5_file,
                        frame_idx,
                        spec["mappings"],
                        h5_path,
                        spec["master_timestamp_path"],
                        cache,
                    )
                    lerobot_dataset.add_frame(frame, task=spec["task"])
            finally:
                h5_dataset.clear_episode_cache(cache)
        lerobot_dataset.save_episode(task=spec["task"])

    if config_bool(config, "push_to_hub"):
        lerobot_dataset.push_to_hub()


def main() -> None:
    args = parse_args()

    if args.print_example_shape_meta:
        print(json.dumps(EXAMPLE_SHAPE_META, indent=2, ensure_ascii=False))
        return

    if args.inspect_only:
        run_inspect(args)
        return

    run_conversion(args)


if __name__ == "__main__":
    main()
