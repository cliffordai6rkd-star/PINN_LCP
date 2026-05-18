import os
import subprocess
import pyarrow as pa
import pyarrow.parquet as pq

UID = os.getuid()
os.environ["MPLCONFIGDIR"] = f"/tmp/matplotlib-{UID}"
os.environ["XDG_CONFIG_HOME"] = f"/tmp/rerun-config-{UID}"
os.environ["XDG_DATA_HOME"] = f"/tmp/rerun-data-{UID}"
os.environ.setdefault("RUST_LOG", "error")
for path in (os.environ["MPLCONFIGDIR"], os.environ["XDG_CONFIG_HOME"], os.environ["XDG_DATA_HOME"]):
    os.makedirs(path, exist_ok=True)

try:
    subprocess.run(
        ["rerun", "analytics", "disable"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
except FileNotFoundError:
    pass

import rerun as rr
import numpy as np
import torch
import pandas as pd

from pathlib import Path
import time
import json

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def build_rerun_blueprint():
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TimeSeriesView(
                        name="Force / Torque",
                        origin="signals/force",
                        contents="$origin/**",
                    ),
                    rrb.Spatial2DView(
                        name="Force Cursor",
                        origin="current/signals/force_cursor",
                        contents="$origin/**",
                    ),
                    rrb.TimeSeriesView(
                        name="Gripper",
                        origin="signals/gripper",
                        contents="$origin/**",
                    ),
                    column_shares=[2, 2, 1]
                ),
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="EE Matrix",
                        origin="world/ee",
                        contents="$origin/**",
                    ),
                    rrb.Spatial3DView(
                        name="EE Quat7",
                        origin="world/ee_quat7",
                        contents="$origin/**",
                    ),
                    column_shares=[1, 1],
                ),
                row_shares=[1, 2],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Current Wrist",
                    origin="current/cameras/wrist",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Current Side 1",
                    origin="current/cameras/side_1",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Current Side 2",
                    origin="current/cameras/side_2",
                    contents="$origin/**",
                ),
                row_shares=[1, 1, 1],
            ),
            column_shares=[2, 1],
        ),
        rrb.TimePanel(
            timeline="timestamp",
            state=rrb.PanelState.Collapsed,
        ),
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


class LeRobotv3Reader:
    def __init__(self, root, repo_id, video_backend="torchcodec"):
        self.root = Path(root)
        self.repo_id = repo_id
        self.video_backend = video_backend
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        return LeRobotDataset(
            repo_id=self.repo_id,
            root=self.root,
            video_backend=self.video_backend
        )

    def print_basic_info(self):
        print("root:", self.root)
        print("repo_id:", self.repo_id)
        print("len:", len(self.dataset))

    def print_features(self):
        print("features:")
        for key, spec in self.dataset.features.items():
            print(f"  {key}: {spec}")

    def print_sample(self, idx=0):
        sample = self.dataset[idx]

        print("sample idx:", idx)
        print("sample keys:", sample.keys())

        for key, value in sample.items():
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", type(value))
            print(f"  {key}: shape={shape}, dtype={dtype}")

        return sample

    def _to_numpy(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return value


class RerunLeRobotVisualizer:
    def __init__(self, reader, ee_pose_mode="matrix", log_force_window_detail=False):
        self.reader = reader
        self.ee_pose_mode = ee_pose_mode
        self.log_force_window_detail = log_force_window_detail
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []
        self.force_curve_cache = {}
        self.image_path_map = {
            "observation.images.wrist": "cameras/wrist",
            "observation.images.side_1": "cameras/side_1",
            "observation.images.side_2": "cameras/side_2",
        }
        self.current_image_path_map = {
            "observation.images.wrist": "current/cameras/wrist",
            "observation.images.side_1": "current/cameras/side_1",
            "observation.images.side_2": "current/cameras/side_2",
        }
        self.force_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.gripper_path_map = {
            "observation.ee_state.gripper": "signals/gripper/observation",
            "action.gripper_state": "signals/gripper/action",
        }

    def reset_episode_state(self):
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []

    def log_world_reference(self):
        rr.log(
            "world/origin",
            rr.Points3D([[0, 0, 0]], radii=0.01, colors=[[255, 255, 255]]),
        )
        rr.log(
            "world/reference_axes",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def log_frame(self, idx):
        sample = self.reader.dataset[idx]
        timestamp = float(sample["timestamp"])

        # Rerun stays on the dataset timestamp timeline; the annotation UI uses local_idx.
        rr.set_time("timestamp", duration=timestamp)

        self.log_images(sample)
        self.log_current_images(sample)
        self.log_force_window(sample)
        self.log_gripper(sample)
        self.log_ee_pose(sample)
        return timestamp

    def log_current_frame(self, idx, episode_index=None, start_idx=None, end_idx=None, local_idx=None):
        sample = self.reader.dataset[idx]
        timestamp = float(sample["timestamp"])
        self.log_current_images(sample)
        self.log_force_cursor(episode_index, start_idx, end_idx, local_idx)
        return timestamp

    def log_images(self, sample):
        for key, rerun_path in self.image_path_map.items():
            if key not in sample:
                continue

            image = self._image_to_uint8(sample[key])
            rr.log(rerun_path, rr.Image(image))

    def log_current_images(self, sample):
        for key, rerun_path in self.current_image_path_map.items():
            if key not in sample:
                continue

            image = self._image_to_uint8(sample[key])
            rr.log(rerun_path, rr.Image(image), static=True)

    def _image_to_uint8(self, value):
        image = self.reader._to_numpy(value)

        # LeRobot 通常是 CHW，需要转 HWC 给 rerun
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)

        # LeRobot 读出来的图像一般是 float32 [0, 1]，Rerun 用 uint8 更直观。
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)

        return image

    def log_force_window(self, sample):
        key = "observation.ft_window"
        if key not in sample:
            return

        ft = np.asarray(self.reader._to_numpy(sample[key]))

        # ft shape: [window_size, 6]
        # 这里忽略高频真实 timestamp，只按窗口内部顺序展开
        # 主曲线：每个 timestamp 只取窗口最后一个点，六个通道显示在同一张 Force 图中。
        current_ft = ft[-1]
        for ch_i in range(current_ft.shape[0]):
            name = self.force_names[ch_i] if ch_i < len(self.force_names) else f"ch_{ch_i}"
            rr.log(f"signals/force/{name}", rr.Scalars(float(current_ft[ch_i])))

        if not self.log_force_window_detail:
            return

        # 窗口细节：如果需要检查窗口方向/历史值，可在左侧树里打开 debug/force_window。
        for window_i in range(ft.shape[0]):
            for ch_i in range(ft.shape[1]):
                name = self.force_names[ch_i] if ch_i < len(self.force_names) else f"ch_{ch_i}"
                rr.log(
                    f"debug/force_window/{name}/past_{window_i}",
                    rr.Scalars(float(ft[window_i, ch_i])),
                )

    def log_force_cursor(self, episode_index, start_idx, end_idx, local_idx):
        if episode_index is None or start_idx is None or end_idx is None or local_idx is None:
            return

        timestamps, force_values = self._get_episode_force_curve(episode_index, start_idx, end_idx)
        if timestamps.size == 0 or force_values.size == 0:
            return

        local_idx = int(np.clip(local_idx, 0, len(timestamps) - 1))
        plot_image = self._render_force_cursor_plot(timestamps, force_values, local_idx)
        rr.log("current/signals/force_cursor/plot", rr.Image(plot_image), static=True)

    def _get_episode_force_curve(self, episode_index, start_idx, end_idx):
        cache_key = int(episode_index)
        if cache_key in self.force_curve_cache:
            return self.force_curve_cache[cache_key]

        parquet_paths = sorted((self.reader.root / "data").glob("**/*.parquet"))
        timestamp_chunks = []
        force_chunks = []
        for parquet_path in parquet_paths:
            df = pd.read_parquet(
                parquet_path,
                columns=["episode_index", "timestamp", "observation.ft_window"],
            )
            mask = df["episode_index"].to_numpy() == int(episode_index)
            row_indices = np.flatnonzero(mask)
            if row_indices.size == 0:
                continue

            timestamp_chunks.append(df["timestamp"].to_numpy(dtype=np.float64)[row_indices])
            force_chunks.extend(
                self._ft_window_current(df["observation.ft_window"].iloc[row_i])
                for row_i in row_indices
            )

        if not timestamp_chunks:
            timestamps = np.empty((0,), dtype=np.float64)
            force_values = np.empty((0, len(self.force_names)), dtype=np.float64)
        else:
            timestamps = np.concatenate(timestamp_chunks, axis=0)
            force_values = np.stack(force_chunks, axis=0)

        expected_len = max(0, int(end_idx) - int(start_idx))
        if expected_len and len(timestamps) != expected_len:
            timestamps = timestamps[:expected_len]
            force_values = force_values[:expected_len]

        self.force_curve_cache[cache_key] = (timestamps, force_values)
        return timestamps, force_values

    def _ft_window_current(self, ft_window):
        ft = np.asarray(list(ft_window), dtype=np.float64)
        return ft[-1]

    def _render_force_cursor_plot(self, timestamps, force_values, local_idx):
        figure = Figure(figsize=(7.2, 3.6), dpi=120)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots(len(self.force_names), 1, sharex=True)
        current_time = timestamps[local_idx]

        for ch_i, axis in enumerate(np.atleast_1d(axes)):
            values = force_values[:, ch_i]
            name = self.force_names[ch_i]
            axis.plot(timestamps, values, color="#2f6fbb", linewidth=0.9)
            axis.axvline(current_time, color="#d62728", linewidth=1.0)
            axis.scatter([current_time], [values[local_idx]], color="#d62728", s=14, zorder=3)
            axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
            axis.tick_params(axis="both", labelsize=7, length=2)
            axis.grid(True, alpha=0.25, linewidth=0.5)

        axes[-1].set_xlabel(f"timestamp (s), current={current_time:.3f}", fontsize=8)
        figure.tight_layout(pad=0.4)
        canvas.draw()

        image = np.asarray(canvas.buffer_rgba())
        return image[:, :, :3].copy()

    def log_gripper(self, sample):
        for key, rerun_path in self.gripper_path_map.items():
            if key not in sample:
                continue

            value = np.asarray(self.reader._to_numpy(sample[key])).reshape(-1)[0]
            rr.log(rerun_path, rr.Scalars(float(value)))

    def log_ee_pose(self, sample):
        key = "observation.ee_state.ee_pose"
        if key not in sample:
            return

        pose = np.asarray(self.reader._to_numpy(sample[key]))
        mode = self.ee_pose_mode

        if mode in ("matrix", "both"):
            self._log_ee_pose_matrix(pose)

        if mode in ("quat7", "both"):
            self._log_ee_pose_quat7(pose)

    def _log_ee_pose_matrix(self, pose):
        # Accept both old [4, 4] matrix poses and converted quat7 poses:
        # [x, y, z, qx, qy, qz, qw].
        if pose.shape == (4, 4):
            # print(f"rotaion type : rotation matrix")
            translation = pose[:3, 3]
            rotation = pose[:3, :3]
        else:
            # print(f"rotaion type : quaternion")
            quat7 = self._pose_to_quat7(pose)
            if quat7 is None:
                return
            translation = quat7[:3]
            rotation = self._quat_xyzw_to_rotation_matrix(quat7[3:7])

        if translation is None or rotation is None:
            return

        self.ee_trajectory.append(translation.copy())

        rr.log(
            "world/ee/current_position",
            rr.Points3D([translation], radii=0.005),
        )

        rr.log(
            "world/ee/trajectory",
            rr.Points3D(
                np.asarray(self.ee_trajectory),
                radii=0.002,
                colors=[[255, 255, 255]],
            ),
        )
        if len(self.ee_trajectory) >= 2:
            rr.log(
                "world/ee/trajectory_line",
                rr.LineStrips3D(
                    [np.asarray(self.ee_trajectory)],
                    radii=0.003,
                    colors=[[255, 255, 255]],
                ),
            )

        # 额外画出 EE 坐标轴：x 红、y 绿、z 蓝。
        axis_length = 0.08
        origins = np.repeat(translation.reshape(1, 3), 3, axis=0)
        vectors = np.stack(
            [
                rotation[:, 0] * axis_length,
                rotation[:, 1] * axis_length,
                rotation[:, 2] * axis_length,
            ],
            axis=0,
        )
        rr.log(
            "world/ee/current_axes",
            rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def _log_ee_pose_quat7(self, pose):
        # quat7 约定：[x, y, z, qx, qy, qz, qw]
        # 兼容旧数据: 如果当前数据仍是 4x4 矩阵，就先转换成 quat7。
        quat7 = self._pose_to_quat7(pose)
        if quat7 is None:
            return

        translation = quat7[:3]
        quat_xyzw = quat7[3:7]
        self.ee_quat7_trajectory.append(translation.copy())

        rr.log(
            "world/ee_quat7/current_position",
            rr.Points3D([translation], radii=0.005, colors=[[0, 0, 255]]),
        )

        rr.log(
            "world/ee_quat7/trajectory",
            rr.Points3D(
                np.asarray(self.ee_quat7_trajectory),
                radii=0.002,
                colors=[[255, 255, 255]],
            ),
            
        )
        if len(self.ee_quat7_trajectory) >= 2:
            rr.log(
                "world/ee_quat7/trajectory_line",
                rr.LineStrips3D(
                    [np.asarray(self.ee_quat7_trajectory)],
                    radii=0.003,
                    colors=[[0, 0, 255]],
                ),
            )

        rr.log(
            "world/ee_quat7/frame",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            ),
        )

    def _matrix_to_quat7(self, pose):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        quat_xyzw = self._rotation_matrix_to_quat_xyzw(rotation)
        return np.concatenate([translation, quat_xyzw])

    def _pose_to_quat7(self, pose):
        pose = np.asarray(pose)
        if pose.shape == (4, 4):
            return self._matrix_to_quat7(pose)

        quat7 = pose.reshape(-1).astype(np.float64)
        if quat7.shape[0] < 7:
            return None

        quat7 = quat7[:7]
        quat = quat7[3:7]
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat7[3:7] = quat / norm
        if quat7[6] < 0:
            quat7[3:7] = -quat7[3:7]
        return quat7

    def _quat_xyzw_to_rotation_matrix(self, quat_xyzw):
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm == 0:
            return np.eye(3, dtype=np.float64)
        qx, qy, qz, qw = quat / norm

        return np.array(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float64,
        )

    def _rotation_matrix_to_quat_xyzw(self, matrix):
        # 返回顺序为 xyzw，和 Rerun rr.Quaternion(xyzw=...) 对齐。
        m = np.asarray(matrix, dtype=np.float64)
        trace = np.trace(m)

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        if quat[3] < 0:
            quat = -quat
        return quat


def append_review_decision(path, episode_info, decision):
    record = {
        "episode_index": int(episode_info["episode_index"]),
        "decision": decision,
        "dataset_from_index": int(episode_info["dataset_from_index"]),
        "dataset_to_index": int(episode_info["dataset_to_index"]),
        "length": int(episode_info["length"]),
        "tasks": list(episode_info["tasks"]),
    }
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

def find_episode_parquet(root, episode_index):
    root = Path(root)
    episode_index = int(episode_index)

    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    matched_paths = []

    for parquet_path in parquet_paths:
        try:
            df_ep = pd.read_parquet(parquet_path, columns=["episode_index"])
        except Exception as exc:
            print(f"Skip parquet without episode_index: {parquet_path}, error={exc}")
            continue

        if (df_ep["episode_index"].to_numpy() == episode_index).any():
            matched_paths.append(parquet_path)

    if len(matched_paths) != 1:
        raise FileNotFoundError(
            f"expected one parquet containing episode {episode_index}, "
            f"found {len(matched_paths)}: {matched_paths}"
        )

    return matched_paths[0]


def write_contact_state_to_episode(root, episode_index, contact_state):
    root = Path(root)
    episode_index = int(episode_index)

    parquet_path = find_episode_parquet(root, episode_index)

    table = pq.read_table(parquet_path)

    if "episode_index" not in table.column_names:
        raise KeyError(f"episode_index column not found in {parquet_path}")

    episode_indices = table["episode_index"].to_numpy()
    mask = episode_indices == episode_index
    row_count = int(mask.sum())

    if row_count != contact_state.shape[0]:
        raise ValueError(
            f"contact_state length {contact_state.shape[0]} does not match "
            f"episode {episode_index} rows {row_count} in {parquet_path}"
        )

    contact_values = contact_state.detach().cpu().numpy().astype(np.float32).reshape(-1)

    full_contact = np.zeros((table.num_rows,), dtype=np.float32)

    # 如果原本已经有 contact_state，则先保留其他 episode 的旧标注
    col_name = "observation.contact_state"
    if col_name in table.column_names:
        old_values = table[col_name].to_pylist()
        restored = []
        for x in old_values:
            if isinstance(x, (list, tuple)):
                restored.append(float(x[0]))
            elif x is None:
                restored.append(0.0)
            else:
                restored.append(float(x))
        full_contact[:] = np.asarray(restored, dtype=np.float32)

    # 只覆盖当前 episode
    full_contact[mask] = contact_values

    # 关键：写成 float32 标量列，不写成 list[1]
    contact_column = pa.array(
        full_contact.astype(np.float32),
        type=pa.float32(),
    )

    if col_name in table.column_names:
        col_idx = table.column_names.index(col_name)
        table = table.set_column(col_idx, col_name, contact_column)
    else:
        table = table.append_column(col_name, contact_column)

    pq.write_table(table, parquet_path)

    print(
        f"Wrote {col_name} for episode {episode_index} "
        f"to {parquet_path}, rows={row_count}"
    )

def ensure_contact_state_feature(root):
    info_path = Path(root) / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)

    features = info.setdefault("features", {})
    features["observation.contact_state"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }

    with info_path.open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)


class ContactAnnotationUI:
    def __init__(self, reader, visualizer, episode_index, start_idx, end_idx):
        self.reader = reader
        self.visualizer = visualizer
        self.episode_index = episode_index
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.episode_len = end_idx - start_idx
        self.local_idx = 0
        self.pending_start = None
        self.intervals = []
        self.saved = False
        self.tk = None
        self.frame_label = None
        self.start_label = None
        self.interval_label = None
        self.frame_slider = None
        self.current_timestamp = None
        self.pending_frame_job = None

    def run(self):
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError("tkinter is not available in this environment") from exc

        self.tk = tk.Tk()
        self.tk.title(f"Contact annotation episode {self.episode_index}")
        self.tk.geometry("520x250")

        self.frame_label = tk.Label(self.tk, text="", font=("Arial", 14))
        self.frame_label.pack(pady=6)

        self.frame_slider = tk.Scale(
            self.tk,
            from_=0,
            to=max(0, self.episode_len - 1),
            orient=tk.HORIZONTAL,
            length=470,
            command=self.on_slider_change,
        )
        self.frame_slider.pack(pady=4)

        button_row = tk.Frame(self.tk)
        button_row.pack(pady=4)
        tk.Button(button_row, text="-", width=8, command=self.prev_frame).pack(side=tk.LEFT, padx=4)
        tk.Button(button_row, text="+", width=8, command=self.next_frame).pack(side=tk.LEFT, padx=4)

        mark_row = tk.Frame(self.tk)
        mark_row.pack(pady=4)
        tk.Button(mark_row, text="Mark start", width=12, command=self.mark_start).pack(side=tk.LEFT, padx=4)
        tk.Button(mark_row, text="Mark end", width=12, command=self.mark_end).pack(side=tk.LEFT, padx=4)

        save_row = tk.Frame(self.tk)
        save_row.pack(pady=4)
        tk.Button(save_row, text="Save", width=12, command=self.save).pack(side=tk.LEFT, padx=4)
        tk.Button(save_row, text="Close", width=12, command=self.close).pack(side=tk.LEFT, padx=4)

        self.start_label = tk.Label(self.tk, text="pending start: None")
        self.start_label.pack(pady=2)
        self.interval_label = tk.Label(self.tk, text="intervals: []", wraplength=430, justify=tk.LEFT)
        self.interval_label.pack(pady=2)

        self.show_current_frame()
        self.tk.mainloop()
        return self.saved

    def show_current_frame(self, update_slider=True):
        self.pending_frame_job = None
        global_idx = self.start_idx + self.local_idx
        self.current_timestamp = self.visualizer.log_current_frame(
            global_idx,
            episode_index=self.episode_index,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            local_idx=self.local_idx,
        )
        if update_slider and self.frame_slider is not None:
            self.frame_slider.set(self.local_idx)
        self.update_labels()

    def on_slider_change(self, value):
        new_idx = int(float(value))
        if new_idx == self.local_idx:
            self.update_labels()
            return
        self.local_idx = new_idx
        self.update_labels()
        self.schedule_frame_update()

    def schedule_frame_update(self, delay_ms=80):
        if self.tk is None:
            self.show_current_frame(update_slider=False)
            return
        if self.pending_frame_job is not None:
            self.tk.after_cancel(self.pending_frame_job)
        self.pending_frame_job = self.tk.after(
            delay_ms,
            lambda: self.show_current_frame(update_slider=False),
        )

    def update_labels(self):
        if self.frame_label is not None:
            timestamp_text = "None" if self.current_timestamp is None else f"{self.current_timestamp:.3f}"
            self.frame_label.config(
                text=(
                    f"episode frame: {self.local_idx} / {self.episode_len - 1}   "
                    f"global: {self.start_idx + self.local_idx}   "
                    f"timestamp: {timestamp_text}"
                )
            )
        if self.start_label is not None:
            self.start_label.config(text=f"pending start: {self.pending_start}")
        if self.interval_label is not None:
            self.interval_label.config(text=f"intervals: {self.intervals}")

    def prev_frame(self):
        self.cancel_pending_frame_update()
        self.local_idx = max(0, self.local_idx - 1)
        self.show_current_frame()

    def next_frame(self):
        self.cancel_pending_frame_update()
        self.local_idx = min(self.episode_len - 1, self.local_idx + 1)
        self.show_current_frame()

    def mark_start(self):
        self.pending_start = self.local_idx
        self.update_labels()

    def mark_end(self):
        if self.pending_start is None:
            print("Please mark contact start first.")
            return
        start = min(self.pending_start, self.local_idx)
        end = max(self.pending_start, self.local_idx)
        self.intervals.append((start, end))
        self.pending_start = None
        self.update_labels()

    def build_contact_state(self):
        contact_state = torch.zeros((self.episode_len, 1), dtype=torch.float32)
        for start, end in self.intervals:
            contact_state[start:end + 1] = 1.0
        return contact_state

    def save(self):
        contact_state = self.build_contact_state()
        ensure_contact_state_feature(self.reader.root)
        write_contact_state_to_episode(self.reader.root, self.episode_index, contact_state)
        self.saved = True
        print(f"Wrote contact_state for episode {self.episode_index}: {self.intervals}")
        self.update_labels()

        # 保存后自动关闭 Tkinter 标注窗口
        # annotation_ui.run() 返回后，外层 for 循环会自动进入下一个 episode
        self.close()

    def close(self):
        self.cancel_pending_frame_update()
        if self.tk is not None:
            self.tk.destroy()

    def cancel_pending_frame_update(self):
        if self.tk is not None and self.pending_frame_job is not None:
            self.tk.after_cancel(self.pending_frame_job)
        self.pending_frame_job = None

def get_visualizer_memory_gb():
    try:
        import psutil
    except ImportError:
        return None

    current_pid = os.getpid()
    memory_bytes = 0

    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = process.info["name"] or ""
            if process.info["pid"] == current_pid or name == "rerun":
                memory_bytes += process.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return memory_bytes / (1024 ** 3)


def wait_for_rerun_process_exit(timeout_s=5.0):
    try:
        import psutil
    except ImportError:
        time.sleep(1.0)
        return

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        has_rerun = False
        for process in psutil.process_iter(["name"]):
            try:
                if process.info["name"] == "rerun":
                    has_rerun = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not has_rerun:
            return
        time.sleep(0.2)


def close_rerun_ui(force=False):
    rr.disconnect()
    if force:
        subprocess.run(["pkill", "-x", "rerun"], check=False)
        wait_for_rerun_process_exit()


def run_episode_review(
    reader,
    visualizer,
    review_path,
    blueprint,
    playback_fps=30,
    realtime_stream=True,
    memory_limit_gb=20,
):
    episodes = reader.dataset.meta.episodes

    print("Episode review controls:")
    print("  Rerun will play each episode automatically.")
    print("  Tkinter annotation window will open after each episode is loaded.")
    print("  Click Save to write contact_state and move to the next episode.")

    for episode_info in episodes:
        episode_index = int(episode_info["episode_index"])
        start_idx = int(episode_info["dataset_from_index"])
        end_idx = int(episode_info["dataset_to_index"])
        length = int(episode_info["length"])

        visualizer.reset_episode_state()
        close_rerun_ui(force=False)

        rr.init(
            "lerobot_wipe_board",
            recording_id=f"episode_{episode_index:06d}",
            spawn=False,
            default_blueprint=blueprint,
        )
        rr.spawn(default_blueprint=blueprint, hide_welcome_screen=True)
        rr.send_blueprint(blueprint)
        visualizer.log_world_reference()

        force_close_ui = False

        try:
            print(f"\nPlaying episode {episode_index} [{start_idx}, {end_idx}) length={length}")

            for idx in range(start_idx, end_idx):
                tic = time.perf_counter()
                visualizer.log_frame(idx)

                if realtime_stream:
                    elapsed = time.perf_counter() - tic
                    time.sleep(max(0.0, 1.0 / playback_fps - elapsed))

            memory_gb = get_visualizer_memory_gb()
            force_close_ui = memory_gb is not None and memory_gb > memory_limit_gb

            if memory_gb is None:
                print("Memory monitor unavailable: psutil is not installed.")
            else:
                print(f"Visualizer memory used: {memory_gb:.2f} GB / limit {memory_limit_gb:.2f} GB")
                if force_close_ui:
                    print("Memory limit exceeded. Rerun UI will be force closed before the next episode.")

            # 不再等待终端输入 c，直接打开 Tkinter 标注窗口
            annotation_ui = ContactAnnotationUI(
                reader=reader,
                visualizer=visualizer,
                episode_index=episode_index,
                start_idx=start_idx,
                end_idx=end_idx,
            )

            saved = annotation_ui.run()

            if saved:
                print(f"Episode {episode_index} saved. Moving to next episode.")
            else:
                print(f"Episode {episode_index} closed without saving. Moving to next episode.")

        finally:
            close_rerun_ui(force=force_close_ui)

if __name__ == "__main__":
    import rerun as rr

    EE_POSE_MODE = "both"  # 可选："matrix"、"quat7"、"both"
    LOG_FORCE_WINDOW_DETAIL = False
    PLAYBACK_FPS = 60
    REALTIME_STREAM = False
    MEMORY_LIMIT_GB = 10
    REVIEW_PATH = "data/train_episode/wipe_board/wipe_board_lerobotv3/episode_review.jsonl"

    reader = LeRobotv3Reader(
        root="data/train_episode/wipe_board/wipe_board_lerobotv3",
        repo_id="local/h5_to_lerobot_v3",
        video_backend="torchcodec",
    )

    blueprint = build_rerun_blueprint()
    visualizer = RerunLeRobotVisualizer(
        reader,
        ee_pose_mode=EE_POSE_MODE,
        log_force_window_detail=LOG_FORCE_WINDOW_DETAIL,
    )
    run_episode_review(
        reader=reader,
        visualizer=visualizer,
        review_path=REVIEW_PATH,
        blueprint=blueprint,
        playback_fps=PLAYBACK_FPS,
        realtime_stream=REALTIME_STREAM,
        memory_limit_gb=MEMORY_LIMIT_GB,
    )
