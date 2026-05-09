import rerun as rr
import numpy as np
from pathlib import Path
import time
import json
import os
import subprocess
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
                    rrb.TimeSeriesView(
                        name="Gripper",
                        origin="signals/gripper",
                        contents="$origin/**",
                    ),
                    column_shares=[2, 1],
                ),
                rrb.Spatial3DView(
                    name="EE Pose",
                    origin="world",
                    contents="$origin/**",
                ),
                row_shares=[1, 2],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Wrist",
                    origin="cameras/wrist",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Side 1",
                    origin="cameras/side_1",
                    contents="$origin/**",
                ),
                rrb.Spatial2DView(
                    name="Side 2",
                    origin="cameras/side_2",
                    contents="$origin/**",
                ),
                row_shares=[1, 1, 1],
            ),
            column_shares=[2, 1],
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

    def show_image(self, idx=2, key="observation.images.wrist"):
        import matplotlib.pyplot as plt

        sample = self.dataset[idx]
        image = sample[key]
        image = self._to_numpy(image)

        # LeRobot 读出来的图像可能是 CHW，也可能是 HWC
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)

        # 如果是 float 且范围是 [0, 1]，imshow 可以直接显示
        # 如果是 uint8 [0, 255]，imshow 也可以直接显示
        plt.imshow(image)
        plt.title(f"{key} @ idx={idx}")
        plt.axis("off")
        # plt.show()
        plt.savefig("/workspace/debug_frame2.png")

class RerunLeRobotVisualizer:
    def __init__(self, reader, ee_pose_mode="matrix"):
        self.reader = reader
        self.ee_pose_mode = ee_pose_mode
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []
        self.image_path_map = {
            "observation.images.wrist": "cameras/wrist",
            "observation.images.side_1": "cameras/side_1",
            "observation.images.side_2": "cameras/side_2",
        }
        self.gripper_path_map = {
            "observation.ee_state.gripper": "signals/gripper/observation",
            "action.gripper_state": "signals/gripper/action",
        }

    def reset_episode_state(self):
        self.ee_trajectory = []
        self.ee_quat7_trajectory = []

    def log_frame(self, idx):
        sample = self.reader.dataset[idx]
        timestamp = float(sample["timestamp"])

        rr.set_time("frame_index", sequence=idx)
        rr.set_time("timestamp", timestamp=timestamp)

        self.log_images(sample)
        self.log_force_window(sample)
        self.log_gripper(sample)
        self.log_ee_pose(sample)

    def log_images(self, sample):
        for key, rerun_path in self.image_path_map.items():
            if key not in sample:
                continue

            image = self.reader._to_numpy(sample[key])

            # LeRobot 通常是 CHW，需要转 HWC 给 rerun
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.transpose(1, 2, 0)

            # LeRobot 读出来的图像一般是 float32 [0, 1]，Rerun 用 uint8 更直观。
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0)
                image = (image * 255.0).astype(np.uint8)

            rr.log(rerun_path, rr.Image(image))

    def log_force_window(self, sample):
        key = "observation.ft_window"
        if key not in sample:
            return

        ft = np.asarray(self.reader._to_numpy(sample[key]))

        # ft shape: [window_size, 6]
        # 这里忽略高频真实 timestamp，只按窗口内部顺序展开
        force_names = ["fx", "fy", "fz", "tx", "ty", "tz"]

        # 主曲线：每个 timestamp 只取窗口最后一个点，六个通道显示在同一张 Force 图中。
        current_ft = ft[-1]
        for ch_i in range(current_ft.shape[0]):
            name = force_names[ch_i] if ch_i < len(force_names) else f"ch_{ch_i}"
            rr.log(f"signals/force/{name}", rr.Scalars(float(current_ft[ch_i])))

        # 窗口细节：如果需要检查窗口方向/历史值，可在左侧树里打开 force_window。
        for window_i in range(ft.shape[0]):
            for ch_i in range(ft.shape[1]):
                name = force_names[ch_i] if ch_i < len(force_names) else f"ch_{ch_i}"
                rr.log(
                    f"signals/force_window/{name}/past_{window_i}",
                    rr.Scalars(float(ft[window_i, ch_i])),
                )

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
        # pose shape: [4, 4]
        if pose.shape != (4, 4):
            return

        translation = pose[:3, 3]
        rotation = pose[:3, :3]
        self.ee_trajectory.append(translation.copy())

        rr.log(
            "world/ee/current_position",
            rr.Points3D([translation], radii=0.015),
        )

        rr.log(
            "world/ee/trajectory",
            rr.Points3D(
                np.asarray(self.ee_trajectory),
                radii=0.006,
                colors=[[255, 255, 0]],
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
        # 如果当前数据仍是 4x4 矩阵，就先从矩阵转换出 xyz + quaternion 用于对照检查。
        if pose.shape == (4, 4):
            quat7 = self._matrix_to_quat7(pose)
        else:
            quat7 = pose.reshape(-1)

        if quat7.shape[0] < 7:
            return

        translation = quat7[:3]
        quat_xyzw = quat7[3:7]
        self.ee_quat7_trajectory.append(translation.copy())

        rr.log(
            "world/ee_quat7/current_position",
            rr.Points3D([translation], radii=0.012, colors=[[255, 0, 255]]),
        )

        rr.log(
            "world/ee_quat7/trajectory",
            rr.Points3D(
                np.asarray(self.ee_quat7_trajectory),
                radii=0.005,
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
    print("  Enter: play next episode without recording a decision")
    print("  d: mark current episode for deletion")
    print("  q: quit")

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
            spawn=True,
            default_blueprint=blueprint,
        )
        rr.send_blueprint(blueprint)

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

            while True:
                choice = input(f"Episode {episode_index} review [Enter next / d delete / q quit]: ").strip().lower()
                if choice == "":
                    break
                if choice == "d":
                    append_review_decision(review_path, episode_info, "delete")
                    print(f"Recorded delete for episode {episode_index}")
                    break
                if choice == "q":
                    print("Quit episode review.")
                    return
                print("Unknown input. Use Enter, d, or q.")
        finally:
            close_rerun_ui(force=force_close_ui)

if __name__ == "__main__":
    import rerun as rr

    EE_POSE_MODE = "matrix"  # 可选："matrix"、"quat7"、"both"
    PLAYBACK_FPS = 45
    REALTIME_STREAM = True
    MEMORY_LIMIT_GB = 10
    REVIEW_PATH = "data/train_episode/wipe_board/wipe_board_lerobotv3/episode_review.jsonl"

    reader = LeRobotv3Reader(
        root="data/train_episode/wipe_board/wipe_board_lerobotv3",
        repo_id="local/h5_to_lerobot_v3",
        video_backend="torchcodec",
    )

    blueprint = build_rerun_blueprint()
    visualizer = RerunLeRobotVisualizer(reader, ee_pose_mode=EE_POSE_MODE)
    run_episode_review(
        reader=reader,
        visualizer=visualizer,
        review_path=REVIEW_PATH,
        blueprint=blueprint,
        playback_fps=PLAYBACK_FPS,
        realtime_stream=REALTIME_STREAM,
        memory_limit_gb=MEMORY_LIMIT_GB,
    )
