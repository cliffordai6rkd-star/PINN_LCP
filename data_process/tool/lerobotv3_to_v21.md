# LeRobot v3 → v2.1（供本地 OpenPI 使用）

## 当前 peel_cucumber 数据的转换命令

在 `PINN_LCP` 仓库根目录运行，确保 Python 环境有 `numpy pandas pyarrow`，
且 PATH 中能找到支持 `libx264` 的 `ffmpeg` 和 `ffprobe`。不依赖 LeRobot 安装版本。

```bash
python data_process/tool/lerobotv3_to_v21.py \
  --input-dir ../xarm_ws/runs/visu_data/peel_cucumber_lerobotv3 \
  --output-dir ../xarm_ws/runs/visu_data/peel_cucumber_lerobotv21 \
  --state-key observation.joint \
  --action-key action.joint \
  --dry-run
```

去掉 `--dry-run` 执行转换。预检查只检查元数据、统计、引用文件和输出路径，
不执行视频解码或完整逐行检查。

本次输入为 `v3.0`，包含 48 个 episode、26241 帧、25 Hz，两路摄像头为
`observation.images.left_wrist` 和 `observation.images.right_wrist`。
脚本以实际 parquet 和元数据为准，不依赖 H5 转换用的 YAML。
新增以下列，数值和统计均复制对应源字段：

- `observation.state` ← `observation.joint`，每帧 `[7]`。
- `action` ← `action.joint`，每帧 `[7]`。

脚本按原值复制 `action.joint`，不推断它是测量值还是命令值。
当前数据没有末端位姿列，不计算 FK、delta action，不改变坐标系或重采样。
所有原始数值列和摄像头名称保留，源目录不修改。

`observation.tau_ext` 的源统计含 NaN。辅助字段及其统计按原值保留，
统计文件采用 LeRobot 自身 JSON 读取器支持的 `NaN` 表示，不填零。
选作 state/action 的字段必须具有有限的统计和数值，非有限值或形状错误会报错。

## 输出布局与其他输入

输入本身是一个数据集时，`--output-dir` 就是精确输出根目录：

```text
peel_cucumber_lerobotv21/
├── meta/
│   ├── info.json                 # codebase_version: v2.1
│   ├── stats.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_000000.parquet
│   ...
└── videos/chunk-000/
    ├── observation.images.left_wrist/episode_000000.mp4
    └── observation.images.right_wrist/episode_000000.mp4
        ...
```

输入是多个 v3 数据集的父目录时，递归发现数据集，在输出父目录保留相对目录。
各数据集独立，不合并。输入输出路径相对当前工作目录，也可使用绝对路径。
两者须显式传入。与旧脚本不同，不再默认指向 Nero 的特定目录，
单数据集输出也不再额外追加输入目录名。

省略 `--state-key` / `--action-key` 时，按以下顺序选择并打印实际映射：

| 目标列 | 自动选择顺序 |
| --- | --- |
| `observation.state` | 已有 `observation.state`、`observation.joint`、`observation.ee_pose`、`observation.eepose` |
| `action` | 已有 `action`、`action.joint`、`action.ee_pose`、`action.eepose` |

已有规范列直接保留。明确指定不同源列而目标列已存在时，拒绝覆盖。
需要位姿动作时可传 `--action-key action.eepose`，但源数据必须实际包含该列。

视频按 v3 每个 episode 的起始时间精确解码，编码为 H.264 CRF 0，
保留源视频解码后的 YUV 像素；不会恢复 AV1 已损失的信息，输出体积通常增大。
每个 episode 的视频从零时间开始，逐个检查帧数、FPS 和起始时间。
Parquet 内 Hugging Face primitive `List` 元数据转换为 `Sequence`，
兼容本地 OpenPI 环境的 `datasets 3.6.0`。

输出目录已存在时拒绝覆盖。每个数据集先写临时目录，成功后重命名到目标；
失败清理本次临时目录，已经完成的其他数据集保留。
支持 v3.0 中每个 episode 的数据属于一个 parquet 文件、每路视频属于一个 MP4
文件的布局，以及没有媒体的数值数据集；episode 编号须连续且从零开始。
不支持跨文件单 episode、图像文件模式、深度视频、非 yuv420p 视频和嵌套 List。

## OpenPI 读取配置

本地 `../openpi/src/openpi/training/data_loader.py` 按 `repo_id` 从
`HF_LEROBOT_HOME` 查找数据。启动 OpenPI 前设置：

```bash
export HF_LEROBOT_HOME=/home/eid/code/xarm_ws/runs/visu_data
```

训练配置使用 `repo_id="peel_cucumber_lerobotv21"`。最终 `DataConfig` 设置：

```python
action_sequence_keys=("action",)
prompt_from_task=True
```

`action_sequence_keys` 使用 LeRobot 原始列名，加载器据此构造未来动作序列；
`RepackTransform` 再把单数 `action` 映射到模型使用的复数 `actions`：

```python
_transforms.RepackTransform({
    "images": {
        "left_wrist": "observation.images.left_wrist",
        "right_wrist": "observation.images.right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
})
```

如果使用 `SimpleDataConfig`，读取选项放在 `base_config=DataConfig(...)` 中。
完整训练仍需 xArm 机器人 transform，把两路图像适配为模型要求的
`image` / `image_mask`，并在推理输出端取回 7 维动作。
选定训练配置后运行 OpenPI 的 `scripts/compute_norm_stats.py --config-name ...`；
数据集 `meta/stats.json` 不替代 OpenPI 的训练归一化统计。

## 验证

```bash
python -m pytest -q tests/test_lerobotv3_to_v21.py
```

测试包含原始数值列保留、关节/位姿/已有规范列、显式字段选择、辅助 NaN 保留、
视频切分边界和像素一致性、错误清理、单数据集与批量 CLI。
安装本地 OpenPI 锁定的旧版 LeRobot 时，还测试实际读取及 episode 末尾的动作填充；
没有该依赖时仅跳过这项读取测试。

在 OpenPI 自己的 Python 环境中读取实际输出：

```python
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

root = Path("/home/eid/code/xarm_ws/runs/visu_data/peel_cucumber_lerobotv21")
dataset = LeRobotDataset(
    repo_id=root.name,
    root=root,
    delta_timestamps={"action": [i / 25 for i in range(50)]},
    video_backend="pyav",
)
sample = dataset[0]
assert len(dataset) == 26241
assert sample["action"].shape == (50, 7)
assert sample["observation.state"].shape == (7,)
assert sample["observation.images.left_wrist"].shape == (3, 224, 224)
assert sample["observation.images.right_wrist"].shape == (3, 224, 224)
print(sample["task"])
```

格式参考：[LeRobot v3 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、
[本地 OpenPI 锁定的 v2.1 工具代码](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/utils.py)。
