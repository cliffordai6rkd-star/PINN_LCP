# 双视角 RGBD 点云重建操作说明

本文档说明如何使用 D455 固定相机和 D435 腕部相机的 RGBD 数据进行点云重建，并用 Open3D 打开可移动视角的三维点云。

## 0. 进入容器

在项目根目录启动容器：

```bash
docker compose -f "docker for LCP_PINN_V1/docker_compose.yaml" up -d v2
```

进入容器：

```bash
docker exec -it st-pinn bash
cd /workspace
```

数据路径在容器内为：

```text
data/train_episode/Ft_test_data
```

## 1. 一键执行完整 RGBD 重建流

最常用入口是薄上层脚本：

```bash
python pointcloud/run_rgbd_reconstruction.py \
  --config pointcloud/config/rgbd_reconstruction/ft_test_data.yaml
```

它会按顺序执行：

```text
1. 需要时导出 pointcloud/calibration_v2/camera_extrinsics.json
2. 写入 data_with_rgbd_extrinsics.json
3. 按 config 重建 fused .ply 和 summary.json
```

常用调试命令：

```bash
python pointcloud/run_rgbd_reconstruction.py \
  --max-frames 10 \
  --stride 8 \
  --cameras third_person_cam \
  --no-view
```

如果想强制重新导出标定：

```bash
python pointcloud/run_rgbd_reconstruction.py \
  --calibration-mode always \
  --write-npy
```

如果只想复用已有标定并跳过重建前的 JSON 写入：

```bash
python pointcloud/run_rgbd_reconstruction.py \
  --calibration-mode skip \
  --skip-extrinsics
```

检查实际会执行哪些命令：

```bash
python pointcloud/run_rgbd_reconstruction.py --dry-run
```

上层脚本只负责编排；实际几何计算仍然在：

```text
pointcloud/calibration_v2/export_camera_extrinsics.py
pointcloud/tools/add_rgbd_extrinsics.py
pointcloud/tools/reconstruct_rgbd_episode.py
```

## 2. 分步执行：导出相机标定外参

如果已经生成过最新的 `pointcloud/calibration_v2/camera_extrinsics.json`，这一步可以跳过。

```bash
python pointcloud/calibration_v2/export_camera_extrinsics.py --write-npy
```

输出文件：

```text
pointcloud/calibration_v2/camera_extrinsics.json
pointcloud/calibration_v2/ee_cam_d435/T_ee_camera.npy
pointcloud/calibration_v2/third_person_cam_d455/T_base_camera.npy
```

外参含义：

```text
ee_cam:
  T_ee_camera      # 腕部相机 D435，相机坐标系到机械臂末端坐标系

third_person_cam:
  T_base_camera    # 固定相机 D455，相机坐标系到 robot_base 坐标系
```

## 3. 分步执行：写入每帧 RGBD 外参

该步骤读取 `data.json`，根据每帧 `ee_states.single.pose` 计算每帧相机位姿，并生成 `data_with_rgbd_extrinsics.json`。

```bash
python pointcloud/tools/add_rgbd_extrinsics.py \
  --dataset-dir data/train_episode/Ft_test_data \
  --calibration pointcloud/calibration_v2/camera_extrinsics.json \
  --input-json data.json \
  --output-json data_with_rgbd_extrinsics.json \
  --robot-key single \
  --ee-pose-order xyzw \
  --depth-scale-m-per-unit 0.001
```

输出文件：

```text
data/train_episode/Ft_test_data/data_with_rgbd_extrinsics.json
```

计算逻辑：

```text
固定相机 D455:
  T_base_camera = 标定得到的 T_base_camera

腕部相机 D435:
  T_base_camera = T_base_ee @ T_ee_camera
```

注意：`Ft_test_data` 中的 `ee_states.single.pose` 是绝对位姿，可以直接用于腕部相机外参计算。

## 4. 修改重建配置

点云重建脚本只通过 config 控制参数。默认配置文件是：

```text
pointcloud/config/rgbd_reconstruction/ft_test_data.yaml
```

默认配置：

```yaml
dataset:
  dir: data/train_episode/Ft_test_data
  input_json: data_with_rgbd_extrinsics.json

output:
  dir: outputs/rgbd_pointcloud/Ft_test_data
  name: episode_fused_stride8.ply
  save_per_frame: false
  ascii: false

frames:
  start: 0
  end: null
  step: 1
  max: null

projection:
  cameras: null
  stride: 8
  depth_min: 0.05
  depth_max: 2.0
  depth_scale_m_per_unit: null

downsample:
  voxel_size: 0.003
  max_points: 2000000

visualization:
  view: false
  view_existing: null
  coord_frame_size: 0.08
  window_width: 1280
  window_height: 800
```

常用参数：

```text
projection.stride:
  图像像素采样间隔。数值越小，点云越密，文件越大。

projection.depth_min / projection.depth_max:
  有效深度范围，单位是米。

projection.cameras:
  null 表示使用所有相机。
  单相机可以写 "ee_cam"，也可以写 ["ee_cam"]。
  只使用腕部相机时写 ["ee_cam"]。
  只使用固定相机时写 ["third_person_cam"]。

frames.max:
  只重建前 N 帧。调试时建议设为 10 或 20。

downsample.voxel_size:
  voxel 下采样尺寸，单位是米。0 表示不做 voxel 下采样。

visualization.view:
  true 表示重建完成后直接打开 Open3D 可视化窗口。

visualization.view_existing:
  指向已有 .ply 时，只打开已有点云并退出，不重新重建。
```

## 5. 分步执行：点云重建

使用默认配置：

```bash
python pointcloud/tools/reconstruct_rgbd_episode.py
```

指定其他配置文件：

```bash
python pointcloud/tools/reconstruct_rgbd_episode.py \
  --config pointcloud/config/rgbd_reconstruction/ft_test_data.yaml
```

输出文件：

```text
outputs/rgbd_pointcloud/Ft_test_data/episode_fused_stride8.ply
outputs/rgbd_pointcloud/Ft_test_data/summary.json
```

## 6. 打开可移动视角点云

方式 A：重建完成后自动打开窗口。

修改 `pointcloud/config/rgbd_reconstruction/ft_test_data.yaml`：

```yaml
visualization:
  view: true
  view_existing: null
  coord_frame_size: 0.08
  window_width: 1280
  window_height: 800
```

然后运行：

```bash
python pointcloud/tools/reconstruct_rgbd_episode.py
```

方式 B：只打开已经生成的点云，不重新重建。

修改 `pointcloud/config/rgbd_reconstruction/ft_test_data.yaml`：

```yaml
visualization:
  view: false
  view_existing: outputs/rgbd_pointcloud/Ft_test_data/episode_fused_stride8.ply
  coord_frame_size: 0.08
  window_width: 1280
  window_height: 800
```

然后运行：

```bash
python pointcloud/tools/reconstruct_rgbd_episode.py
```

Open3D 窗口操作：

```text
左键拖动：旋转视角
滚轮：缩放
右键或中键拖动：平移
```

## 7. 快速调试建议

第一次调参数时，建议先只重建少量帧：

```yaml
frames:
  start: 0
  end: null
  step: 1
  max: 10

projection:
  stride: 8
  depth_min: 0.05
  depth_max: 2.0
```

确认点云方向和位置正常后，再把 `frames.max` 改回 `null` 重建全量 episode。

## 8. 常见问题

如果提示找不到 `data_with_rgbd_extrinsics.json`：

```text
先执行第 3 步 add_rgbd_extrinsics.py。
```

如果点云很大或可视化卡顿：

```text
增大 projection.stride，例如 8 -> 12。
增大 downsample.voxel_size，例如 0.003 -> 0.005。
降低 downsample.max_points。
```

如果点云包含大量背景：

```text
当前脚本默认使用整张 depth 图重建，所以会包含桌面、墙面、机械臂等背景。
后续如果只需要目标物体，需要增加 mask 或 robot_base 空间裁剪。
```

如果 Open3D 窗口打不开：

```text
确认已经在支持 GUI 的容器里运行。
确认 docker compose 中 DISPLAY 和 XAUTHORITY 已正确映射。
```

如果双相机点云明显错位：

```text
先确认 pointcloud/calibration_v2/camera_extrinsics.json 是最新标定结果。
再确认 data_with_rgbd_extrinsics.json 使用的是 ee_states.single.pose，而不是 actions.single.ee.pose。
最后检查 depth_scale_m_per_unit 是否为 0.001。
```
