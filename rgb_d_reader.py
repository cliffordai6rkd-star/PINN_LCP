import argparse
import json
import time
from pathlib import Path

cv2 = None
np = None
rs = None


def load_runtime_deps():
    global cv2, np

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency. Install/import OpenCV and NumPy "
            "in the environment where you run this script."
        ) from exc

    load_realsense()
    cv2 = cv2_module
    np = np_module


def load_realsense():
    global rs

    try:
        import pyrealsense2 as rs_module
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency: pyrealsense2. Run this script in an environment "
            "with Intel RealSense SDK Python bindings installed."
        ) from exc

    rs = rs_module


def parse_args():
    parser = argparse.ArgumentParser(description="Capture aligned RGB-D frames from an Intel RealSense camera.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/realsense_rgbd"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30, help="Frames to drop before saving.")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of RGB-D frames to save.")
    parser.add_argument("--serial", default=None, help="Optional RealSense device serial number.")
    parser.add_argument("--preview", action="store_true", help="Show live preview while capturing.")
    parser.add_argument("--list-devices", action="store_true", help="List connected RealSense devices and exit.")
    return parser.parse_args()


def list_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    for idx, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        firmware = device.get_info(rs.camera_info.firmware_version)
        print(f"[{idx}] name={name}, serial={serial}, firmware={firmware}")


def build_pipeline(args):
    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial:
        config.enable_device(args.serial)

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    return pipeline, align, profile, depth_scale


def frame_intrinsics(video_profile):
    intr = video_profile.get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "model": str(intr.model),
        "coeffs": list(intr.coeffs),
    }


def wait_for_aligned_frame(pipeline, align, timeout_ms=5000):
    frames = pipeline.wait_for_frames(timeout_ms)
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError("Failed to get aligned color/depth frames.")

    return color_frame, depth_frame


def save_frame(output_dir, frame_idx, color_frame, depth_frame, depth_scale):
    color_bgr = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_meters = depth_raw.astype(np.float32) * depth_scale

    prefix = f"frame_{frame_idx:06d}"
    color_path = output_dir / f"{prefix}_rgb.png"
    depth_raw_path = output_dir / f"{prefix}_depth_raw.png"
    depth_meter_path = output_dir / f"{prefix}_depth_meters.npy"
    depth_vis_path = output_dir / f"{prefix}_depth_vis.png"
    metadata_path = output_dir / f"{prefix}_metadata.json"

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imwrite(str(color_path), color_bgr)
    cv2.imwrite(str(depth_raw_path), depth_raw)
    np.save(depth_meter_path, depth_meters)
    cv2.imwrite(str(depth_vis_path), depth_vis)

    metadata = {
        "timestamp_unix": time.time(),
        "color_timestamp_ms": color_frame.get_timestamp(),
        "depth_timestamp_ms": depth_frame.get_timestamp(),
        "depth_scale_m_per_unit": depth_scale,
        "color_path": str(color_path),
        "depth_raw_path": str(depth_raw_path),
        "depth_meters_path": str(depth_meter_path),
        "depth_vis_path": str(depth_vis_path),
        "color_intrinsics": frame_intrinsics(color_frame.profile.as_video_stream_profile()),
        "depth_intrinsics_aligned_to_color": frame_intrinsics(depth_frame.profile.as_video_stream_profile()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "color": color_bgr,
        "depth_raw": depth_raw,
        "depth_vis": depth_vis,
        "metadata_path": metadata_path,
    }


def main():
    args = parse_args()

    if args.list_devices:
        load_realsense()
        list_devices()
        return

    load_runtime_deps()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, align, profile, depth_scale = build_pipeline(args)
    try:
        for _ in range(args.warmup):
            wait_for_aligned_frame(pipeline, align)

        for frame_idx in range(args.num_frames):
            color_frame, depth_frame = wait_for_aligned_frame(pipeline, align)
            saved = save_frame(args.output_dir, frame_idx, color_frame, depth_frame, depth_scale)

            print(
                f"Saved frame {frame_idx}: "
                f"rgb/depth/depth_meters/depth_vis, metadata={saved['metadata_path']}"
            )

            if args.preview:
                preview = np.hstack((saved["color"], saved["depth_vis"]))
                cv2.imshow("RealSense RGB-D", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
