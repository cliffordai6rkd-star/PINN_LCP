import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-frame SAM3/RGB-D object point-cloud reconstruction."
    )
    parser.add_argument("--rgb", type=Path, default=Path("outputs/realsense_rgbd/frame_000000_rgb.png"))
    parser.add_argument("--depth-meters", type=Path, default=Path("outputs/realsense_rgbd/frame_000000_depth_meters.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/realsense_rgbd/frame_000000_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sam3_pointcloud_single_frame"))

    parser.add_argument("--mask", type=Path, default=None, help="Optional mask .png/.jpg/.npy. Non-zero pixels are kept.")
    parser.add_argument("--sam-model", default="sam3.pt", help="Ultralytics SAM model path/name, e.g. sam3.pt.")
    parser.add_argument("--prompt", default=None, help="Text prompt for SAM3 concept segmentation.")
    parser.add_argument("--box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--point", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--mask-index", default="largest", help="'largest' or a zero-based mask index.")

    parser.add_argument("--depth-min", type=float, default=0.05, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max", type=float, default=3.0, help="Maximum valid depth in meters.")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth valid pixel.")
    parser.add_argument("--max-points", type=int, default=200000, help="Randomly downsample point cloud if larger.")

    parser.add_argument("--query-point-camera", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--sdf-grid", action="store_true", help="Save an unsigned distance grid around the point cloud.")
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--grid-padding", type=float, default=0.03)

    parser.add_argument("--view", action="store_true", help="Open an Open3D visualization window if open3d is installed.")
    return parser.parse_args()


def load_cv_numpy():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install/import opencv-python and numpy.") from exc
    return cv2, np


def load_inputs(args, cv2, np):
    color_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"failed to read RGB image: {args.rgb}")

    depth_m = np.load(args.depth_meters).astype(np.float32)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    if depth_m.shape[:2] != color_bgr.shape[:2]:
        raise ValueError(
            f"RGB/depth shape mismatch: rgb={color_bgr.shape[:2]}, depth={depth_m.shape[:2]}"
        )

    return color_bgr, depth_m, metadata


def load_mask(mask_path, image_shape, cv2, np):
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"failed to read mask image: {mask_path}")

    if mask.shape[:2] != image_shape:
        raise ValueError(f"mask shape mismatch: mask={mask.shape[:2]}, image={image_shape}")
    return mask.astype(bool)


def extract_masks_from_results(results, np):
    if not isinstance(results, (list, tuple)):
        results = [results]
    if not results:
        return []

    result = results[0]
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None:
        return []

    data = getattr(masks_obj, "data", None)
    if data is None:
        return []

    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    else:
        data = np.asarray(data)

    if data.ndim == 2:
        data = data[None]
    return [(m > 0.5) for m in data]


def run_sam3(args, image_shape, cv2, np):
    if args.prompt is None and args.box is None and args.point is None:
        raise ValueError("Provide --mask, or provide one of --prompt/--box/--point for SAM3.")

    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it in the container, or pass --mask."
        ) from exc

    model = SAM(args.sam_model)

    if args.box is not None:
        results = model.predict(source=str(args.rgb), bboxes=list(args.box))
    elif args.point is not None:
        results = model.predict(source=str(args.rgb), points=[list(args.point)], labels=[1])
    else:
        results = run_sam3_text_prompt(args, np)

    masks = extract_masks_from_results(results, np)
    if not masks:
        raise RuntimeError("SAM3 produced no masks. Try --box/--point, or pass a manual --mask.")

    if args.mask_index == "largest":
        areas = [int(mask.sum()) for mask in masks]
        mask = masks[int(np.argmax(areas))]
    else:
        idx = int(args.mask_index)
        mask = masks[idx]

    if mask.shape != image_shape:
        mask = cv2.resize(mask.astype(np.uint8), (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)

    return mask


def run_sam3_text_prompt(args, np):
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError:
        # Fallback for environments exposing text prompts through the basic SAM wrapper.
        from ultralytics import SAM

        model = SAM(args.sam_model)
        return model.predict(source=str(args.rgb), texts=[args.prompt])

    overrides = {"model": args.sam_model}
    predictor = SAM3SemanticPredictor(overrides=overrides)
    return predictor(source=str(args.rgb), text=[args.prompt])


def intrinsics_from_metadata(metadata):
    intr = metadata.get("depth_intrinsics_aligned_to_color") or metadata.get("color_intrinsics")
    if intr is None:
        raise KeyError("metadata missing depth_intrinsics_aligned_to_color/color_intrinsics")
    return intr


def mask_to_point_cloud(mask, color_bgr, depth_m, intr, args, np, cv2):
    valid = mask & np.isfinite(depth_m) & (depth_m >= args.depth_min) & (depth_m <= args.depth_max)
    if args.stride > 1:
        stride_mask = np.zeros_like(valid)
        stride_mask[:: args.stride, :: args.stride] = True
        valid &= stride_mask

    v, u = np.nonzero(valid)
    if v.size == 0:
        raise RuntimeError("No valid depth pixels remained after mask/depth filtering.")

    z = depth_m[v, u]
    x = (u.astype(np.float32) - float(intr["ppx"])) / float(intr["fx"]) * z
    y = (v.astype(np.float32) - float(intr["ppy"])) / float(intr["fy"]) * z
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    colors = color_rgb[v, u].astype(np.uint8)

    if points.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=args.max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    return points, colors


def write_ply(path, points, colors):
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_overlay(path, color_bgr, mask, cv2, np):
    overlay = color_bgr.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    overlay = np.where(mask[..., None], (0.55 * overlay + 0.45 * green).astype(np.uint8), overlay)
    cv2.imwrite(str(path), overlay)


def nearest_distance(points, query, np, chunk_size=200000):
    query = np.asarray(query, dtype=np.float32).reshape(1, 3)
    best = np.inf
    for start in range(0, points.shape[0], chunk_size):
        diff = points[start : start + chunk_size] - query
        dist2 = np.sum(diff * diff, axis=-1)
        best = min(best, float(np.min(dist2)))
    return float(np.sqrt(best))


def save_unsigned_sdf_grid(path, points, args, np):
    lo = points.min(axis=0) - args.grid_padding
    hi = points.max(axis=0) + args.grid_padding
    axes = [np.linspace(lo[i], hi[i], args.grid_size, dtype=np.float32) for i in range(3)]
    xx, yy, zz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    queries = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    distances = np.empty((queries.shape[0],), dtype=np.float32)
    for start in range(0, queries.shape[0], 4096):
        q = queries[start : start + 4096]
        best = np.full((q.shape[0],), np.inf, dtype=np.float32)
        for p_start in range(0, points.shape[0], 50000):
            p = points[p_start : p_start + 50000]
            diff = q[:, None, :] - p[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            best = np.minimum(best, np.min(dist2, axis=1))
        distances[start : start + q.shape[0]] = np.sqrt(best)

    np.savez_compressed(
        path,
        sdf=distances.reshape(args.grid_size, args.grid_size, args.grid_size),
        axes=np.asarray(axes, dtype=object),
        bounds=np.stack([lo, hi], axis=0),
        note="Unsigned distance to visible RGB-D surface points, not a watertight signed SDF.",
    )


def view_point_cloud(points, colors):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("Missing dependency: open3d. Install open3d or omit --view.") from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors.astype("float32") / 255.0)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([pcd, frame])


def main():
    args = parse_args()
    cv2, np = load_cv_numpy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    color_bgr, depth_m, metadata = load_inputs(args, cv2, np)
    image_shape = color_bgr.shape[:2]

    if args.mask is not None:
        mask = load_mask(args.mask, image_shape, cv2, np)
    else:
        mask = run_sam3(args, image_shape, cv2, np)

    intr = intrinsics_from_metadata(metadata)
    points, colors = mask_to_point_cloud(mask, color_bgr, depth_m, intr, args, np, cv2)

    mask_path = args.output_dir / "mask.png"
    overlay_path = args.output_dir / "mask_overlay.png"
    ply_path = args.output_dir / "object_pointcloud.ply"
    summary_path = args.output_dir / "summary.json"

    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    save_overlay(overlay_path, color_bgr, mask, cv2, np)
    write_ply(ply_path, points, colors)

    summary = {
        "rgb": str(args.rgb),
        "depth_meters": str(args.depth_meters),
        "metadata": str(args.metadata),
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "pointcloud_ply": str(ply_path),
        "num_points": int(points.shape[0]),
        "camera_frame": "x right, y down, z forward, units meters",
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
    }

    if args.query_point_camera is not None:
        phi_k = nearest_distance(points, args.query_point_camera, np)
        summary["query_point_camera"] = list(args.query_point_camera)
        summary["phi_k_m"] = phi_k

    if args.sdf_grid:
        sdf_path = args.output_dir / "unsigned_sdf_grid.npz"
        save_unsigned_sdf_grid(sdf_path, points, args, np)
        summary["unsigned_sdf_grid"] = str(sdf_path)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved point cloud: {ply_path}")
    print(f"Saved summary: {summary_path}")
    if "phi_k_m" in summary:
        print(f"phi_k_m: {summary['phi_k_m']:.6f}")

    if args.view:
        view_point_cloud(points, colors)


if __name__ == "__main__":
    main()
