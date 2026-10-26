from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from app_logging import get_logger, install_exception_hook
from app_paths import APP_DIR, CAL_DIR, OUTPUT_DIR
from nsl_geometry import geometry_spec, range_to_xyz, rt_geometry, detected_lens, fusion_block_reason, ray_scale, scale_ray_angles


ROOT = APP_DIR
DEFAULT_CAPTURE_ROOT = Path(os.environ.get("CAM_ALIGN_CAPTURE_ROOT", str(ROOT / "captures")))
DEFAULT_COLOR_CAL = CAL_DIR / "c"
LOGGER = get_logger("fusion")


def load_camera(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    k = np.loadtxt(folder / "Intrinsic.txt", dtype=np.float64).reshape(3, 3)
    dist = np.loadtxt(folder / "Distortion.txt", dtype=np.float64).reshape(-1)
    return k, dist
def load_rt(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rt_geometry(data)
    if "R" in data and "t" in data:
        r = np.asarray(data["R"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data["t"], dtype=np.float64).reshape(3)
    elif "RT" in data:
        rt = np.asarray(data["RT"], dtype=np.float64).reshape(4, 4)
        r, t = rt[:3, :3], rt[:3, 3]
    else:
        raise ValueError("RT JSON에는 R+t 또는 4x4 RT가 필요합니다.")

    source = str(data.get("from", "")).lower()
    target = str(data.get("to", "")).lower()
    if source and not any(x in source for x in ("nsl", "lidar", "pointcloud")):
        raise ValueError(f"이 RT의 source는 NSL 점군이 아닙니다: {source}")
    if target and target != "c" and "color" not in target:
        raise ValueError(f"이 RT의 target은 Color 카메라가 아닙니다: {target}")

    unit = str(data.get("translation_unit", "m")).lower()
    if unit in ("mm", "millimeter", "millimeters"):
        t = t / 1000.0
    elif unit not in ("m", "meter", "meters"):
        raise ValueError(f"지원하지 않는 translation_unit: {unit}")
    if not np.isfinite(r).all() or not np.isfinite(t).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1., atol=1e-4):
        raise ValueError("유효한 회전행렬과 유한한 이동벡터가 필요합니다.")
    data = {**data, "R": r.tolist(), "t": t.tolist(), "translation_unit": "m",
            "RT": np.block([[r, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist()}
    return r, t, data
def read_nsl_pcd(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PCD DATA 헤더를 찾지 못했습니다.")
            line = raw.decode("ascii").strip()
            if not line or line.startswith("#"):
                continue
            key, *values = line.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break

        if header["DATA"][0].lower() != "binary":
            raise ValueError("현재 변환기는 샘플과 같은 binary PCD를 사용합니다.")
        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = [x.upper() for x in header["TYPE"]]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        if any(c != 1 for c in counts):
            raise ValueError("COUNT > 1 PCD 필드는 지원하지 않습니다.")
        codes = {
            ("F", 4): "<f4", ("F", 8): "<f8",
            ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
            ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
        }
        try:
            dtype = np.dtype([
                (name, codes[(kind, size)])
                for name, size, kind in zip(fields, sizes, types)
            ])
        except KeyError as exc:
            raise ValueError(f"지원하지 않는 PCD 필드 형식: {exc}") from exc

        count = int(header["POINTS"][0])
        payload = stream.read(count * dtype.itemsize)
        if len(payload) != count * dtype.itemsize:
            raise ValueError("PCD binary payload 크기가 헤더와 다릅니다.")
        data = np.frombuffer(payload, dtype=dtype, count=count)

    if not all(name in data.dtype.names for name in ("x", "y", "z")):
        raise ValueError("PCD에 x, y, z 필드가 필요합니다.")
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    intensity = np.asarray(data["intensity"], dtype=np.float32) if "intensity" in data.dtype.names else np.zeros(count, np.float32)
    width = int(header.get("WIDTH", [count])[0])
    height = int(header.get("HEIGHT", [1])[0])
    if width * height != count:
        height, width = 1, count
    return xyz, intensity, (height, width)

def read_source_cloud(pair, geometry='pcd'):
    geometry_spec(geometry)
    xyz, intensity, shape = read_nsl_pcd(pair['cloud'])
    if geometry == 'wf_range':
        path = pair.get('range')
        if path is None or not Path(path).is_file():
            raise ValueError('WF 복원용 원본 거리맵이 없습니다. NSL_distance_raw_u16 PNG가 필요합니다.')
        raw = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape != shape:
            raise ValueError('원본 거리맵과 PCD 해상도가 다릅니다.')
        recovered = range_to_xyz(raw)
        recovered[~np.isfinite(xyz).all(axis=1)] = np.nan
        xyz = recovered
    return xyz, intensity, shape

def capture_pairs(root: Path) -> list[dict]:
    pairs = []
    for manifest_path in sorted(root.glob("capture_manifest_*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [root / Path(x.replace("\\", "/")) for x in manifest.get("files", [])]
        color = next((p for p in files if p.parent.name == "RealSense_color" and p.suffix.lower() == ".png"), None)
        cloud = next((p for p in files if p.parent.name == "NSL_pointcloud" and p.suffix.lower() == ".pcd"), None)
        distance = next((p for p in files if p.name.startswith('NSL_distance_raw_u16_') and p.suffix.lower() == '.png'), None)
        if color and cloud and color.is_file() and cloud.is_file():
            pairs.append({
                "timestamp": manifest["capture_timestamp"],
                "color": color,
                "cloud": cloud,
                "range": distance,
                "manifest": manifest_path,
            })
    return pairs


def _visible_indices(u: np.ndarray, v: np.ndarray, z: np.ndarray, width: int) -> np.ndarray:
    order = np.argsort(z)
    keys = (v[order].astype(np.int64) * width + u[order].astype(np.int64))
    _, first = np.unique(keys, return_index=True)
    return order[first]


def surface_depth_buffer(points, uv, shape, image_shape, max_edge_m=.035):
    """Rasterize measured neighboring triangles for VISIBILITY, not new cloud points.

    Reject discontinuities/missing vertices. Interpolate inverse Z in image space;
    unlike same-pixel point tests this also covers gaps between foreground samples.
    """
    height, width = image_shape[:2]
    depth = np.full((height, width), np.inf, np.float32)
    if shape is None or min(shape) < 2 or np.prod(shape) != len(points):
        return depth
    ids = np.arange(len(points)).reshape(shape)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    triangles = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    xyz, pixels = points[triangles], uv[triangles]
    valid = np.isfinite(xyz).all(axis=(1, 2)) & np.isfinite(pixels).all(axis=(1, 2))
    valid &= (xyz[..., 2] > 0).all(axis=1)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        valid &= np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1) < max_edge_m
    # Bound work as well as rejecting very large, poorly sampled projected cells.
    extent = np.ptp(pixels, axis=1)
    valid &= (extent <= 40).all(axis=1)
    pixels, z = pixels[valid], xyz[valid, :, 2]
    low = np.maximum(np.ceil(pixels.min(axis=1)), 0).astype(int)
    high = np.minimum(np.floor(pixels.max(axis=1)), (width - 1, height - 1)).astype(int)
    sizes = high - low + 1
    a, b, c = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    # Batch equal-sized pixel tiles: bounded memory, no Python loop per triangle.
    for tile_size in np.unique(sizes, axis=0):
        w, h = tile_size
        if min(w, h) <= 0:
            continue
        ids = np.flatnonzero(np.all(sizes == tile_size, axis=1) & (abs(den) > 1e-8))
        yy, xx = np.mgrid[:h, :w]
        for start in range(0, len(ids), 512):
            subset = ids[start:start + 512]
            x = low[subset, 0, None, None] + xx
            y = low[subset, 1, None, None] + yy
            ax, ay = a[subset].T[:, :, None, None]
            bx, by = b[subset].T[:, :, None, None]
            cx, cy = c[subset].T[:, :, None, None]
            divisor = den[subset, None, None]
            wa = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / divisor
            wb = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / divisor
            wc = 1 - wa - wb
            inverse = wa / z[subset, 0, None, None] + wb / z[subset, 1, None, None] + wc / z[subset, 2, None, None]
            inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6) & (inverse > 0)
            np.minimum.at(depth.ravel(), (y * width + x)[inside], 1. / inverse[inside])
    return depth
def fuse_points(
    xyz_mm: np.ndarray,
    intensity: np.ndarray,
    image: np.ndarray,
    k: np.ndarray,
    dist: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    z_min: float,
    z_max: float,
    visible_only: bool = True,
    return_indices: bool = False,
    organized_shape=None,
    occlusion_tolerance_m=.015,
    source_ray_scale=1.,
):
    # The UI range is NSL Z, consistently for detection, preview and exports.
    source_z = xyz_mm[:, 2] * .001
    source_valid = np.isfinite(xyz_mm).all(axis=1) & (xyz_mm[:, 2] < 64000)
    source_valid &= (source_z >= z_min) & (source_z <= z_max) & (source_z > 0)
    xyz_m = scale_ray_angles(xyz_mm[source_valid] * 0.001, source_ray_scale)
    indices = np.flatnonzero(source_valid)
    intensity = intensity[source_valid]
    points = (r @ xyz_m.T).T + t

    front = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points, intensity = points[front], intensity[front]
    indices = indices[front]
    uv = (cv2.projectPoints(points, np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
          if len(points) else np.empty((0, 2)))
    height, width = image.shape[:2]
    pixels = np.rint(uv)
    inside = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    points, intensity = points[inside], intensity[inside]
    indices = indices[inside]
    u, v = pixels[inside].astype(np.int32).T
    in_fov = len(points)
    surface_rejected = 0
    if visible_only and in_fov and organized_shape is not None:
        full_points = np.full((len(xyz_mm), 3), np.nan)
        full_uv = np.full((len(xyz_mm), 2), np.nan)
        # Keep off-screen vertices too: a triangle may cross the image boundary.
        all_points = scale_ray_angles(xyz_mm * .001, source_ray_scale) @ r.T + t
        all_valid = source_valid & np.isfinite(all_points).all(axis=1) & (all_points[:, 2] > 0)
        full_points[all_valid] = all_points[all_valid]
        if all_valid.any():
            full_uv[all_valid] = cv2.projectPoints(all_points[all_valid], np.zeros(3), np.zeros(3), k, dist)[0].reshape(-1, 2)
        buffer = surface_depth_buffer(full_points, full_uv, organized_shape, image.shape)
        keep = points[:, 2] <= buffer[v, u] + occlusion_tolerance_m
        surface_rejected = int(np.count_nonzero(~keep))
        points, intensity, u, v, indices = points[keep], intensity[keep], u[keep], v[keep], indices[keep]
    if visible_only and len(points):
        keep = _visible_indices(u, v, points[:, 2], width)
        points, intensity, u, v = points[keep], intensity[keep], u[keep], v[keep]
        indices = indices[keep]

    rgb = image[v, u, ::-1].copy() if len(points) else np.empty((0, 3), np.uint8)
    stats = {
        "pcd_points": int(len(xyz_mm)),
        "valid_source_points": int(np.count_nonzero(source_valid)),
        "in_color_fov_points": int(in_fov),
        "output_points": int(len(points)),
        "visible_only": bool(visible_only),
        "visibility_method": "organized_surface_zbuffer" if organized_shape is not None else "same_pixel_points",
        "surface_occluded_points": surface_rejected,
        "occlusion_tolerance_m": float(occlusion_tolerance_m),
        "z_range_frame": "NSL",
        "output_coordinate_frame": "RealSense Color camera",
        "coordinate_unit": "meter",
        "source_ray_angle_scale": float(source_ray_scale),
    }
    result = points.astype(np.float32), rgb, intensity.astype(np.float32), u, v, stats
    return (*result, indices) if return_indices else result


def projection_preview(image: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    out = image.copy()
    if not len(z):
        return out
    lo, hi = np.percentile(z, [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    gray = np.uint8(np.clip((z - lo) * 255.0 / (hi - lo), 0, 255))
    colors = cv2.applyColorMap(gray.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    base = out[v, u].astype(np.float32)
    out[v, u] = np.rint(base * 0.35 + colors * 0.65).astype(np.uint8)
    return out
def write_binary_ply(path: Path, points: np.ndarray, rgb: np.ndarray, intensity=None,
                     coordinate_frame="RealSense Color camera") -> None:
    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if intensity is not None:
        fields.append(("intensity", "<f4"))
    dtype = np.dtype(fields)
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if intensity is not None:
        data["intensity"] = intensity
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment coordinate_unit meter\ncomment coordinate_frame {coordinate_frame}\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        + ("property float intensity\n" if intensity is not None else "")
        + "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(data.tobytes())
    LOGGER.info("PLY saved path=%s points=%d color=RGB intensity=%s frame=%s", path, len(points), intensity is not None, coordinate_frame)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    path.write_bytes(encoded.tobytes())
def process_pair(pair: dict, output: Path, k, dist, r, t, rt_meta, args) -> dict:
    if ('target_intrinsics' in rt_meta and not np.array_equal(k, rt_meta['target_intrinsics'])) or (
            'target_distortion' in rt_meta and not np.array_equal(dist, rt_meta['target_distortion'])):
        raise ValueError('선택한 Color 내부 파라미터가 RT 계산에 사용한 값과 다릅니다.')
    image = cv2.imdecode(np.fromfile(pair["color"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(pair["color"])
    xyz_mm, intensity, shape = read_source_cloud(pair, rt_geometry(rt_meta))
    mode = rt_geometry(rt_meta)
    nf_comparison = bool(getattr(args, 'allow_nf_comparison', False))
    if reason := fusion_block_reason({'source_geometry':geometry_spec(mode),
                                      'source_lens':detected_lens(xyz_mm,shape) if mode == 'pcd' else None,
                                      'allow_nf_comparison':nf_comparison}):
        raise ValueError(reason)
    points, rgb, intensity, u, v, stats = fuse_points(
        xyz_mm, intensity, image, k, dist, r, t,
        args.z_min, args.z_max, not args.keep_occluded, organized_shape=shape,
        source_ray_scale=ray_scale(rt_meta),
    )
    if not len(points):
        raise ValueError('Color 시야에 투영되는 점이 없습니다. 빈 PLY는 저장하지 않습니다.')

    output.mkdir(parents=True, exist_ok=True)
    stem = f"NSL_to_Color_{pair['timestamp']}"
    ply_path = output / f"{stem}.ply"
    preview_path = output / f"{stem}_projection.png"
    meta_path = output / f"{stem}.json"
    write_binary_ply(ply_path, points, rgb, intensity if args.include_intensity else None)
    write_image(preview_path, projection_preview(image, u, v, points[:, 2]))

    meta = {
        **stats,
        "timestamp": pair["timestamp"],
        "coordinate_frame": "RealSense Color camera",
        "coordinate_unit": "meter",
        "source_pcd": str(pair["cloud"]),
        "source_color": str(pair["color"]),
        "rt": rt_meta,
        "z_range_m": [args.z_min, args.z_max],
        "includes_intensity": bool(args.include_intensity),
        "nf_comparison_only": nf_comparison and mode == 'pcd',
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "FUSION %s valid=%d fov=%d output=%d ply=%s",
        pair["timestamp"], stats["valid_source_points"],
        stats["in_color_fov_points"], stats["output_points"], ply_path,
    )
    return {"ply": ply_path, "preview": preview_path, "meta": meta_path, **stats}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSL PCD를 RealSense Color 영상과 융합해 RGB PLY로 저장")
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--color-cal", type=Path, default=DEFAULT_COLOR_CAL)
    parser.add_argument("--rt", type=Path, required=True, help="NSL -> Color RT JSON")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "fusion")
    parser.add_argument("--timestamp", default="", help="특정 YYYYMMDD_HHMMSS_mmm만 처리")
    parser.add_argument("--all", action="store_true", help="매칭되는 모든 촬영쌍 처리")
    parser.add_argument("--z-min", type=float, default=0.10)
    parser.add_argument("--z-max", type=float, default=20.0)
    parser.add_argument("--keep-occluded", action="store_true", help="같은 Color 픽셀 뒤쪽 점도 유지")
    parser.add_argument('--allow-nf-comparison', action='store_true', help='알려진 NF/광각 불일치 원본을 비교용으로만 허용')
    parser.add_argument("--include-intensity", action="store_true", help="NSL intensity 스칼라 필드도 저장 (CloudCompare에서 Colors=RGB 선택 필요)")
    return parser


def main() -> None:
    install_exception_hook(LOGGER)
    args = build_parser().parse_args()
    if args.z_min <= 0 or args.z_max <= args.z_min:
        raise ValueError("z 범위가 올바르지 않습니다.")
    pairs = capture_pairs(args.capture_root)
    if args.timestamp:
        pairs = [p for p in pairs if p["timestamp"] == args.timestamp]
    elif not args.all:
        pairs = pairs[:1]
    if not pairs:
        raise RuntimeError("Color PNG와 NSL PCD가 함께 있는 촬영쌍을 찾지 못했습니다.")

    k, dist = load_camera(args.color_cal)
    r, t, rt_meta = load_rt(args.rt)
    print(f"pairs={len(pairs)}  color_cal={args.color_cal}")
    for pair in pairs:
        result = process_pair(pair, args.output, k, dist, r, t, rt_meta, args)
        print(
            f"{pair['timestamp']}: {result['output_points']:,} points -> "
            f"{result['ply'].name}"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
