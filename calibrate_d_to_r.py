import json
import re
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import least_squares

from app_logging import get_logger


ROOT = Path(__file__).parent / "cal"
RESULT_PATH = Path(__file__).parent / "d_to_r_rt.json"
DEFAULT_DEPTH_DIR = ROOT / "d"
DEFAULT_TARGET_DIR = ROOT / "r"
PATTERN = (7, 8)                 # inner corners
SQUARE_M = .03
BOARD_M = (.25, .28)
LOGGER = get_logger("calibration")


def camera(folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = ROOT / folder
    return np.loadtxt(folder / "Intrinsic.txt"), np.loadtxt(folder / "Distortion.txt")


def _stamp(path):
    match = re.search(r"\d{8}_\d{6}_\d{3}", path.stem)
    return match.group() if match else path.stem


def frame_pairs(depth_dir=DEFAULT_DEPTH_DIR, target_dir=DEFAULT_TARGET_DIR):
    extensions = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
    images = [p for p in Path(depth_dir).iterdir() if p.suffix.lower() in extensions]
    depth = [p for p in images if "depth_raw" in p.stem.lower()]
    if not depth:
        depth = [p for p in images if (image := cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) is not None
                 and image.ndim == 2 and image.dtype == np.uint16]
    target = [p for p in Path(target_dir).iterdir() if p.suffix.lower() in extensions]
    depth, target = {_stamp(p): p for p in depth}, {_stamp(p): p for p in target}
    return {stamp: (depth[stamp], target[stamp]) for stamp in sorted(depth.keys() & target.keys())}


def timestamps(depth_dir=DEFAULT_DEPTH_DIR, target_dir=DEFAULT_TARGET_DIR):
    return list(frame_pairs(depth_dir, target_dir))


def project_axis(center, normal, K, dist):
    points = np.float32([center, center + .06 * normal])
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)[0].reshape(2, 2)


def right_plane(image, K, dist):
    ok, grid = cv2.findChessboardCorners(
        image, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    grid = cv2.cornerSubPix(image, grid, (5, 5), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .01))
    obj = np.float32([(x * SQUARE_M, y * SQUARE_M, 0)
                      for y in range(PATTERN[1]) for x in range(PATTERN[0])])
    ok, rvec, tvec = cv2.solvePnP(obj, grid, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    R = cv2.Rodrigues(rvec)[0]
    center = R @ np.array([.09, .105, 0]) + tvec.ravel()
    normal = R[:, 2]
    if normal[2] > 0:
        normal = -normal

    mx, my = (BOARD_M[0] - 8 * SQUARE_M) / 2, (BOARD_M[1] - 9 * SQUARE_M) / 2
    board = np.float32([[-SQUARE_M - mx, -SQUARE_M - my, 0],
                        [7 * SQUARE_M + mx, -SQUARE_M - my, 0],
                        [7 * SQUARE_M + mx, 8 * SQUARE_M + my, 0],
                        [-SQUARE_M - mx, 8 * SQUARE_M + my, 0]])
    region = cv2.projectPoints(board, rvec, tvec, K, dist)[0].reshape(-1, 2)
    projected = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - grid.reshape(-1, 2)) ** 2, axis=1))))
    return {"center": center, "normal": normal, "h": float(normal @ center),
            "grid": grid.reshape(-1, 2), "region": region,
            "corners3d": board @ R.T + tvec.ravel(),
            "axis": project_axis(center, normal, K, dist), "rmse": rmse}


def depth_cloud(depth, K, dist, scale, step=1, z_min=.15, z_max=2.0, roi=None):
    y, x = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[y, x] * scale
    valid = (z > z_min) & (z < z_max)
    if roi is not None:
        valid &= roi[y, x] > 0
    x, y, z = x[valid], y[valid], z[valid]
    if not len(z):
        return np.empty((0, 3)), x, y
    rays = cv2.undistortPoints(np.float32([x, y]).T[:, None], K, dist).reshape(-1, 2)
    return np.column_stack((rays * z[:, None], z)), x, y


def target_to_depth_roi(target_plane, calibration, K, dist, shape, margin=.20):
    """Project the PnP board outline from Target into the Depth image."""
    if target_plane is None or calibration is None:
        return None
    R, t = np.asarray(calibration["R"]), np.asarray(calibration["t"])
    corners = (np.asarray(target_plane["corners3d"]) - t) @ R
    if np.any(corners[:, 2] <= 0):
        return None
    polygon = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    polygon = polygon.mean(0) + (polygon - polygon.mean(0)) * (1 + margin)
    # polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1] - 1)
    # polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0] - 1)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
    if mask.sum() < 100:
        return None
    return {"polygon": polygon, "mask": mask, "margin": float(margin),
            "coverage": float(mask.mean())}


def write_ply(path, points, colors, coordinate_system):
    started = perf_counter()
    LOGGER.info("PLY write start path=%s points=%d", path, len(points))
    header = ("ply\nformat ascii 1.0\n" f"comment coordinate_system {coordinate_system}\n"
              "comment unit meter\n" f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "w", encoding="ascii", newline="\n") as file:
        file.write(header)
        np.savetxt(file, np.column_stack((points, colors)),
                   fmt="%.9f %.9f %.9f %d %d %d")
    LOGGER.info("PLY write done path=%s elapsed=%.3fs size=%d", path,
                perf_counter() - started, Path(path).stat().st_size)


def save_point_cloud(path, frame):
    """Save a Depth-camera point cloud as colored ASCII PLY (meters)."""
    K, dist = camera(frame.get("depth_dir", DEFAULT_DEPTH_DIR))
    points, x, y = depth_cloud(frame["depth"], K, dist, frame["depth_scale"],
                               z_min=frame.get("z_min", .15), z_max=frame.get("z_max", 2.0))
    colors = np.full((len(points), 3), 180, np.uint8)
    plane = frame["depth_plane"]
    selected = plane["mask"][y, x] > 0 if plane is not None else np.zeros(len(points), bool)
    colors[selected] = (0, 255, 0)
    write_ply(path, points, colors, "depth_camera")
    return len(points), int(selected.sum())


def calibration_values(frame, result):
    if "depth_K" in result:
        return (np.asarray(result["depth_K"]), np.asarray(result["depth_dist"]),
                np.asarray(result["target_K"]), np.asarray(result["target_dist"]),
                result["depth_scale"])
    Kd, dd = camera(frame.get("depth_dir", DEFAULT_DEPTH_DIR))
    Kt, dt = camera(frame.get("target_dir", DEFAULT_TARGET_DIR))
    return Kd, dd, Kt, dt, frame["depth_scale"]


def project_depth_to_right(frame, result):
    """Transform Depth points to target coordinates and project them to target pixels."""
    Kd, dd, Kt, dt, scale = calibration_values(frame, result)
    points, _, _ = depth_cloud(frame["depth"], Kd, dd, scale,
                               z_min=frame.get("z_min", .15), z_max=frame.get("z_max", 2.0))
    source_z = points[:, 2].copy()
    points = points @ np.asarray(result["R"]).T + np.asarray(result["t"])
    if not len(points):
        return points, np.empty((0, 2)), source_z
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), Kt, dt)[0].reshape(-1, 2)
    height, width = frame["right"].shape[:2]
    valid = ((points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) &
             (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    return points[valid], uv[valid], source_z[valid]


def align_depth_to_right(frame, result):
    """Create a Right-IR-sized Z map with a nearest-surface z-buffer."""
    points, uv, source_z = project_depth_to_right(frame, result)
    height, width = frame["right"].shape[:2]
    base = np.floor(uv).astype(int)
    u = np.r_[base[:, 0], base[:, 0] + 1, base[:, 0], base[:, 0] + 1]
    v = np.r_[base[:, 1], base[:, 1], base[:, 1] + 1, base[:, 1] + 1]
    z = np.tile(source_z, 4)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[valid], v[valid], z[valid]
    flat = v * width + u
    order = np.argsort(z, kind="stable")
    pixels, first = np.unique(flat[order], return_index=True)
    aligned = np.zeros((height, width), np.float32)
    aligned.flat[pixels] = z[order[first]]
    return aligned


def save_ir_point_cloud(path, frame, result):
    """Save target-coordinate points colored from the selected target image."""
    points, uv, _ = project_depth_to_right(frame, result)
    pixels = np.rint(uv).astype(int)
    image = frame["right"]
    rgb = (np.repeat(image[..., None], 3, axis=2) if image.ndim == 2
           else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    write_ply(path, points, rgb[pixels[:, 1], pixels[:, 0]], "target_camera")
    return len(points)


def save_aligned_point_cloud(path, aligned, frame, result, colors=None):
    _, _, K, dist, _ = calibration_values(frame, result)
    points, x, y = depth_cloud(aligned, K, dist, 1.0,
                               z_min=frame.get("z_min", .15), z_max=frame.get("z_max", 2.0))
    if colors is None:
        image = frame["right"]
        colors = (np.repeat(image[..., None], 3, axis=2) if image.ndim == 2
                  else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    write_ply(path, points, colors[y, x], "target_camera")
    return len(points)


def depth_plane(depth, K, dist, scale, z_min=.15, z_max=2.0, roi=None):
    sample, sx, sy = depth_cloud(depth, K, dist, scale, 4, z_min, z_max, roi)
    organized = np.full((depth.shape[0] // 4, depth.shape[1] // 4, 3), np.nan)
    organized[sy // 4, sx // 4] = sample
    valid_yx = np.argwhere(np.isfinite(organized[..., 2]))
    if len(valid_yx) < 3:
        return None
    rng, models = np.random.default_rng(0), []

    for _ in range(1400):
        ay, ax = valid_yx[rng.integers(len(valid_yx))]
        offsets = rng.integers(-22, 23, (2, 2))
        by, bx = ay + offsets[0, 0], ax + offsets[0, 1]
        cy, cx = ay + offsets[1, 0], ax + offsets[1, 1]
        if not (0 <= by < organized.shape[0] and 0 <= bx < organized.shape[1] and
                0 <= cy < organized.shape[0] and 0 <= cx < organized.shape[1]):
            continue
        p = organized[[ay, by, cy], [ax, bx, cx]]
        if not np.isfinite(p).all():
            continue
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        if length < 1e-7:
            continue
        normal /= length
        h = normal @ p[0]
        if h > 0:
            normal, h = -normal, -h
        models.append((normal, h))
        if len(models) == 400:
            break

    if not models:
        return None
    normals = np.asarray([model[0] for model in models])
    offsets = np.asarray([model[1] for model in models])
    counts = np.zeros(len(models), int)
    for start in range(0, len(models), 40):
        stop = start + 40
        counts[start:stop] = np.sum(abs(sample @ normals[start:stop].T - offsets[start:stop]) < .005, axis=0)
    candidates = []
    for index in np.argsort(counts)[::-1]:
        normal, h = normals[index], offsets[index]
        if counts[index] < 150:
            break
        if any(n @ normal > np.cos(np.deg2rad(3)) and abs(d - h) < .02 for _, n, d in candidates):
            continue
        candidates.append((counts[index], normal, h))
        if len(candidates) == 15:
            break

    cloud, x, y = depth_cloud(depth, K, dist, scale, z_min=z_min, z_max=z_max, roi=roi)
    best = None
    for _, normal, h in candidates:
        raw = abs(cloud @ normal - h) < .006
        mask = np.zeros(depth.shape, np.uint8)
        mask[y[raw], x[raw]] = 1
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 800:
                continue
            selected = raw & (labels[y, x] == label)
            points = cloud[selected]
            if len(points) < 500:
                continue
            center = points.mean(0)
            normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
            if normal @ center > 0:
                normal = -normal
            error = abs((points - center) @ normal)
            points = points[error < .004]
            center = points.mean(0)
            normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
            if normal @ center > 0:
                normal = -normal
            covariance = np.cov(points, rowvar=False)
            size = np.sqrt(12 * np.maximum(np.linalg.eigvalsh(covariance)[1:], 0))
            size_error = np.linalg.norm((np.sort(size) - np.sort(BOARD_M)) / BOARD_M)
            rmse = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))
            score = size_error + .2 * rmse / .004
            if best is None or score < best[0]:
                actual = np.zeros(depth.shape, np.uint8)
                keep = selected.copy()
                keep[selected] = abs((cloud[selected] - center) @ normal) < .004
                actual[y[keep], x[keep]] = 1
                best = score, center, normal, actual, rmse, size, len(points) / stats[label, cv2.CC_STAT_AREA]

    if best is None or best[0] > .55:
        return None
    _, center, normal, mask, rmse, size, ratio = best
    return {"center": center, "normal": normal, "h": float(normal @ center),
            "mask": mask, "axis": project_axis(center, normal, K, dist),
            "rmse": rmse, "inliers": int(mask.sum()), "ratio": float(ratio),
            "size": size.tolist()}


def analyze_frame(stamp, depth_dir=DEFAULT_DEPTH_DIR, target_dir=DEFAULT_TARGET_DIR,
                  z_min=.15, z_max=2.0, roi_calibration=None, roi_margin=.20):
    depth_dir, target_dir = Path(depth_dir), Path(target_dir)
    pair = frame_pairs(depth_dir, target_dir).get(stamp)
    if pair is None:
        raise FileNotFoundError(stamp)
    depth_path, target_path = pair
    right = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if right is not None and right.ndim == 3 and right.shape[2] == 4:
        right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)
    color_paths = [p for p in depth_dir.glob(f"*{stamp}*") if "depth_color" in p.stem.lower()]
    color = cv2.imread(str(color_paths[0]), cv2.IMREAD_COLOR) if color_paths else None
    if right is None or depth is None:
        raise FileNotFoundError(stamp)
    Kd, dd = camera(depth_dir)
    Kr, dr = camera(target_dir)
    metadata = list(depth_dir.glob(f"*{stamp}*.json"))
    meta = json.loads(metadata[0].read_text()) if metadata else {}
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    rp = right_plane(gray, Kr, dr)
    scale = meta.get("depth_scale_m_per_unit", .001)
    roi = target_to_depth_roi(rp, roi_calibration, Kd, dd, depth.shape, roi_margin)
    dp = depth_plane(depth, Kd, dd, scale, z_min, z_max, None if roi is None else roi["mask"])
    return {"right": right, "depth": depth, "color": color,
            "depth_plane": dp, "right_plane": rp, "depth_scale": scale,
            "depth_roi": roi, "z_min": z_min, "z_max": z_max,
            "depth_dir": depth_dir, "target_dir": target_dir,
            "depth_path": depth_path, "target_path": target_path}


def rotation_from_normals(source, target):
    u, _, vt = np.linalg.svd(source.T @ target)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R


def calibrate(frames=None, result_path=RESULT_PATH):
    started = perf_counter()
    if frames is None:
        frames = {stamp: analyze_frame(stamp) for stamp in timestamps()}
    LOGGER.info("RT calibration start frames=%d output=%s", len(frames), result_path)
    if not frames:
        raise RuntimeError("No matched frames")
    pairs = []
    for frame in frames.values():
        if frame["depth_plane"] and frame["right_plane"]:
            pairs.append((frame["depth_plane"], frame["right_plane"]))
    if len(pairs) < 6:
        raise RuntimeError(f"Not enough plane pairs: {len(pairs)}")

    nd = np.asarray([d["normal"] for d, _ in pairs])
    nr = np.asarray([r["normal"] for _, r in pairs])
    hd = np.asarray([d["h"] for d, _ in pairs])
    hr = np.asarray([r["h"] for _, r in pairs])
    rng, best = np.random.default_rng(0), None
    for _ in range(2000):
        sample = rng.choice(len(pairs), 3, replace=False)
        R = rotation_from_normals(nd[sample], nr[sample])
        predicted = nd @ R.T
        if np.linalg.matrix_rank(predicted[sample]) < 3 or np.linalg.cond(predicted[sample]) > 1e4:
            continue
        t = np.linalg.lstsq(predicted[sample], hr[sample] - hd[sample], rcond=None)[0]
        angle = np.degrees(np.arccos(np.clip(np.sum(predicted * nr, axis=1), -1, 1)))
        distance = abs(hd + predicted @ t - hr)
        idx = np.flatnonzero((angle < 1.5) & (distance < .008))
        score = (len(idx), -(angle[idx].mean() if len(idx) else 999))
        if best is None or score > best[0]:
            best = score, R, t, idx
    if best is None:
        raise RuntimeError("RANSAC failed")
    _, R, t, idx = best
    if len(idx) < 6:
        raise RuntimeError(f"Not enough RANSAC inliers: {len(idx)}")

    R = rotation_from_normals(nd[idx], nr[idx])
    predicted = nd[idx] @ R.T
    t = np.linalg.lstsq(predicted, hr[idx] - hd[idx], rcond=None)[0]

    def residual(x):                            # 비선형 최적화에 사용될, 오차를 구하는 함수
        rotation = cv2.Rodrigues(x[:3])[0]      # 회전 벡터 -> R행렬
        predicted = nd[idx] @ rotation.T
        return np.r_[(predicted - nr[idx]).ravel() / .01,
                     (hd[idx] + predicted @ x[3:] - hr[idx]) / .003] 

    x0 = np.r_[cv2.Rodrigues(R)[0].ravel(), t] # ransac으로 구해진 초기 Rt
    x = least_squares(residual, x0, method="lm").x # LM 비선형 최적화
    R, t = cv2.Rodrigues(x[:3])[0], x[3:]
    predicted = nd[idx] @ R.T
    angle = np.degrees(np.arccos(np.clip(np.sum(predicted * nr[idx], axis=1), -1, 1)))
    distance = hd[idx] + predicted @ t - hr[idx]
    condition = float(np.linalg.cond(predicted))
    RT = np.block([[R, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    first_frame = next(iter(frames.values()))
    depth_dir = Path(first_frame.get("depth_dir", DEFAULT_DEPTH_DIR))
    target_dir = Path(first_frame.get("target_dir", DEFAULT_TARGET_DIR))
    result = {"from": depth_dir.name, "to": target_dir.name, "translation_unit": "m",
              "depth_intrinsic_source": str(depth_dir / "Intrinsic.txt"),
              "target_intrinsic_source": str(target_dir / "Intrinsic.txt"),
              "square_size_m": SQUARE_M,
              "board_size_m": list(BOARD_M), "R": R.tolist(), "t": t.tolist(), "RT": RT.tolist(),
              "plane_pairs": len(pairs), "ransac_inliers": len(idx),
              "nonlinear_method": "LM",
              "normal_condition": condition,
              "normal_rmse_deg": float(np.sqrt(np.mean(angle ** 2))),
              "plane_rmse_m": float(np.sqrt(np.mean(distance ** 2)))}
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("RT calibration done optimizer=LM pairs=%d inliers=%d elapsed=%.3fs",
                len(pairs), len(idx), perf_counter() - started)
    return result


if __name__ == "__main__":
    print(json.dumps(calibrate(), indent=2))
