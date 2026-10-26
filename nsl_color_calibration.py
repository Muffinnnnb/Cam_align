from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app_logging import get_logger
from app_paths import output_dir
from calibrate_d_to_r import (
    BOARD_M, SQUARE_M, BOARD_LABEL, camera, calibrate, right_plane,
)
from fuse_color_pointcloud import (
    DEFAULT_CAPTURE_ROOT, DEFAULT_COLOR_CAL, capture_pairs, fuse_points,
    read_source_cloud, write_binary_ply, write_image,
)
from nsl_geometry import geometry_spec, check_geometry, source_geometry_warning, detected_lens, fusion_block_reason, corrected_points, ray_scale
from hashlib import sha256
from nsl_plane_fit import complete_support, fit_point_planes, support_quality, cell_means
from nsl_surface_support import (range_support, paired_support, plane_conditioned_support,
                                 metric_support, local_support, basin_support)
from nsl_projection_fit import observation, outline_errors, refine_projection
from nsl_dimensions import estimate_source_correction, measured_size
from nsl_board_identity import identify_board, frame_fingerprint, IDENTITY_VERSION
from nsl_overlay import point_overlay


RESULT_PATH = output_dir("calibration") / "nsl_to_color_rt.json"
LOGGER = get_logger("nsl_color")
PLANE_CANDIDATE_RGB = (0, 255, 0)  # Visibility only, NOT an RT/ground-truth pass.


def _fit_plane(points: np.ndarray):
    center = points.mean(axis=0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    if normal @ center > 0:
        normal = -normal
    return center, normal, float(normal @ center)


def plane_quality(points, normal):
    """Independent support/size checks; no target ROI or trial RT is used."""
    center = points.mean(0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    local = (points - center) @ axes[:2].T
    # Trim isolated edge/flying pixels before measuring the physical footprint.
    lo, hi = np.percentile(local, [1, 99], axis=0)
    inner = local[np.all((local >= lo) & (local <= hi), axis=1)]
    rect = cv2.minAreaRect(inner.astype(np.float32))
    size = np.sort(rect[1])
    hull_area = cv2.contourArea(cv2.convexHull(inner.astype(np.float32)))
    rectangularity = hull_area / max(float(np.prod(size)), 1e-12)
    rmse = float(np.std((points - center) @ normal))
    ratio = size / np.sort(BOARD_M)
    reasons = []
    if np.any(ratio < .70) or np.any(ratio > 1.30):
        reasons.append(f"보드 지지폭 {size[0]*1000:.0f}×{size[1]*1000:.0f} mm (기대 {BOARD_LABEL})")
    if rectangularity < .70:
        reasons.append(f"지지영역 불완전 {rectangularity:.0%}")
    if rmse > .006:
        reasons.append(f"평면 잔차 {rmse*1000:.1f} mm > 6 mm")
    # Check stability across spatial halves; a narrow/noisy patch can fit locally
    # yet yield a materially different normal on the other half of the board.
    angles = []
    for axis in range(2):
        for side in (-1, 1):
            part = points[side * (local[:, axis] - np.median(local[:, axis])) >= 0]
            if len(part) >= 50:
                n = _fit_plane(part)[1]
                angles.append(np.degrees(np.arccos(np.clip(abs(n @ normal), -1, 1))))
    stability = float(max(angles, default=0))
    if stability > 5:
        reasons.append(f"영역별 법선 불안정 {stability:.1f}° > 5°")
    return {"size": size.tolist(), "rectangularity": float(rectangularity),
            "normal_stability_deg": stability, "quality_ok": not reasons,
            "quality_reasons": reasons, "rmse": rmse}


def nsl_plane_candidates(xyz_mm: np.ndarray, shape: tuple[int, int],
                         z_min=.10, z_max=20.0):
    height, width = shape
    grid = xyz_mm.reshape(height, width, 3) * .001
    valid = np.isfinite(grid).all(axis=2)
    valid &= (grid[..., 2] >= z_min) & (grid[..., 2] <= z_max)
    valid_yx = np.argwhere(valid)
    if len(valid_yx) < 200:
        return []

    points = grid[valid]
    rng = np.random.default_rng(0)
    models = []
    # A local least-squares seed is less sensitive to ToF noise than a normal
    # from just three neighboring pixels. Original XYZ is still used for fits.
    for _ in range(220):
        ay, ax = valid_yx[rng.integers(len(valid_yx))]
        radius = int(rng.integers(5, 18))
        patch = grid[max(0, ay-radius):ay+radius+1, max(0, ax-radius):ax+radius+1]
        good = valid[max(0, ay-radius):ay+radius+1, max(0, ax-radius):ax+radius+1]
        if good.sum() < 50:
            continue
        local = patch[good]
        center, normal, h = _fit_plane(local)
        if np.std((local-center) @ normal) < .008:
            models.append((normal, h))
    for _ in range(2400):
        ay, ax = valid_yx[rng.integers(len(valid_yx))]
        offsets = rng.integers(-28, 29, (2, 2))
        by, bx = ay + offsets[0, 0], ax + offsets[0, 1]
        cy, cx = ay + offsets[1, 0], ax + offsets[1, 1]
        if not (0 <= by < height and 0 <= bx < width and
                0 <= cy < height and 0 <= cx < width):
            continue
        if not (valid[by, bx] and valid[cy, cx]):
            continue
        p = grid[[ay, by, cy], [ax, bx, cx]]
        if not np.isfinite(p).all():
            continue
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        if length < 1e-7:
            continue
        normal /= length
        h = float(normal @ p[0])
        if h > 0:
            normal, h = -normal, -h
        models.append((normal, h))
        if len(models) >= 500:
            break
    if not models:
        return []
    normals = np.asarray([n for n, _ in models])
    offsets = np.asarray([h for _, h in models])
    counts = np.zeros(len(models), np.int32)
    for start in range(0, len(models), 40):
        stop = min(start + 40, len(models))
        counts[start:stop] = np.sum(
            np.abs(points @ normals[start:stop].T - offsets[start:stop]) < .008,
            axis=0,
        )

    candidates = []
    for index in np.argsort(counts)[::-1]:
        normal, h = normals[index], offsets[index]
        if counts[index] < 120:
            break
        if any(n @ normal > np.cos(np.deg2rad(4)) and abs(d - h) < .03
               for _, n, d in candidates):
            continue
        candidates.append((counts[index], normal, h))
        if len(candidates) == 32:
            break

    found = []
    for _, normal0, h0 in candidates:
        mask = valid & (np.abs(grid @ normal0 - h0) < .015)
        closed = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 120:
                continue
            selected = mask & (labels == label)
            component = grid[selected]
            if len(component) < 100:
                continue

            center, normal, h = _fit_plane(component)
            error = np.abs((component - center) @ normal)
            selected[selected.copy()] = error < .008
            component = grid[selected]
            if len(component) < 100:
                continue
            center, normal, h = _fit_plane(component)
            # Expand from the seed once, then refit all consistent neighboring
            # support. Do not report only a tiny RANSAC strip as the whole board.
            expanded = valid & (labels == label) & (np.abs(grid @ normal - h) < .012)
            component = grid[expanded]
            center, normal, h = _fit_plane(component)
            selected = expanded & (np.abs(grid @ normal - h) < .010)
            component = grid[selected]
            if len(component) < 100:
                continue
            center, normal, h = _fit_plane(component)
            quality = plane_quality(component, normal)
            size = np.asarray(quality['size'])
            size_error = np.linalg.norm(
                (np.sort(size) - np.sort(BOARD_M)) / np.asarray(BOARD_M)
            )
            rmse = float(np.sqrt(np.mean(((component - center) @ normal) ** 2)))
            score = size_error + .20 * rmse / .008
            final_mask = selected.copy()
            found.append((score, {
                "center": center, "normal": normal, "h": h, "mask": final_mask,
                "rmse": rmse, "inliers": int(len(component)), "size": size.tolist(),
                "size_score": float(size_error),
                **quality,
            }))

    # Precision warnings are not target identity. Do not favor a tiny quiet
    # patch over a complete but noisier board before spatial support is fitted.
    found.sort(key=lambda item: item[0])
    unique = []
    for score, plane in found:
        if score > 2.00:
            continue
        if any(np.count_nonzero(old['mask'] & plane['mask']) /
               max(1, np.count_nonzero(old['mask'] | plane['mask'])) > .65
               and abs(float(old['normal'] @ plane['normal'])) > np.cos(np.deg2rad(4))
               for old in unique):
            continue
        unique.append(plane)
        if len(unique) == 24:
            break
    grid = np.where(valid[..., None], grid, np.nan)
    planes = complete_support(grid, unique, plane_quality)
    planes += range_support(grid, plane_quality)
    planes += paired_support(grid, planes, plane_quality)
    # Preserve successful hypotheses and RT inputs. Only use this alternative
    # when image-grid proposals cannot supply any structurally usable board.
    if not any(support_quality(grid[p['mask']], p)['eligible'] for p in planes):
        planes += metric_support(grid, plane_quality)
    if not any(support_quality(grid[p['mask']], p)['eligible'] for p in planes):
        planes += plane_conditioned_support(grid, planes, plane_quality)
    planes.sort(key=lambda p: p['score'])
    planes += local_support(grid, planes, plane_quality)
    return planes + basin_support(grid, planes, plane_quality)


def nsl_plane(xyz_mm: np.ndarray, shape: tuple[int, int],
              z_min=.10, z_max=20.0):
    candidates = nsl_plane_candidates(xyz_mm, shape, z_min, z_max)
    grid = xyz_mm.reshape(*shape, 3)*.001
    return next((p for p in candidates if support_quality(grid[p['mask']], p)['eligible']), None)


def pointcloud_view(xyz_mm: np.ndarray, shape: tuple[int, int],
                    plane=None, z_min=.10, z_max=20.0,
                    plane_color=(0, 255, 0)):
    grid = xyz_mm.reshape(*shape, 3)
    z = grid[..., 2] * .001
    valid = np.isfinite(z) & (z >= z_min) & (z <= z_max)
    gray = np.zeros(shape, np.uint8)
    if np.any(valid):
        values = z[valid]
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        gray[valid] = np.uint8(np.clip((z[valid] - lo) * 255 / (hi - lo), 0, 255))
    view = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    if plane is not None:
        mask = plane["mask"].astype(bool)
        color = np.asarray(plane_color, np.float32)
        view[mask] = np.rint(view[mask] * .25 + color * .75).astype(np.uint8)
    return view


def color_plane_view(image: np.ndarray, plane) -> np.ndarray:
    out = image.copy()
    if plane is None:
        return out
    overlay = out.copy()
    cv2.fillConvexPoly(overlay, np.rint(plane["region"]).astype(np.int32), (0, 255, 0))
    return cv2.addWeighted(overlay, .25, out, .75, 0)
def analyze_pair(pair: dict, color_cal=DEFAULT_COLOR_CAL,
                 z_min=.10, z_max=20.0, geometry='pcd'):
    started = perf_counter()
    image = cv2.imdecode(np.fromfile(pair["color"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(pair["color"])
    xyz_mm, intensity, shape = read_source_cloud(pair, geometry)
    lens = detected_lens(xyz_mm, shape) if geometry == 'pcd' else None
    k, dist = camera(color_cal)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Always automatic. Historical annotation files are preserved but NOT read.
    source_candidates = nsl_plane_candidates(xyz_mm, shape, z_min, z_max)
    grid = xyz_mm.reshape(*shape, 3)*.001
    for candidate in source_candidates:
        candidate['support_quality'] = support_quality(grid[candidate['mask']], candidate)
    source_plane = next((p for p in source_candidates if p['support_quality']['eligible']), None)
    target_plane = right_plane(gray, k, dist)
    frame = {
        "timestamp": pair["timestamp"], "color": image,
        "xyz_mm": xyz_mm, "intensity": intensity, "pcd_shape": shape,
        "source_candidates": source_candidates,
        "source_plane": source_plane, "target_plane": target_plane,
        "source_path": pair["cloud"], "target_path": pair["color"],
        "range_path": pair.get('range'), "source_geometry": geometry_spec(geometry),
        "source_lens": lens,
        "geometry_warning": source_geometry_warning(xyz_mm, shape, geometry, lens),
        "target_dir": Path(color_cal), "z_min": z_min, "z_max": z_max,
    }
    assess_support(frame)
    frame['board_identity'] = identify_board(frame, None, k, dist)
    LOGGER.info(
        "NSL frame analyzed stamp=%s geometry=%s source_plane=%s target_plane=%s elapsed=%.3fs",
        pair["timestamp"], geometry, source_plane is not None, target_plane is not None,
        perf_counter() - started,
    )
    return frame


def board_candidates(frame, for_calibration=False):
    """Source-only target identity, with an explicit 'none confirmed' outcome."""
    candidates = frame.get('source_candidates') or []
    if for_calibration:
        candidates = [p for p in candidates if p.get('calibration_eligible', True)]
    if not candidates or not candidates[0].get('detector_version'):
        return candidates  # Historical synthetic/legacy compatibility.
    grid = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)*.001
    for p in candidates:
        if 'support_quality' not in p:
            p['support_quality'] = support_quality(grid[p['mask']], p)
    return [p for p in candidates if p['support_quality']['eligible']]


def assess_support(frame):
    plane = frame.get('source_plane')
    if plane is None:
        result = {'eligible': False, 'reasons': ['NSL 보드 검출 실패']}
    elif plane.get('fit_method') and 'size' in plane:
        points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[np.asarray(plane['mask'], bool)]*.001
        result = support_quality(points, plane)
    else:
        result = {'eligible': plane.get('quality_ok', False), 'reasons': plane.get('quality_reasons', [])}
    frame['calibration_support'] = result
    return result


def support_signature(frame, plane):
    """Training provenance must match the actual points and Color plane, not a filename."""
    points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[np.asarray(plane['mask'], bool)]
    target = frame['target_plane']
    digest = sha256(np.asarray(points, dtype='<f8').tobytes())
    digest.update(np.asarray(plane['mask'], bool).tobytes())
    digest.update(np.asarray(np.r_[target['normal'], target['h'], target['center']], dtype='<f8').tobytes())
    return digest.hexdigest()


def analyze_capture(root=DEFAULT_CAPTURE_ROOT, color_cal=DEFAULT_COLOR_CAL,
                    z_min=.10, z_max=20.0, geometry='pcd'):
    return {
        pair["timestamp"]: analyze_pair(pair, color_cal, z_min, z_max, geometry)
        for pair in capture_pairs(Path(root))
    }


def plane_projection_metrics(frame: dict, plane: dict, calibration: dict,
                             color_cal=DEFAULT_COLOR_CAL, margin=0.):
    empty = {"overlap": 0., "coverage": 0., "iou": 0., "outside_fraction": 1., "margin": float(margin)}
    if plane is None or frame.get("target_plane") is None:
        return empty
    grid = np.asarray(frame["xyz_mm"], np.float64).reshape(*frame["pcd_shape"], 3) * .001
    points = grid[np.asarray(plane["mask"]).astype(bool)]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return empty

    R = np.asarray(calibration["R"], np.float64)
    t = np.asarray(calibration["t"], np.float64)
    transformed = corrected_points(points, calibration) @ R.T + t
    K, dist = camera(color_cal)
    uv = cv2.projectPoints(
        transformed, np.zeros(3), np.zeros(3), K, dist
    )[0].reshape(-1, 2)

    height, width = frame["color"].shape[:2]
    finite = np.isfinite(uv).all(axis=1) & (transformed[:, 2] > 0)
    safe_uv = np.clip(np.nan_to_num(uv, nan=-100000, posinf=-100000, neginf=-100000), -100000, 100000)
    u = np.rint(safe_uv[:, 0]).astype(np.int32)
    v = np.rint(safe_uv[:, 1]).astype(np.int32)
    inside = (
        finite & (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )
    if not np.any(inside):
        return empty

    polygon = np.asarray(frame["target_plane"]["region"], np.float64)
    board = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(board, np.rint(polygon).astype(np.int32), 1)
    # Count measured support in board cells. A few boundary points must not
    # turn an empty middle into 100% coverage through a filled convex hull.
    undistorted = cv2.undistortPoints(safe_uv[finite].reshape(-1,1,2), K, dist).reshape(-1,2)
    corners = cv2.undistortPoints(polygon.reshape(-1,1,2), K, dist).reshape(-1,2)
    H = cv2.getPerspectiveTransform(corners.astype('float32'), np.float32([[0,0],[1,0],[1,1],[0,1]]))
    local = cv2.perspectiveTransform(undistorted.reshape(-1,1,2), H).reshape(-1,2)
    cells = np.floor(np.clip(local, -100, 100)*8).astype(int)
    cells = np.unique(cells, axis=0)
    board_cells = np.all((cells >= 0) & (cells < 8), axis=1)
    intersection = np.count_nonzero(board_cells)
    coverage = intersection / 64
    iou = intersection / max(64 + len(cells) - intersection, 1)
    center = polygon.mean(axis=0)
    polygon = center + (polygon - center) * (1.0 + margin)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    # Points outside the Color FOV count as non-overlap as well.
    strict_overlap = float(np.count_nonzero(board[v[inside], u[inside]]) / len(points))
    return {"overlap": float(np.count_nonzero(mask[v[inside], u[inside]]) / len(points)),
            "coverage": float(coverage), "iou": float(iou),
            "outside_fraction": 1-strict_overlap, "margin": float(margin)}


def plane_projection_overlap(frame, plane, calibration, color_cal=DEFAULT_COLOR_CAL, margin=0.):
    return plane_projection_metrics(frame, plane, calibration, color_cal, margin)["overlap"]




def guided_plane_from_color(frame: dict, calibration: dict,
                            color_cal=DEFAULT_COLOR_CAL,
                            plane_tolerance=.050, min_points=80,
                            max_angle_deg=12.0, max_plane_error_m=.050,
                            min_coverage=.35):
    """ROI/visibility-guided point RANSAC; fit the measured plane independently.

    A broad 50 mm prior only selects a surface neighborhood. The 8 mm fit
    residual is measured against each sampled NSL plane, not the Color plane.
    A tight prior slab would select points to agree with the RT being tested.
    """
    target = frame.get("target_plane")
    if target is None:
        return None

    xyz = np.asarray(frame["xyz_mm"], np.float64) * .001
    valid = np.isfinite(xyz).all(axis=1)
    valid &= (xyz[:, 2] >= frame["z_min"]) & (xyz[:, 2] <= frame["z_max"])
    valid_indices = np.flatnonzero(valid)
    source = corrected_points(xyz[valid], calibration)
    if len(source) < min_points:
        return None

    R = np.asarray(calibration["R"], np.float64)
    t = np.asarray(calibration["t"], np.float64)
    transformed = source @ R.T + t
    K, dist = camera(color_cal)
    uv = cv2.projectPoints(
        transformed, np.zeros(3), np.zeros(3), K, dist
    )[0].reshape(-1, 2)

    height, width = frame["color"].shape[:2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    inside = (
        (transformed[:, 2] > 0) & (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    board_mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(
        board_mask,
        np.rint(target["region"]).astype(np.int32),
        1,
    )
    roi = np.zeros(len(source), bool)
    roi[inside] = board_mask[v[inside], u[inside]] > 0

    # Resolve visibility among ALL in-FOV points before the board/depth gates.
    visible = np.flatnonzero(inside)
    keys = v[visible].astype(np.int64) * width + u[visible]
    order = np.argsort(transformed[visible, 2], kind="stable")
    _, first = np.unique(keys[order], return_index=True)
    visible_mask = np.zeros(len(source), bool)
    visible_mask[visible[order[first]]] = True
    roi &= visible_mask
    signed = (
        transformed - np.asarray(target["center"], np.float64)
    ) @ np.asarray(target["normal"], np.float64)
    roi &= np.abs(signed) <= plane_tolerance
    local_indices = np.flatnonzero(roi)
    if len(local_indices) < min_points:
        return None

    # Estimate a plane from NSL points instead of trimming directly to the prior.
    points = source[local_indices]
    rng = np.random.default_rng(0)
    triplets = points[rng.integers(len(points), size=(512, 3))]
    normals = np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-7
    normals = normals[usable] / lengths[usable, None]
    offsets = np.sum(normals * triplets[usable, 0], axis=1)
    signs = np.where(offsets > 0, -1.0, 1.0)
    normals, offsets = normals * signs[:, None], offsets * signs
    predicted = normals @ R.T
    angles = np.degrees(np.arccos(np.clip(predicted @ target["normal"], -1, 1)))
    plausible = (angles < max_angle_deg) & (abs(offsets + predicted @ t - target["h"]) < max_plane_error_m)
    normals, offsets = normals[plausible], offsets[plausible]
    if not len(normals):
        return None
    counts = np.array([np.count_nonzero(abs(points @ n - h) <= .008)
                       for n, h in zip(normals, offsets)])
    best = int(np.argmax(counts))
    if counts[best] < max(min_points, .45 * len(points)):
        return None
    working_indices = local_indices[abs(points @ normals[best] - offsets[best]) <= .008]
    points = source[working_indices]
    center, normal, h = _fit_plane(points)
    for _ in range(3):
        residual = np.abs((points - center) @ normal)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = float(np.clip(
            median + 3.0 * 1.4826 * mad, .004, .008
        ))
        keep = residual <= threshold
        if np.count_nonzero(keep) < min_points:
            return None
        working_indices = working_indices[keep]
        points = source[working_indices]
        center, normal, h = _fit_plane(points)

    predicted = R @ normal
    angle = float(np.degrees(np.arccos(np.clip(
        predicted @ target["normal"], -1, 1
    ))))
    plane_error = float(abs(h + predicted @ t - target["h"]))
    if angle >= max_angle_deg or plane_error >= max_plane_error_m:
        return None

    final_flat_indices = valid_indices[working_indices]
    mask = np.zeros(len(xyz), np.uint8)
    mask[final_flat_indices] = 1
    mask = mask.reshape(frame["pcd_shape"])
    rmse = float(np.sqrt(np.mean(((points - center) @ normal) ** 2)))

    projected = transformed[working_indices]
    corners = np.asarray(target["corners3d"], np.float64)
    edge_x = corners[1] - corners[0]
    edge_y = corners[3] - corners[0]
    board_w = max(float(np.linalg.norm(edge_x)), 1e-9)
    board_h = max(float(np.linalg.norm(edge_y)), 1e-9)
    axis_x, axis_y = edge_x / board_w, edge_y / board_h
    delta = projected - corners.mean(axis=0)
    local_x = delta @ axis_x
    local_y = delta @ axis_y
    coverage_x = float(
        (np.percentile(local_x, 95) - np.percentile(local_x, 5)) / board_w
    )
    coverage_y = float(
        (np.percentile(local_y, 95) - np.percentile(local_y, 5)) / board_h
    )
    eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
    if min(coverage_x, coverage_y) < min_coverage or eigenvalues[1] < .08 * eigenvalues[2]:
        return None

    return {
        "center": center, "normal": normal, "h": float(h),
        "mask": mask, "rmse": rmse, "inliers": int(len(points)),
        "size": [board_w * coverage_x, board_h * coverage_y],
        "guided": True,
        "guided_angle_deg": angle,
        "guided_plane_error_m": plane_error,
        "guided_plane_tolerance_m": float(plane_tolerance),
        "coverage_x": coverage_x,
        "coverage_y": coverage_y,
    }


def calibrate_nsl_color(frames: dict, result_path=RESULT_PATH,
                        capture_root=DEFAULT_CAPTURE_ROOT,
                        color_cal=DEFAULT_COLOR_CAL, progress=None):
    def status(message):
        LOGGER.info(message)
        if progress:
            progress(message)

    if not frames:
        raise RuntimeError('분석 가능한 촬영쌍이 없습니다.')
    geometry = next(iter(frames.values())).get('source_geometry', {'mode': 'pcd'})
    if any(f.get('source_geometry', {'mode': 'pcd'}) != geometry for f in frames.values()):
        raise ValueError('서로 다른 NSL 점군 모델을 혼합할 수 없습니다.')
    if any(fusion_block_reason(f) for f in frames.values()):
        raise RuntimeError('NF 원본은 현재 장비의 정상 보정 입력이 아닙니다. WF 복원을 사용하세요.')
    status('NSL에서 독립 검출한 전체 지지영역 고정 중...')
    records, all_records, excluded, references, weights = [], [], {}, [], []
    for name, frame in frames.items():
        candidates = board_candidates(frame, for_calibration=True)
        plane = candidates[0] if candidates else None
        frame['source_plane'] = plane
        if plane is None or frame.get('target_plane') is None:
            excluded[str(name)] = ['RT 학습용 NSL 후보 또는 Color 평면 없음 (대응 보조 후보는 학습 제외)']
            continue
        grid = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)*.001
        record = (str(name), grid[plane['mask']], frame['target_plane'], plane['board_center'])
        all_records.append(record)
        quality = assess_support(frame)
        if quality['eligible']:
            for p in candidates[:4]:
                q = support_quality(grid[p['mask']], p)
                if q['eligible']:
                    records.append((str(name), grid[p['mask']], frame['target_plane'], p['board_center']))
                    references.append(p)
                    weights.append(q['weight'])
        else:
            excluded[str(name)] = quality['reasons']
    # Keep the NSL-only selection frozen: later RT candidate choices must not
    # feed back into dimensional calibration, including in held-out folds.
    dimension_input = {name: {'source_plane': f.get('source_plane'), 'xyz_mm': f['xyz_mm'],
                             'pcd_shape': f['pcd_shape']} for name, f in frames.items()}
    K, distortion = camera(color_cal)

    def fit_training(ids, train_names):
        correction, audit = (estimate_source_correction({n: dimension_input[n] for n in train_names})
                             if geometry['mode'] == 'wf_range' else (None, {'applied': False}))
        model = {'source_geometry': geometry}
        if correction:
            model['source_ray_correction'] = correction
        adjusted = [(records[i][0], corrected_points(records[i][1], model), records[i][2],
                     corrected_points(records[i][3], model)) for i in ids]
        coarse = fit_point_planes(adjusted, weights=[weights[i] for i in ids])
        coarse.update(model)
        items = [observation(frames[name], references[ids[index]])
                 for name, index in coarse['selected_record_indices'].items()]
        result = refine_projection(items, coarse, K, distortion) if any(p is not None for p in items) else coarse
        result['nsl_dimension_audit'] = audit
        return result

    status('NSL 독립 치수 검사 → 고정 광선 모델 → RANSAC / 영상 경계 + 평면 LM 계산 중...')
    current = fit_training(list(range(len(records))), set(frames))
    chosen = current['selected_record_indices']
    for name, index in chosen.items():
        frames[name]['source_plane'] = references[index]
        assess_support(frames[name])
    current.update({
        'from': 'nsl_pointcloud', 'to': 'color', 'translation_unit': 'm',
        'source_type': 'NSL organized point cloud', 'target_type': 'RealSense color',
        'square_size_m': SQUARE_M, 'board_size_m': list(BOARD_M),
        'source_geometry': geometry, 'calibration_version': 13,
        'capture_root': str(Path(capture_root).resolve()),
        'target_intrinsic_source': str(Path(color_cal)/'Intrinsic.txt'),
        'target_distortion_source': str(Path(color_cal)/'Distortion.txt'),
        'target_intrinsics': camera(color_cal)[0].tolist(),
        'target_distortion': camera(color_cal)[1].tolist(),
        'z_range_m': [next(iter(frames.values()))['z_min'], next(iter(frames.values()))['z_max']],
        'input_frame_count': len(frames), 'projection_overlap_threshold': .80,
        'eligible_frame_count': len(chosen), 'support_excluded_frames': excluded,
        'eligibility_policy': 'independent source hypotheses; structural support + quadrilateral footprint; identity-only basin proposals excluded from RT/dimension learning; explicit no-board outcome',
        'projection_coverage_threshold': .75,
        'source_geometry_check': 'fixed full support / raw residual / occupied cells / depth discontinuities / plane-conditioned, local-metric and range-basin fallback',
        'guided_refinement': 'fixed source candidates for training; separate finite-board association for display; no RT-based point trimming',
    })
    if 'color' in next(iter(frames.values())):
        current['target_image_size'] = list(next(iter(frames.values()))['color'].shape[1::-1])
    # Hold out contiguous TIME BLOCKS, not alternating neighboring frames.
    # Test support is source-only candidate #1, NEVER selected by the trial RT.
    rows, errors, identity_models = [], [], []
    names = sorted(frames)
    if len(chosen) >= 12:
        for fold in range(4):
            status(f'시간 구간 분리 교차검증 {fold+1}/4...')
            held_out = {name for i,name in enumerate(names) if min(3, i*4//len(names)) == fold}
            train_ids = [i for i,r in enumerate(records) if r[0] not in held_out]
            train = [records[i] for i in train_ids]
            test = [r for r in all_records if r[0] in held_out]
            try:
                baseline = fit_point_planes(train, weights=[weights[i] for i in train_ids])
                trial = fit_training(train_ids, set(names)-held_out)
                identity_models.append({
                    **{k: trial[k] for k in ('R', 't', 'source_geometry', 'source_ray_correction') if k in trial},
                    'training_frames': sorted(set(names)-held_out),
                    'heldout_fingerprints': {n: frame_fingerprint(frames[n]) for n in held_out},
                })
                R, t = np.array(trial['R']), np.array(trial['t'])
                for name, points, target, center in test:
                    transformed = corrected_points(points, trial)@R.T+t
                    residual = transformed@target['normal']-target['h']
                    old_residual = (points@np.asarray(baseline['R']).T+baseline['t'])@target['normal']-target['h']
                    row = {'name': name, 'fold': fold,
                                 'baseline_point_rms_mm': float(np.sqrt(np.mean(old_residual**2))*1000),
                                 'point_rms_mm': float(np.sqrt(np.mean(residual**2))*1000),
                                 'center_error_mm': float(np.linalg.norm(corrected_points(center, trial)@R.T+t-target['center'])*1000)}
                    # Test the untouched source-only footprint, not a candidate
                    # preferred by the held-out RT. No manual annotations read.
                    f = frames[name]
                    p = dimension_input[name]['source_plane']
                    if item := observation(f, p):
                        row['boundary_mean_px'] = outline_errors(item, trial, K, distortion)['boundary_mean_px']
                        row['baseline_boundary_mean_px'] = outline_errors(item, baseline, K, distortion)['boundary_mean_px']
                        row['outside_fraction'] = plane_projection_metrics(f, p, trial, color_cal)['outside_fraction']
                        row['baseline_outside_fraction'] = plane_projection_metrics(f, p, baseline, color_cal)['outside_fraction']
                    row['source_ray_scale'] = ray_scale(trial)
                    rows.append(row)
            except RuntimeError as exc:
                errors.append(f'fold {fold}: {exc}')
                LOGGER.warning('Cross validation failed: %s', errors[-1])
    if errors:
        # Do not overwrite the active result after a failed holdout fit.
        raise RuntimeError('시간 구간 분리 검증 실패. 기존 RT/설정 유지: ' + '; '.join(errors))
    values = np.array([[r['point_rms_mm'], r['center_error_mm']] for r in rows])
    current['cross_validation'] = {
        'description': '4 contiguous time-block holdouts; source-only test support; not an external dataset or RGB ground truth',
        'split': 'contiguous_time_blocks',
        'unconfirmed_frames': [name for name in names if name not in {r[0] for r in all_records}],
        'evaluated': len(rows), 'rows': rows, 'failures': errors,
        'point_rms_median_mm': float(np.median(values[:,0])) if len(values) else None,
        'center_median_mm': float(np.median(values[:,1])) if len(values) else None,
        'point_rms_p90_mm': float(np.percentile(values[:,0],90)) if len(values) else None,
        'center_p90_mm': float(np.percentile(values[:,1],90)) if len(values) else None,
    }
    pixel_rows = [r for r in rows if 'boundary_mean_px' in r]
    if current.get('projection_refinement', {}).get('applied') and not pixel_rows:
        raise RuntimeError('영상 경계 RT의 분리 촬영 검증 자료가 부족합니다. 기존 RT/설정을 유지합니다.')
    if pixel_rows and current.get('projection_refinement', {}).get('applied'):
        v = current['cross_validation']
        for key in ('boundary_mean_px', 'baseline_boundary_mean_px', 'outside_fraction',
                    'baseline_outside_fraction', 'baseline_point_rms_mm'):
            v[key+'_median'] = float(np.median([r[key] for r in pixel_rows]))
        passed = (v['boundary_mean_px_median'] < v['baseline_boundary_mean_px_median']
                  and v['point_rms_median_mm'] <= v['baseline_point_rms_mm_median']+3)
        v['projection_improved'] = passed
        if not passed:
            raise RuntimeError('분리 촬영의 경계/깊이 오차 개선 확인 실패. 기존 RT/설정을 유지합니다.')
    current['fitted_plane_selection'] = {
        name: {'candidate_index': next(i for i,p in enumerate(frames[name]['source_candidates']) if p is frames[name]['source_plane']), 'normal': frames[name]['source_plane']['normal'].tolist(),
               'h': frames[name]['source_plane']['h'],
               'support_signature': support_signature(frames[name], frames[name]['source_plane'])}
        for name in chosen}
    current['board_identity_models'] = identity_models
    current['board_identity_version'] = IDENTITY_VERSION
    current['board_identity_policy'] = ('Fixed source hypotheses associated in metric board coordinates; '
        'known frames use time-block-excluded models; source-only surface continuity distinguishes complementary fragments; '
        'source-only range components and independent pair supports refitted before RT; '
        'prefer complete hypotheses of the same object; no RT-based point trimming or candidate feedback into RT fitting.')
    current['final_validated_frames'] = []
    for frame in frames.values():
        select_plane_for_rt(frame, current, color_cal)
        if frame.get('plane_valid'):
            current['final_validated_frames'].append(frame['timestamp'])
    states = [f['board_identity']['state'] for f in frames.values()]
    current['board_association_summary'] = {
        'denominator': len(states), 'matched': states.count('matched'),
        'partial': states.count('partial'), 'matched_rate': states.count('matched')/len(states),
        'partial_included': False, 'metric': 'matched boards / all input frames; not RT precision',
    }
    current['calibration_warning'] = (
        '검출 후보와 RT 학습 이력은 별도입니다. 시간 분리 검증도 내부 일관성 검사이며 절대 정확도 검증은 아닙니다. '
        'NSL 치수 보정은 RGB/RT와 독립 추정이며 개별 장비 공장 보정값이 아닙니다. '
        '비동기 촬영/원본 점군 왜곡을 RT 하나로 제거할 수 없습니다. '
        '교차검증과 실제 Color 시점 투영을 확인하세요.')
    status(f"자동 계산 완료: 전체 {len(frames)} / 학습 후보 {len(chosen)} / "
           f"실제 사용 {current['ransac_inliers']} / 제외 {len(frames)-current['ransac_inliers']}")
    if result_path is not None:
        from datetime import datetime
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup = path.with_name(path.stem+'.previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
            backup.write_bytes(path.read_bytes())
        current['result_path'] = str(path.resolve())
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
        LOGGER.info('Point/plane RT saved: %s cross_validation=%s', path, current['cross_validation'])
    return current


def select_plane_for_rt(frame: dict, calibration: dict,
                        color_cal=DEFAULT_COLOR_CAL):
    """Preserve source-only support; record board association separately for display.

    Training support, source detection and displayed association are distinct.
    None is external ground truth, and association cannot change RT training.
    """
    check_geometry(frame, calibration)
    if (frame.get('target_plane') or {}).get('corners3d') is not None:
        frame['board_identity'] = identify_board(frame, calibration, *camera(color_cal))
    elif 'board_identity' in frame:
        frame['board_identity'] = dict(state='unverified', plane=None,
                                      basis='Color 보드 검출 없음: 대응 확인 불가')
    if identity := frame.get('board_identity'):
        LOGGER.info('Board association stamp=%s state=%s candidate=%s basis=%s evidence=%s reason=%s',
                    frame['timestamp'], identity['state'], identity.get('index'), identity['basis'],
                    identity.get('evidence'), identity.get('reason'))
    target = frame.get("target_plane")
    candidates = board_candidates(frame)
    frame.update(plane_valid=False, rt_consistent=False, rt_inlier=False,
                 calibration_plane=None, fitted_plane_reproduced=False, training_record_known=False)
    for key in ("rt_normal_error_deg", "rt_plane_error_m", "projection_overlap", "projection_overlap_best",
                "projection_coverage", "projection_iou", "quality_failures", "rt_point_rms_m", "rt_cell_rms_m", "rt_center_error_m", "projection_outside", "boundary_mean_px", "boundary_p90_px", "corrected_source_size_m"):
        frame.pop(key, None)
    if target is None or not candidates:
        frame["source_plane"] = candidates[0] if candidates else None
        assess_support(frame)
        return None

    # Source-only ranking, including legacy data without structural metadata.
    plane = candidates[0] if candidates[0].get('fit_method') else next(
        (p for p in candidates if p.get('quality_ok')), candidates[0])
    frame['source_plane'] = plane
    assess_support(frame)

    saved = calibration.get("fitted_plane_selection")
    record = (saved or {}).get(frame["timestamp"])
    same_capture = (not calibration.get("capture_root") or not frame.get("source_path")
                    or Path(frame["source_path"]).parent.parent.resolve() == Path(calibration["capture_root"]).resolve())
    if not same_capture:
        record = None
    def matches(p):
        if not record:
            return False
        if record.get('support_signature'):
            return support_signature(frame, p) == record['support_signature']
        return (np.allclose(p['normal'], record['normal'], atol=1e-6)
                and np.isclose(p['h'], record['h'], atol=1e-6))

    learned = next((p for p in candidates if matches(p)), None)
    frame['calibration_plane'] = learned
    frame['fitted_plane_reproduced'] = learned is not None
    frame['training_record_known'] = same_capture and learned is not None
    frame['rt_inlier'] = (frame['training_record_known']
                         and frame['timestamp'] in calibration.get('ransac_inlier_frames', []))

    R, t = np.asarray(calibration["R"]), np.asarray(calibration["t"])
    angle_limit = float(calibration.get("ransac_angle_threshold_deg", 3.0))
    distance_limit = float(calibration.get("ransac_plane_threshold_m", .020))
    overlap_limit = float(calibration.get("projection_overlap_threshold", .80))
    projection = plane_projection_metrics(frame, plane, calibration, color_cal)
    overlap = projection['overlap']
    source_normal, source_h = plane['normal'], plane['h']
    if ray_scale(calibration) != 1.:
        points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[plane['mask']]*.001
        _, source_normal, source_h = _fit_plane(cell_means(corrected_points(points, calibration)))
    predicted = R @ source_normal
    angle = float(np.degrees(np.arccos(np.clip(predicted @ target['normal'], -1, 1))))
    distance = float(abs(source_h + predicted @ t - target['h']))
    frame.update(source_plane=plane, rt_normal_error_deg=angle, rt_plane_error_m=distance,
                 projection_overlap=overlap, projection_overlap_best=overlap,
                 projection_coverage=projection['coverage'], projection_iou=projection['iou'],
                 projection_outside=projection.get('outside_fraction', 1-overlap))
    if item := observation(frame, plane):
        frame.update(outline_errors(item, calibration, *camera(color_cal)))
    if plane.get('fit_method'):
        points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[plane['mask']]*.001
        points = corrected_points(points, calibration)
        if ray_scale(calibration) != 1.:
            frame['corrected_source_size_m'] = measured_size(points).tolist()
        residual = (points@R.T+t)@target['normal']-target['h']
        frame['rt_point_rms_m'] = float(np.sqrt(np.mean(residual**2)))
        balanced = cell_means(points)
        frame['rt_cell_rms_m'] = float(np.sqrt(np.mean(((balanced@R.T+t)@target['normal']-target['h'])**2)))
        frame['rt_center_error_m'] = float(np.linalg.norm(corrected_points(plane['board_center'], calibration)@R.T+t-target['center']))
    failures = list(plane.get('quality_reasons', ['지지영역 품질 미검사']))
    if angle >= angle_limit:
        failures.append(f'RT 법선 {angle:.2f}° ≥ {angle_limit:g}°')
    if calibration.get('calibration_version', 0) < 6 and distance >= distance_limit:
        failures.append(f'평면 오프셋 차이 {distance*1000:.1f} mm ≥ {distance_limit*1000:g} mm (거리 아님)')
    if overlap < overlap_limit:
        failures.append(f'보드 안 투영점 {overlap:.0%} < {overlap_limit:.0%}')
    coverage_limit = float(calibration.get('projection_coverage_threshold', .75))
    if projection['coverage'] < coverage_limit:
        failures.append(f"보드 실측 셀 면적 커버 {projection['coverage']:.0%} < {coverage_limit:.0%}")
    if projection['iou'] < .50:
        failures.append(f"보드 교집합/합집합 {projection['iou']:.0%} < 50%")
    if frame.get('rt_point_rms_m', 0) >= .020:
        failures.append(f"전체 점-평면 RMS {frame['rt_point_rms_m']*1000:.1f} mm ≥ 20 mm")
    if calibration.get('calibration_version', 0) < 6 and frame.get('rt_center_error_m', 0) >= .030:
        failures.append(f"보드 중심 정합 {frame['rt_center_error_m']*1000:.1f} mm ≥ 30 mm")
    frame['quality_failures'] = failures
    LOGGER.info('RT consistency (not ground truth) stamp=%s training_used=%s same_display_support=%s full_rms=%s occupied_cells=%.3f notes=%s',
                frame['timestamp'], frame['rt_inlier'], learned is plane,
                plane.get('full_rmse'), projection['coverage'], failures)
    frame['rt_consistent'] = plane.get('quality_ok', False) and not failures
    frame['plane_valid'] = frame['rt_consistent']  # Legacy API, not a detection verdict.
    return plane


def plane_status(frame: dict, calibration=None):
    """Board association state; RT training membership is reported separately."""
    training = ('used' if frame.get('rt_inlier') else 'excluded') if frame.get('training_record_known') else 'unknown'
    training_text = {'used': '사용', 'excluded': '제외', 'unknown': '미확인/미계산'}[training]
    if frame.get('target_plane') is None or not frame.get('source_candidates'):
        return {'state': 'failed', 'detected': False, 'training': training,
                'message': 'NSL 후보 또는 Color 보드 검출 없음. RT 정확도 판정과 별개.'}
    if identity := frame.get('board_identity'):
        state = identity['state']
        return {'state': {'matched': 'candidate', 'partial': 'partial'}.get(state, 'unconfirmed'),
                'detected': state in ('matched', 'partial'), 'training': training,
                'identity': state, 'evidence': identity.get('evidence'), 'basis': identity['basis'],
                'candidate_index': identity.get('index'),
                'compatible_fragments': identity.get('compatible_fragments', []),
                'identity_version': identity.get('version', 1),
                'message': f"{identity.get('reason', identity['basis'])} / {identity['basis']} / "
                f'RT 학습 이력: {training_text}. 대응 확인은 정밀도 합격 판정이 아닙니다.'}
    if frame.get("target_plane") is None or not board_candidates(frame):
        return {'state': 'failed', 'detected': False, 'training': training,
                'message': '평면쌍 검출 불가 (NSL 후보 또는 Color 체스보드 없음). RT 정확도 판정과 별개.'}
    quality = frame.get('calibration_support') or assess_support(frame)
    note = '' if quality['eligible'] else ' / 구조 품질 참고: ' + '; '.join(quality['reasons'])
    return {'state': 'candidate', 'detected': True, 'training': training,
            'message': f'NSL 독립 평면 후보 + Color 체스보드 검출 / RT 학습 이력: {training_text}. '
                       '물체 식별 및 정합 정확도는 보증하지 않음.' + note}


def project_to_color(frame: dict, calibration: dict, color_cal=DEFAULT_COLOR_CAL,
                     visible_only=True, return_indices=False):
    check_geometry(frame, calibration)
    if reason := fusion_block_reason(frame):
        raise ValueError(reason)
    k, dist = camera(color_cal)
    if calibration.get('target_image_size') and list(frame['color'].shape[1::-1]) != calibration['target_image_size']:
        raise ValueError('Color 영상 해상도가 보정 당시와 다릅니다. 내부 파라미터 확인 후 재보정하세요.')
    if ('target_intrinsics' in calibration and not np.allclose(k, calibration['target_intrinsics'], rtol=0, atol=1e-8)) or (
            'target_distortion' in calibration and not np.allclose(dist, calibration['target_distortion'], rtol=0, atol=1e-8)):
        raise ValueError('Color 내부 파라미터가 RT 계산 당시와 다릅니다. 해당 폴더로 다시 보정하세요.')
    r = np.asarray(calibration["R"], np.float64)
    t = np.asarray(calibration["t"], np.float64)
    key = (tuple(r.ravel()), tuple(t), tuple(k.ravel()), tuple(dist.ravel()), ray_scale(calibration),
           frame['z_min'], frame['z_max'], visible_only, id(frame['xyz_mm']), id(frame['color']))
    cached = frame.get('_color_projection')
    if cached is not None and cached[0] == key:
        return cached[1] if return_indices else cached[1][:-1]
    result = fuse_points(
        frame["xyz_mm"], frame["intensity"], frame["color"], k, dist, r, t,
        frame["z_min"], frame["z_max"], visible_only, True,
        organized_shape=frame["pcd_shape"], source_ray_scale=ray_scale(calibration),
    )
    frame['_color_projection'] = key, result
    frame['projection_stats'] = result[5]
    return result if return_indices else result[:-1]


def aligned_depth(frame: dict, calibration: dict, color_cal=DEFAULT_COLOR_CAL):
    points, _, _, u, v, _ = project_to_color(frame, calibration, color_cal, True)
    height, width = frame["color"].shape[:2]
    depth = np.zeros((height, width), np.float32)
    if len(points):
        depth[v, u] = points[:, 2]
    return depth
def displayed_board(frame):
    identity = frame.get('board_identity')
    return identity['plane'] if identity is not None else frame.get('source_plane')


def save_source_cloud(path: Path, frame: dict, inspection=False, include_intensity=False, calibration=None):
    xyz = frame["xyz_mm"] * .001
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] >= frame["z_min"]) & (xyz[:, 2] <= frame["z_max"])
    if calibration is not None:
        check_geometry(frame, calibration)
    points = corrected_points(xyz[valid], calibration or {}).astype(np.float32)
    colors = np.full((len(points), 3), 180, np.uint8)
    plane = frame.get("inspection_plane") if inspection else displayed_board(frame)
    if plane is not None:
        selected = np.asarray(plane["mask"], bool).reshape(-1)[valid]
        partial = (frame.get('board_identity') or {}).get('state') == 'partial'
        colors[selected] = (0, 190, 255) if inspection or partial else PLANE_CANDIDATE_RGB
    else:
        selected = np.zeros(len(points), bool)
    intensity = frame["intensity"][valid] if include_intensity else None
    mode = frame.get('source_geometry', {'mode':'pcd'})['mode']
    scale = ray_scale(calibration or {})
    write_binary_ply(Path(path), points, colors, intensity,
                     coordinate_frame=f"NSL camera / geometry {mode} / ray_angle_scale {scale:.9f} / before RT")
    LOGGER.info("Source PLY saved path=%s points=%d selected=%d inspection=%s", path, len(points), np.count_nonzero(selected), inspection)
    return len(points), int(np.count_nonzero(selected))


def save_fused_cloud(path: Path, frame: dict, calibration: dict,
                     color_cal=DEFAULT_COLOR_CAL, include_intensity=False):
    points, rgb, intensity, _, _, stats, indices = project_to_color(
        frame, calibration, color_cal, True, return_indices=True
    )
    if not len(points):
        raise ValueError("Color 시야에 투영되는 점이 없습니다. RT·내부 파라미터·Z 범위를 확인하세요.")
    path = Path(path)
    preview = path.with_name(path.stem + "_rgb.png")
    metadata = path.with_name(path.stem + "_meta.json")
    k, distortion = camera(color_cal)
    write_binary_ply(path, points, rgb, intensity if include_intensity else None)
    write_image(preview, rgb_source_image(frame["pcd_shape"], indices, rgb))
    stats = {**stats, "includes_intensity": bool(include_intensity), "preview": str(preview),
             "metadata": str(metadata), "timestamp": frame.get("timestamp"),
             "source_cloud": str(frame.get("source_path", "")),
             "source_geometry": frame.get('source_geometry', {'mode': 'pcd'}),
             "nf_comparison_only": bool(frame.get('allow_nf_comparison') and frame.get('source_lens') == 'nf'),
             "source_range": str(frame.get('range_path', '')),
             "source_color": str(frame.get("target_path", "")),
             "R": np.asarray(calibration["R"]).tolist(), "t": np.asarray(calibration["t"]).tolist(),
             "target_intrinsics": k.tolist(), "target_distortion": distortion.tolist(),
             "target_image_size": list(frame["color"].shape[1::-1]),
             "z_range_m": [frame["z_min"], frame["z_max"]], "color_source": "original Color image RGB"}
    stats.update(calibration_version=calibration.get('calibration_version'),
                 source_ray_correction=calibration.get('source_ray_correction'),
                 calibration_warning=calibration.get('calibration_warning', ''),
                 plane_validation=plane_status(frame, calibration))
    metadata.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("RGB fusion saved path=%s points=%d original_image=%s preview=%s", path, len(points), frame.get("target_path"), preview)
    return stats


def rgb_source_image(shape, indices, rgb):
    image = np.zeros((*shape, 3), np.uint8)
    image.reshape(-1, 3)[indices] = rgb[:, ::-1]
    return image


def fused_color_view(frame: dict, calibration: dict, color_cal=DEFAULT_COLOR_CAL):
    _, rgb, _, _, _, stats, indices = project_to_color(frame, calibration, color_cal, True, return_indices=True)
    return rgb_source_image(frame["pcd_shape"], indices, rgb), stats["output_points"]


def board_error_view(frame, calibration, color_cal=DEFAULT_COLOR_CAL, *,
                     point_size=1, point_shape='dot'):
    """Color-view diagnostic only; never changes RGB export or calibration input."""
    image = frame['color'].copy()
    plane, target = displayed_board(frame), frame.get('target_plane')
    if plane is None or target is None:
        return image
    check_geometry(frame, calibration)
    R, t = np.asarray(calibration['R']), np.asarray(calibration['t'])
    points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[plane['mask']]*.001
    points = corrected_points(points, calibration)@R.T+t
    K, dist = camera(color_cal)
    uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1,2)
    finite = np.isfinite(uv).all(1) & (points[:,2] > 0)
    uv = np.rint(np.clip(np.nan_to_num(uv), -100000, 100000)).astype(int)
    good = finite & (uv[:,0] >= 0) & (uv[:,0] < image.shape[1]) & (uv[:,1] >= 0) & (uv[:,1] < image.shape[0])
    error = abs(points@target['normal']-target['h'])
    colors = np.tile([0, 220, 0], (len(points),1))
    colors[error >= .010] = [0, 180, 255]
    colors[error >= .020] = [0, 0, 255]
    # A low point-plane error can still land in the background. Show it explicitly.
    outside = np.array([cv2.pointPolygonTest(np.asarray(target['region'], np.float32),
                                           (float(u), float(v)), False) < 0 for u, v in uv])
    colors[outside] = [255, 0, 255]
    image = point_overlay(image, uv[good], colors[good], points[good, 2],
                          size=point_size, shape=point_shape, opacity=1.)
    polygon = np.rint(target['region']).astype('int32')
    cv2.polylines(image, [polygon], True, (255,255,255), 1)
    cv2.putText(image, 'Plane mm: green <10 / amber 10-20 / red >=20; magenta: outside board', (8,20),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    return image


def projection_view(frame: dict, calibration: dict,
                    color_cal=DEFAULT_COLOR_CAL, *, point_size=1, point_shape='dot'):
    points, _, _, u, v, _ = project_to_color(frame, calibration, color_cal, True)
    image = frame["color"].copy()
    if len(points):
        z = points[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        q = np.uint8(np.clip((z - lo) * 255 / (hi - lo), 0, 255))
        colors = cv2.applyColorMap(q.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        image = point_overlay(image, np.c_[u, v], colors, z,
                              size=point_size, shape=point_shape)
    return image, len(points)


def save_projection(path: Path, frame: dict, calibration: dict,
                    color_cal=DEFAULT_COLOR_CAL, *, point_size=1, point_shape='dot'):
    image, count = projection_view(frame, calibration, color_cal,
                                  point_size=point_size, point_shape=point_shape)
    write_image(Path(path), image)
    return count
