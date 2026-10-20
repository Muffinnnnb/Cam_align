"""Fixed-support NSL board fitting. No Color ROI, RT, or fitted edge lines."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import distance_transform_edt
from itertools import combinations


def quadrilateral_support(hull):
    """Largest inscribed quadrilateral / hull area; shape evidence, not a pose."""
    vertices = hull.reshape(-1, 2)
    if len(vertices) < 4:
        return 0.
    if len(vertices) > 28:
        vertices = vertices[np.linspace(0, len(vertices)-1, 28).astype(int)]
    polygons = vertices[np.array(list(combinations(range(len(vertices)), 4)))]
    area = abs(np.sum(polygons[...,0]*np.roll(polygons[...,1], -1, axis=1)
                      -polygons[...,1]*np.roll(polygons[...,0], -1, axis=1), axis=1))/2
    return float(area.max()/max(1.,cv2.contourArea(hull)))


def fit_plane(points):
    center = points.mean(0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    if normal @ center > 0:
        normal = -normal
    return center, normal, float(normal @ center)


def cell_means(points, bins=8):
    """Equal spatial weight: dense/bright patches must not dominate a normal."""
    center = points.mean(0)
    axes = np.linalg.svd(points-center, full_matrices=False)[2][:2]
    uv = (points-center) @ axes.T
    lo, hi = np.percentile(uv, [1, 99], axis=0)
    cells = np.clip(((uv-lo)/np.maximum(hi-lo, 1e-9)*bins).astype(int), 0, bins-1)
    key = cells[:, 1]*bins+cells[:, 0]
    if points.dtype == np.float64 and points.shape[1] == 3:
        # Same occupied cells, order and per-cell weights; no point subsampling.
        # Accumulate once instead of scanning/copying every point for each cell.
        count = np.bincount(key)
        occupied = count > 0
        sums = np.column_stack([np.bincount(key, weights=points[:, axis])
                                for axis in range(3)])
        return sums[occupied]/count[occupied, None]
    # Preserve the original accumulation precision for float32/other inputs.
    return np.array([points[key == k].mean(0) for k in np.unique(key)])


def complete_support(grid, seeds, quality_function):
    """A RANSAC patch locates a surface; refit its spatial support before RT.

    The hull only proposes where to read ORIGINAL points, never synthesizes
    points or measures coverage. Gross 30 mm source-only outliers are removed;
    failed expansion falls back to the original seed, never to a trial RT.
    Untrimmed error and rejected fraction remain available as diagnostics.
    """
    valid = np.isfinite(grid).all(2)
    if not valid.any():
        return []
    distance = np.linalg.norm(grid, axis=2)
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    smooth = cv2.medianBlur(np.where(valid, distance, distance[tuple(nearest)]).astype('float32'), 5)
    gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    filled = cv2.morphologyEx(valid.astype('uint8'), cv2.MORPH_CLOSE, np.ones((5,5),'uint8'))
    boundary = cv2.morphologyEx(filled, cv2.MORPH_GRADIENT, np.ones((3,3),'uint8')) > 0
    edge_distance = distance_transform_edt(~((gradient > .10) | boundary))
    # Keep each hypothesis independent. Transitive unions bridged the board to
    # hands/body; the resulting hull then discarded ALL good board hypotheses.
    # A failed completion must never destroy another seed of the same board.
    found = []
    for seed in seeds:
        y, x = np.where(seed['mask'])
        if len(x) < 100:
            continue
        hull = cv2.convexHull(np.c_[x, y].astype('int32'))
        # Finite target identity only, never a normal/pose fitted from image lines.
        # Round body patches can be planar and have a rectangular bounding box.
        quad_support = quadrilateral_support(hull)
        perimeter = np.zeros(valid.shape, np.uint8)
        cv2.polylines(perimeter, [hull], True, 1, 1)
        edge_support = float(np.mean(edge_distance[perimeter > 0] <= 3))
        region = np.zeros(valid.shape, np.uint8)
        cv2.fillConvexPoly(region, hull, 1)
        region = cv2.erode(region, np.ones((3, 3), np.uint8)).astype(bool)
        support = region & valid
        points = grid[support]
        if len(points) < 100:
            continue
        center, normal, h = fit_plane(cell_means(points))
        keep = abs(points @ normal-h) < .030
        fallback = keep.mean() < .85
        if fallback:
            # Expansion may cross another surface or contain ToF outliers.
            # Preserve the independently detected seed instead of deleting it.
            # Still read every original point and require a majority of support.
            original = grid[seed['mask'] & valid]
            center, normal, h = fit_plane(cell_means(original))
            keep = abs(points @ normal-h) < .030
        if keep.mean() < .50 or keep.sum() < 100:
            continue
        mask = support.copy()
        mask[support] = keep
        points = grid[mask]
        center, normal, h = fit_plane(cell_means(points))
        quality = quality_function(points, normal)
        quality['rmse'] = float(np.sqrt(np.mean((points @ normal-h)**2)))
        full_rmse = float(np.sqrt(np.mean((grid[support] @ normal-h)**2)))
        reasons = list(quality['quality_reasons'])
        if quality['rmse'] > .006 and not any(s.startswith('평면 잔차') for s in reasons):
            reasons.append(f"평면 잔차 {quality['rmse']*1000:.1f} mm > 6 mm")
        if full_rmse > .010:
            reasons.append(f'전체 지지점 잔차 {full_rmse*1000:.1f} mm > 10 mm')
        # Actual occupied cells, not the area of a convex hull.
        axes = np.linalg.svd(points-center, full_matrices=False)[2][:2]
        uv = (points-center) @ axes.T
        rect = cv2.minAreaRect(uv.astype('float32'))
        board_center = center + np.array(rect[0]) @ axes
        box = cv2.boxPoints(rect)
        edge = box[1]-box[0]
        angle = np.arctan2(edge[1], edge[0])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        local = (uv-np.array(rect[0])) @ rot
        lo, hi = local.min(0), local.max(0)
        cells = np.clip(((local-lo)/np.maximum(hi-lo, 1e-9)*8).astype(int), 0, 7)
        occupancy = len(np.unique(cells[:, 0]+8*cells[:, 1]))/64
        if occupancy < .75:
            reasons.append(f'실측 지지 셀 {occupancy:.0%} < 75% (빈 영역을 채우지 않음)')
        if edge_support < .55:
            reasons.append(f'거리 경계 지지 {edge_support:.0%} < 55% (보드 일부/다른 물체 가능)')
        image_rect = cv2.minAreaRect(hull)
        rectangularity = cv2.contourArea(hull)/max(1, np.prod(image_rect[1]))
        score = (np.linalg.norm((np.array(quality['size'])-[.25,.28])/[.25,.28])
                 + full_rmse/.030 + 2*(1-rectangularity) + 2*(1-edge_support))
        found.append({**quality, 'normal': normal, 'h': h, 'center': center,
                      'board_center': board_center, 'mask': mask, 'support_mask': support,
                      'full_rmse': full_rmse, 'support_occupancy': occupancy,
                      'edge_support': edge_support,
                      'quad_support': float(quad_support),
                      'completion_fallback': bool(fallback),
                      'support_inlier_fraction': float(keep.mean()),
                      'quality_ok': not reasons, 'quality_reasons': reasons,
                      'inliers': len(points), 'score': float(score),
                      'detector_version': 6,
                      'fit_method': 'fixed spatial support / cell-balanced SVD'})
    found.sort(key=lambda p: p['score'])
    unique = []
    for plane in found:
        # Keep distinct surfaces, not fifteen differently tilted fits of one board.
        if any(np.count_nonzero(p['mask'] & plane['mask']) /
               max(1, np.count_nonzero(p['mask'] | plane['mask'])) > .60 for p in unique):
            continue
        unique.append(plane)
    return unique


def support_quality(points, plane):
    """Structural eligibility, separate from millimetre-level accuracy warnings.

    Noise is relative to the measured in-plane spread. A noisy 2-D board is
    usable; a thin strip, 3-D blob, or unsupported fragment is not.
    """
    values = np.linalg.eigvalsh(np.cov(points.T))
    flatness = float(np.sqrt(max(0, values[0]) / max(values[1], 1e-12)))
    ratio = np.sort(plane['size']) / [.25, .28]
    reasons = []
    if len(points) < 100 or np.any(ratio < .70) or np.any(ratio > 1.40):
        reasons.append('보드 크기/실측 지지점 부족 (250×280 mm의 70~140%)')
    if flatness > .20:
        reasons.append(f'평면 두께/접선 분산 비 {flatness:.2f} > 0.20')
    if plane.get('rectangularity', 0) < .70 or plane.get('support_occupancy', 0) < .75:
        reasons.append('실측 지지영역 불완전')
    if plane.get('edge_support', 0) < .55:
        reasons.append('독립 거리 경계 지지 부족')
    if plane.get('normal_stability_deg', 0) > 15:
        reasons.append('영역별 법선 차이 > 15° (굽음/다른 물체 가능)')
    if plane.get('quad_support', 1.) < .80:
        reasons.append('판의 사각형 지지영역 불충분 (몸/혼합 영역 가능)')
    # Fixed, source-only uncertainty; never adjusted to match a trial RT.
    noise = max(.006, float(plane.get('full_rmse', plane['rmse'])))
    weight = float(np.clip((.008 / noise)**2, .25, 1.))
    return dict(eligible=not reasons, reasons=reasons, flatness=flatness, weight=weight)


def fit_point_planes(records, seed=0, trials=1600, weights=None, starts=8, expand_consensus=True,
                     use_centers=False, finite_board=True):
    """Multi-candidate frame RANSAC + linear-loss LM on FIXED source support.

    Repeated names are alternative objects, not independent observations.
    At most one candidate per frame votes/fits. No points are trimmed by RT.
    Finite-board overflow prevents sliding without equating partial/full centers.
    The retained 20mm gates are provisional engineering limits, not a measured
    sensor confidence interval. use_centers is for historical ablation only.
    """
    groups = {}
    for i, r in enumerate(records):
        groups.setdefault(r[0], []).append(i)
    group_indices = list(groups.values())
    if len(groups) < 6:
        raise RuntimeError('고정 지지영역 평면쌍이 6개 이상 필요합니다.')
    if weights is None or isinstance(weights, dict):
        weights = [1. if weights is None else weights[r[0]] for r in records]
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(records),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Frame weights must be finite and positive.')
    points = [cell_means(r[1]) for r in records]
    ns = np.array([fit_plane(p)[1] for p in points])
    hs = np.array([fit_plane(p)[2] for p in points])
    nt = np.array([r[2]['normal'] for r in records])
    ht = np.array([r[2]['h'] for r in records])
    centers = np.array([r[3] for r in records])
    targets = np.array([r[2]['center'] for r in records])
    means = np.array([p.mean(0) for p in points])
    covariance = np.array([(p-p.mean(0)).T@(p-p.mean(0))/len(p) for p in points])
    bounds = []
    for r in records:
        corners = r[2].get('corners3d') if finite_board else None
        if corners is None:
            bounds.append(None)
        else:
            corners = np.asarray(corners)
            edges = np.array([corners[1]-corners[0], corners[3]-corners[0]])
            lengths = np.linalg.norm(edges, axis=1)
            bounds.append((corners.mean(0), edges/lengths[:,None], lengths/2))

    def outside(R, t, i):
        if bounds[i] is None:
            return np.zeros((len(points[i]), 2))
        center, axes, half = bounds[i]
        return np.maximum(abs((points[i]@R.T+t-center)@axes.T)-half, 0)

    def errors(R, t):
        n = nt @ R
        bias = np.sum(means*n, axis=1)+nt@t-ht
        rms = np.sqrt(np.maximum(0, np.einsum('ni,nij,nj->n', n, covariance, n)+bias**2))
        center_error = (np.linalg.norm(centers@R.T+t-targets, axis=1) if use_centers else
                        np.array([np.sqrt(np.mean(np.sum(outside(R,t,i)**2,axis=1))) for i in range(len(records))]))
        return rms, center_error

    def consensus(rms, ce):
        loss = (rms/.020)**2 + (ce/(.060 if use_centers else .020))**2
        keep = np.zeros(len(records), bool)
        for indices in group_indices:
            best = indices[int(np.argmin(loss[indices]))]
            keep[best] = rms[best] < .020 and ce[best] < (.060 if use_centers else .020)
        return keep, loss

    rng, hypotheses = np.random.default_rng(seed), {}
    for _ in range(trials):
        idx = [rng.choice(group_indices[i]) for i in rng.choice(len(groups), 3, replace=False)]
        if np.linalg.cond(nt[idx]) > 50:
            continue
        if not use_centers or rng.random() < .5:
            u, _, vt = np.linalg.svd(nt[idx].T @ ns[idx])
            R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            t = np.linalg.lstsq(nt[idx], ht[idx]-hs[idx], rcond=None)[0]
        else:
            a, b = centers[idx], targets[idx]
            u, _, vt = np.linalg.svd((b-b.mean(0)).T @ (a-a.mean(0)))
            R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            t = b.mean(0)-R@a.mean(0)
        rms, ce = errors(R, t)
        keep, _ = consensus(rms, ce)
        score = (int(keep.sum()), -float(np.median(rms[keep])) if keep.any() else -1.)
        if keep.sum() >= 6:
            key = keep.tobytes()
            if key not in hypotheses or score > hypotheses[key][0]:
                hypotheses[key] = score, R, t, keep
            if len(hypotheses) > starts:
                hypotheses.pop(min(hypotheses, key=lambda k: hypotheses[k][0]))
    if not hypotheses:
        raise RuntimeError('Point-to-plane RANSAC 인라이어 부족. 기존 RT를 유지합니다.')
    solutions = []
    # Polish several distinct consensus sets. A promising raw hypothesis can
    # lose many frames after LM; judging only its BEFORE-fit count was unstable.
    for _, R, t, keep in hypotheses.values():
        initial = int(keep.sum())
        x = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
        seen = set()
        while keep.sum() >= 6 and keep.tobytes() not in seen and len(seen) < 20:
            seen.add(keep.tobytes())
            def residual(x):
                R = cv2.Rodrigues(x[:3])[0]
                plane = np.concatenate([((p@R.T+x[3:])@nt[i]-ht[i])*np.sqrt(weights[i]/len(p))
                                        for i, p in enumerate(points) if keep[i]])
                finite = np.concatenate([(outside(R,x[3:],i)*np.sqrt(weights[i]/len(points[i]))).ravel()
                                         for i in np.flatnonzero(keep)])
                if use_centers:
                    finite = ((centers[keep]@R.T+x[3:]-targets[keep])*np.sqrt(weights[keep,None])).ravel()
                return np.r_[plane, finite]
            fit = least_squares(residual, x, method='lm', max_nfev=300)
            if not fit.success or not np.isfinite(fit.x).all():
                break
            x = fit.x
            R, t = cv2.Rodrigues(x[:3])[0], x[3:]
            rms, ce = errors(R, t)
            final, _ = consensus(rms, ce)
            if not expand_consensus:
                final &= keep
            if np.array_equal(final, keep):
                condition = float(np.linalg.cond(nt[keep]))
                if condition <= 30 and np.linalg.matrix_rank(fit.jac) == 6:
                    score = (int(keep.sum()), -float(np.mean((rms[keep]/.020)**2+(ce[keep]/(.060 if use_centers else .020))**2)))
                    solutions.append((score, R, t, keep, rms, ce, condition, initial))
                break
            keep = final
    if not solutions:
        raise RuntimeError('LM 후 인라이어/자세 다양성 부족. 기존 RT를 유지합니다.')
    _, R, t, keep, rms, ce, condition, initial = max(solutions, key=lambda s: s[0])
    angles = np.degrees(np.arccos(np.clip(np.sum((ns@R.T)*nt, axis=1), -1, 1)))
    loss = (rms/.020)**2+(ce/(.060 if use_centers else .020))**2
    selected = {name: int(ids[int(np.argmin(loss[ids]))]) for name,ids in groups.items()}
    RT = np.eye(4); RT[:3,:3] = R; RT[:3,3] = t
    return dict(R=R.tolist(), t=t.tolist(), RT=RT.tolist(), nonlinear_method='LM',
                calibration_objective='fixed support, cell-balanced point-to-plane + finite board overflow (no partial-center equality)',
                lm_loss='linear', consensus_rechecked=expand_consensus,
                plane_pairs=len(groups), candidate_count=len(records), ransac_initial_inliers=initial, ransac_inliers=int(keep.sum()),
                ransac_inlier_frames=[r[0] for r,k in zip(records,keep) if k],
                ransac_outlier_frames=[name for name,ids in groups.items() if not keep[ids].any()],
                selected_record_indices=selected,
                frame_weights={name: float(weights[i]) for name,i in selected.items()},
                ransac_hypotheses_refined=len(hypotheses),
                ransac_point_rms_threshold_m=.020, ransac_board_overflow_threshold_m=.020,
                threshold_policy='provisional 20mm RMS gates; not device-characterized confidence limits',
                center_constraint=bool(use_centers),
                normal_rmse_deg=float(np.sqrt(np.mean(angles[keep]**2))),
                plane_rmse_m=float(np.sqrt(np.mean(rms[keep]**2))), normal_condition=condition)
