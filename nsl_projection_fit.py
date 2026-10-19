"""Plane + bidirectional outline refinement. No RGB clipping or per-frame warps."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree

from nsl_geometry import corrected_points, scale_ray_angles, ray_scale
from nsl_plane_fit import cell_means, fit_plane


def sample_polygon(vertices, count=96):
    vertices = np.asarray(vertices, float)
    delta = np.roll(vertices, -1, axis=0)-vertices
    lengths = np.linalg.norm(delta, axis=1)
    distance = np.r_[0., np.cumsum(lengths)]
    positions = np.arange(count)*distance[-1]/count
    ids = np.minimum(np.searchsorted(distance, positions, side='right')-1, len(vertices)-1)
    return vertices[ids]+delta[ids]*((positions-distance[ids])/np.maximum(lengths[ids], 1e-12))[:, None]


def segment_residual(points, polygon):
    """Closest segment vectors, including along-plane error while inside the board."""
    delta = np.roll(polygon, -1, axis=0)-polygon
    offset = points[:, None]-polygon
    t = np.clip(np.sum(offset*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-12), 0, 1)
    residual = offset-t[..., None]*delta
    ids = np.argmin(np.sum(residual*residual, axis=-1), axis=1)
    return residual[np.arange(len(points)), ids]


def project(points, K, distortion):
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)


def observation(frame, plane=None):
    """Freeze source-only support; use its hull only to measure an outer footprint.

    The boundary is not a reconstructed complete board. Holes/partial support
    remain detection limitations. Normals come from 3D support, never image lines.
    """
    plane = frame.get('source_plane') if plane is None else plane
    target = frame.get('target_plane')
    if plane is None or target is None or 'corners3d' not in target:
        return None
    y, x = np.where(plane['mask'])
    if len(x) < 100:
        return None
    pixels = np.c_[x, y]
    hull = cv2.convexHull(pixels.astype('float32')).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) < 16:
        return None
    points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[y, x]*.001
    ids = cKDTree(pixels).query(sample_polygon(hull))[1]
    edge = points[ids].copy()
    edge -= (edge@plane['normal']-plane['h'])[:, None]*plane['normal']
    return dict(name=frame['timestamp'], points=cell_means(points), edge=edge, target=target)


def outline_errors(item, calibration, K, distortion):
    R, t = np.asarray(calibration['R']), np.asarray(calibration['t'])
    edge = corrected_points(item['edge'], calibration)@R.T+t
    if not np.isfinite(edge).all() or np.any(edge[:, 2] <= 0):
        return {'boundary_mean_px': float('inf'), 'boundary_p90_px': float('inf')}
    uv = project(edge, K, distortion)
    source = uv[ConvexHull(uv).vertices]
    target = project(sample_polygon(item['target']['corners3d']), K, distortion)
    errors = np.r_[np.linalg.norm(segment_residual(uv, target), axis=1),
                   np.linalg.norm(segment_residual(target, source), axis=1)]
    return dict(boundary_mean_px=float(errors.mean()), boundary_p90_px=float(np.percentile(errors, 90)))


def refine_projection(items, initial, K, distortion):
    """RANSAC initializes frame membership; linear-loss LM refines fixed support.

    8 mm / 5 px are optimization weights, not accuracy guarantees. The 20 mm
    plane gate is retained. A 15 px boundary RMS gate rejects gross bad frames
    after LM; it never removes individual points or crops the colored cloud.
    """
    used = set(initial['ransac_inlier_frames'])
    selected = [p for p in items if p is not None and p['name'] in used]
    if len(selected) < 8:
        return {**initial, 'projection_refinement': {'applied': False, 'reason': 'outline pairs < 8'}}
    target_edges = [project(sample_polygon(p['target']['corners3d']), K, distortion) for p in selected]
    x = np.r_[cv2.Rodrigues(np.asarray(initial['R']))[0].ravel(), initial['t']]
    # This value is measured from NSL dimensions BEFORE RT. RGB cannot change it.
    scale = ray_scale(initial)
    keep = np.ones(len(selected), bool)

    def components(x, i):
        item = selected[i]
        R, t = cv2.Rodrigues(x[:3])[0], x[3:6]
        points = scale_ray_angles(item['points'], scale)@R.T+t
        edge = scale_ray_angles(item['edge'], scale)@R.T+t
        uv = project(edge, K, distortion)
        # Keep float64: float32 quantization corrupts LM's numerical Jacobian.
        hull = uv[ConvexHull(uv).vertices]
        return (points@item['target']['normal']-item['target']['h'],
                segment_residual(uv, target_edges[i]), segment_residual(target_edges[i], hull))

    # Membership only shrinks, so convergence needs at most one removal per
    # observation. Four arbitrary passes can abort a valid gradual rejection.
    for _ in range(len(selected)+1):
        def residual(x):
            rows = []
            for i in np.flatnonzero(keep):
                plane, forward, reverse = components(x, i)
                rows.extend((plane/.008/np.sqrt(len(plane)), forward.ravel()/5/np.sqrt(len(forward)),
                             reverse.ravel()/5/np.sqrt(len(reverse))))
            return np.concatenate(rows)
        fit = least_squares(residual, x, method='lm', max_nfev=300)
        if not fit.success or not np.isfinite(fit.x).all() or np.linalg.matrix_rank(fit.jac) < len(x):
            raise RuntimeError('영상 경계 LM 수렴/관측성 검사 실패. 기존 RT를 유지합니다. '
                               f'status={fit.status}, nfev={fit.nfev}, rank={np.linalg.matrix_rank(fit.jac)}')
        x = fit.x
        plane_rms, edge_rms = [], []
        for i in range(len(selected)):
            a, b, c = components(x, i)
            plane_rms.append(np.sqrt(np.mean(a*a)))
            edge_rms.append(np.sqrt((np.mean(np.sum(b*b, axis=1))+np.mean(np.sum(c*c, axis=1)))/2))
        final = keep & (np.asarray(plane_rms) < .020) & (np.asarray(edge_rms) < 15)
        if final.sum() < 8:
            raise RuntimeError('영상 경계/평면 정합 인라이어 부족. 기존 RT를 유지합니다.')
        if np.array_equal(final, keep):
            break
        keep = final
    else:
        raise RuntimeError('영상 경계 LM 합의집합 미수렴. 기존 RT를 유지합니다.')
    if not .7 <= scale <= 1.3:
        raise RuntimeError('추정 광선 각도 계수가 유효 범위 밖입니다. 기존 RT를 유지합니다.')
    R, t = cv2.Rodrigues(x[:3])[0], x[3:6]
    RT = np.eye(4); RT[:3, :3] = R; RT[:3, 3] = t
    names = [p['name'] for p, good in zip(selected, keep) if good]
    normals = np.array([fit_plane(scale_ray_angles(p['points'], scale))[1] for p in selected])
    targets = np.array([p['target']['normal'] for p in selected])
    angles = np.degrees(np.arccos(np.clip(np.sum((normals@R.T)*targets, axis=1), -1, 1)))
    result = {**initial, 'R': R.tolist(), 't': t.tolist(), 'RT': RT.tolist(),
              'coarse_ransac_inlier_frames': initial['ransac_inlier_frames'],
              'ransac_inlier_frames': names, 'ransac_inliers': len(names),
              'ransac_outlier_frames': sorted(set(initial['selected_record_indices'])-set(names)),
              'plane_rmse_m': float(np.sqrt(np.mean(np.asarray(plane_rms)[keep]**2))),
              'normal_rmse_deg': float(np.sqrt(np.mean(angles[keep]**2))),
              'normal_condition': float(np.linalg.cond(targets[keep])),
              'projection_refinement': {'applied': True, 'method': 'LM', 'loss': 'linear',
                  'boundary_rms_gate_px': 15., 'plane_rms_gate_m': .020,
                  'frame_count': len(names), 'jacobian_condition': float(np.linalg.cond(fit.jac)),
                  'boundary_rms_median_px': float(np.median(np.asarray(edge_rms)[keep])),
                  'objective': 'fixed point-plane + bidirectional silhouette, no color clipping'},
              'calibration_objective': 'fixed point-plane + bidirectional boundary reprojection'}
    return result
