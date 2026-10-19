"""NSL-only dimensional audit. No RT, Color image, camera intrinsics or PnP input."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from nsl_geometry import scale_ray_angles

BOARD_SIZE = np.array([.25, .28])


def measured_size(points):
    centered = points-points.mean(0)
    axes = np.linalg.svd(centered, full_matrices=False)[2][:2]
    local = centered@axes.T
    low, high = np.percentile(local, [1, 99], axis=0)
    core = local[np.all((local >= low) & (local <= high), axis=1)]
    return np.sort(cv2.minAreaRect(core.astype('float32'))[1])


def dimension_records(frames):
    records = []
    for name, frame in sorted(frames.items()):
        p = frame.get('source_plane')
        # Freeze source-only completeness tests BEFORE fitting a correction.
        if (p is None or p.get('quad_support', 0) < .90 or p.get('edge_support', 0) < .90
                or p.get('support_occupancy', 0) < .90 or p.get('rectangularity', 0) < .80
                or min(np.array(p['size'])/BOARD_SIZE) < .85):
            continue
        points = frame['xyz_mm'].reshape(*frame['pcd_shape'], 3)[p['mask']]*.001
        if len(points) >= 100 and np.isfinite(points).all():
            records.append((str(name), points))
    return records


def fit_size_scale(records):
    def residual(x):
        return np.concatenate([(measured_size(scale_ray_angles(points, np.exp(x[0])))-BOARD_SIZE)/.005
                               for _, points in records])
    # minAreaRect uses float32. At x=0, a relative finite-difference step can
    # round away; initialize from measured size and use a resolvable step.
    sizes = np.array([measured_size(points) for _, points in records])
    x0 = float(np.log(np.median(BOARD_SIZE/sizes)))
    if abs(x0) < .001:
        x0 = .001
    fit = least_squares(residual, [x0], method='lm', diff_step=1e-4, max_nfev=150)
    scale = float(np.exp(fit.x[0]))
    if not fit.success or not .7 <= scale <= 1.3:
        raise RuntimeError('NSL 독립 치수 보정 수렴/범위 검사 실패.')
    return scale


def estimate_source_correction(frames):
    """Accept a separate empirical ray correction only after NSL-only holdouts.

    This cannot distinguish all range/lens/board-boundary biases. It preserves
    measured radial ranges and records that it is NOT factory calibration.
    """
    records = dimension_records(frames)
    audit = {'method': 'NSL-only two measured dimensions; no RGB or RT',
             'board_size_m': BOARD_SIZE.tolist(), 'eligible_frames': len(records), 'applied': False}
    if len(records) < 16:
        audit['reason'] = '완전한 NSL 보드 관측 16개 미만; 광선 모델 유지'
        return None, audit
    before = np.array([measured_size(points) for _, points in records])
    audit['before_median_mm'] = (np.median(before, axis=0)*1000).tolist()
    audit['before_mae_mm'] = float(np.mean(abs(before-BOARD_SIZE))*1000)
    # A shared angle correction is unjustified if both dimensions already agree
    # or their errors point in opposite directions. Never fit merely to RGB.
    bias = np.median(before/BOARD_SIZE-1, axis=0)
    if np.any(abs(bias) < .03) or bias[0]*bias[1] <= 0:
        audit['reason'] = '두 변의 반복적인 동방향 치수 오차 없음; 광선 모델 유지'
        return None, audit
    folds, held_before, held_after = [], [], []
    for fold in range(4):
        test = [i for i in range(len(records)) if min(3, i*4//len(records)) == fold]
        train = [r for i, r in enumerate(records) if i not in test]
        scale = fit_size_scale(train)
        old = before[test]
        new = np.array([measured_size(scale_ray_angles(records[i][1], scale)) for i in test])
        old_mae, new_mae = np.mean(abs(old-BOARD_SIZE)), np.mean(abs(new-BOARD_SIZE))
        folds.append({'fold': fold, 'scale': scale, 'count': len(test),
                      'before_mae_mm': float(old_mae*1000), 'after_mae_mm': float(new_mae*1000)})
        held_before.extend(abs(old-BOARD_SIZE).ravel()); held_after.extend(abs(new-BOARD_SIZE).ravel())
    audit['holdouts'] = folds
    audit['heldout_before_mae_mm'] = float(np.mean(held_before)*1000)
    audit['heldout_after_mae_mm'] = float(np.mean(held_after)*1000)
    if (any(r['after_mae_mm'] >= r['before_mae_mm'] for r in folds)
            or np.mean(held_after) >= .85*np.mean(held_before)
            or np.ptp([r['scale'] for r in folds]) > .05):
        audit['reason'] = 'NSL 치수 교차검증 개선/안정성 부족; 광선 모델 유지'
        return None, audit
    scale = fit_size_scale(records)
    after = np.array([measured_size(scale_ray_angles(p, scale)) for _, p in records])
    audit.update(applied=True, scale=scale, after_median_mm=(np.median(after, axis=0)*1000).tolist(),
                 after_mae_mm=float(np.mean(abs(after-BOARD_SIZE))*1000),
                 rows=[{'name': n, 'before_mm': (b*1000).tolist(), 'after_mm': (a*1000).tolist()}
                       for (n, _), b, a in zip(records, before, after)])
    correction = {'model': 'radial_angle_scale', 'scale': scale, 'range_preserved': True,
                  'estimated_from': 'NSL-only known board dimensions, frozen before RT',
                  'warning': 'empirical nominal WF correction; not per-unit factory intrinsics'}
    return correction, audit
