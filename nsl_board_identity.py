"""Finite-board association, distinct from plane fitting and RT accuracy checks.

Only whole, independently generated source hypotheses are compared. No points
are trimmed with RT, no expected sensor baseline, image position or pose is used.
"""
from hashlib import sha256
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from nsl_geometry import corrected_points
from nsl_plane_fit import cell_means, fit_plane

IDENTITY_VERSION = 2


def same_surface(grid, first, second):
    """Source-only evidence that two masks describe pieces of one finite surface.

    No union is exported or fitted to RT. Require actual neighboring samples,
    continuous 3D depth and a compact common plane, not merely overlapping boxes.
    30 mm is the existing source support gate, not an RT precision guarantee.
    """
    a, b = np.asarray(first['mask'], bool), np.asarray(second['mask'], bool)
    minimum_links = max(12, int(.02*min(a.sum(), b.sum())))
    for left, right in ((a, b), (b, a)):
        distance, nearest = distance_transform_edt(~left, return_indices=True)
        boundary = right & ~left & (distance <= 3)
        if boundary.sum() < minimum_links:
            return False
        gaps = np.linalg.norm(grid[boundary]-grid[tuple(nearest[:, boundary])], axis=1)
        if not np.isfinite(gaps).all() or np.median(gaps) > .030:
            return False
    points = grid[a | b]
    if not np.isfinite(points).all():
        return False
    center, normal, offset = fit_plane(cell_means(points))
    error = abs(points@normal-offset)
    if np.sqrt(np.mean(error**2)) > .020 or np.percentile(error, 90) > .030:
        return False
    # Extents are rotation invariant and cannot grow across an entire body.
    axes = np.linalg.svd(points-center, full_matrices=False)[2][:2]
    size = np.sort(cv2.minAreaRect(((points-center)@axes.T).astype('float32'))[1])
    return bool(np.all(size <= np.array([.25, .28])*1.40))


def frame_fingerprint(frame):
    target = frame.get('target_plane') or {}
    digest = sha256(np.asarray(frame['xyz_mm'], dtype='<f8').tobytes())
    digest.update(np.asarray(target.get('corners3d', []), dtype='<f8').tobytes())
    return digest.hexdigest()


def association_model(frame, calibration, K, distortion):
    """Prefer a model trained without this time block; never reuse it by name alone."""
    if calibration is None:
        return None, 'RT 없음: 기하 후보만 있으며 보드 대응은 미확인'
    if calibration.get('target_image_size') and list(frame['color'].shape[1::-1]) != calibration['target_image_size']:
        return None, '카메라 해상도 변경: 내부 파라미터 확인 후 재보정 필요'
    for key, actual in (('target_intrinsics', K), ('target_distortion', distortion)):
        if key in calibration:
            saved = np.asarray(calibration[key])
            if saved.shape != np.asarray(actual).shape or not np.allclose(saved, actual, rtol=0, atol=1e-8):
                return None, '카메라 내부 파라미터 변경: 해당 카메라로 재보정 필요'
    for model in calibration.get('board_identity_models', []):
        expected = model.get('heldout_fingerprints', {}).get(frame['timestamp'])
        if expected and expected == frame_fingerprint(frame):
            if frame['timestamp'] in model.get('training_frames', []):
                return None, '교차검증 학습/검증 중복: 대응 판정 중단'
            return model, '이 촬영 구간을 제외하고 추정한 RT로 대응 검사'
    return calibration, '현재 RT 기반 대응 추정 (독립 정확도 검증 아님)'


def candidate_evidence(points, target, model):
    corners = np.asarray(target['corners3d'], float)
    edges = np.array([corners[1]-corners[0], corners[3]-corners[0]])
    size = np.linalg.norm(edges, axis=1)
    if np.min(size) <= 1e-6:
        raise ValueError('유효한 실제 보드 크기가 필요합니다.')
    diagonal = float(np.linalg.norm(size))
    axes = edges/size[:, None]
    normal = np.cross(axes[0], axes[1])
    normal /= np.linalg.norm(normal)
    transformed = corrected_points(points, model)@np.asarray(model['R']).T+model['t']
    delta = transformed-corners.mean(0)
    xy = delta@axes.T
    plane_error = abs(delta@normal)
    overflow = np.maximum(abs(xy)-size/2, 0)
    distance = np.sqrt(plane_error**2+np.sum(overflow**2, axis=1))
    inside = np.all(abs(xy) <= size/2, axis=1)
    # Coarse identity, NOT millimetre accuracy. Normalize by physical board size,
    # not focal length, resolution, camera baseline, or absolute sensor depth.
    near = distance <= .15*diagonal
    cells = np.floor((xy/size+.5)*8).astype(int)
    cells = cells[inside & near]
    coverage = len(np.unique(np.clip(cells, 0, 7), axis=0))/64
    rms = float(np.sqrt(np.mean(distance**2)))
    near_fraction = float(near.mean())
    overlap = float(inside.mean())
    plausible = (np.all(transformed[:, 2] > 0) and near_fraction >= .80
                 and overlap >= .65 and coverage >= .20)
    score = rms/diagonal+.30*(1-coverage)+.25*(1-overlap)
    return dict(plausible=bool(plausible), score=float(score), coverage=float(coverage),
                overlap=overlap, near_fraction=near_fraction, finite_board_rms_m=rms,
                identity_tolerance_m=.15*diagonal)


def identify_board(frame, calibration, K, distortion):
    """Return a selected fixed hypothesis or explicit uncertainty, never a substitute."""
    model, basis = association_model(frame, calibration, K, distortion)
    result = dict(state='unverified', plane=None, basis=basis, candidates=[], version=IDENTITY_VERSION)
    target = frame.get('target_plane')
    if model is None or target is None or 'corners3d' not in target:
        return result
    grid = np.asarray(frame['xyz_mm']).reshape(*frame['pcd_shape'], 3)*.001
    plausible = []
    for index, plane in enumerate(frame.get('source_candidates', [])):
        points = grid[np.asarray(plane['mask'], bool)]
        if len(points) < 100 or not np.isfinite(points).all():
            continue
        evidence = candidate_evidence(points, target, model)
        result['candidates'].append(dict(index=index, **evidence))
        if evidence['plausible']:
            plausible.append((evidence['score'], index, plane, evidence))
    if not plausible:
        result.update(state='unconfirmed', reason='보드와 대응하는 후보 없음: RT/촬영 동기/가림 또는 검출 누락 확인')
        return result
    plausible.sort(key=lambda row: row[0])
    _, index, plane, evidence = plausible[0]
    compatible = []
    for score, other_index, other, other_evidence in plausible[1:]:
        a, b = np.asarray(plane['mask'], bool), np.asarray(other['mask'], bool)
        overlap = np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b))
        if overlap < .20 and score < evidence['score']+.10 and other_evidence['coverage'] >= .35:
            # Complementary noisy patches need not share pixels. Check physical
            # continuity against EVERY accepted patch; never transitive merging.
            if all(same_surface(grid, member, other) for member in [plane]+[p for _, p in compatible]):
                compatible.append((other_index, other))
            else:
                result.update(state='ambiguous', reason='분리된 표면 후보의 대응 근거가 비슷함: 보드 선택 보류')
                return result
    # Recognizing a partial board does not make its normal suitable for fitting RT.
    full = plane.get('support_quality', {}).get('eligible', False) and evidence['coverage'] >= .60
    result.update(state='matched' if full else 'partial', plane=plane, index=index,
                  evidence=evidence, reason='고정 후보의 유한 보드 대응 확인' if full else '보드 부분 대응: 완전한 평면/RT 학습 적합성은 미확인')
    if compatible:
        result['compatible_fragments'] = [i for i, _ in compatible]
        result['reason'] += ' / 원본 점군 연속성으로 같은 표면의 조각 확인 (점 합치기 없음)'
    return result
