"""Finite-board association, distinct from plane fitting and RT accuracy checks.

Only whole, independently generated source hypotheses are compared. No points
are trimmed with RT, no expected sensor baseline, image position or pose is used.
"""
from hashlib import sha256
import numpy as np
from nsl_geometry import corrected_points


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
    result = dict(state='unverified', plane=None, basis=basis, candidates=[])
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
    for score, _, other, other_evidence in plausible[1:]:
        a, b = np.asarray(plane['mask'], bool), np.asarray(other['mask'], bool)
        overlap = np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b))
        if overlap < .20 and score < evidence['score']+.10 and other_evidence['coverage'] >= .35:
            result.update(state='ambiguous', reason='서로 다른 후보의 대응 근거가 비슷함: 보드 선택 보류')
            return result
    # Recognizing a partial board does not make its normal suitable for fitting RT.
    full = plane.get('support_quality', {}).get('eligible', False) and evidence['coverage'] >= .60
    result.update(state='matched' if full else 'partial', plane=plane, index=index,
                  evidence=evidence, reason='고정 후보의 유한 보드 대응 확인' if full else '보드 부분 대응: 완전한 평면/RT 학습 적합성은 미확인')
    return result
