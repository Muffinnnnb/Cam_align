"""Explicit, reversible range-to-XYZ recovery; never edits captured PCD files."""
from hashlib import sha256
from pathlib import Path

import numpy as np

TABLES = Path(__file__).resolve().parent / 'cal' / 'nsl'
GEOMETRY_MODES = ('pcd', 'wf_range')
GEOMETRY_LABELS = ('PCD 원본 좌표 (기존 RT)', '원본 거리맵 → 제조사 WF 복원 (별도 RT)')


def geometry_spec(mode='pcd'):
    if mode not in GEOMETRY_MODES:
        raise ValueError(f'지원하지 않는 NSL 점군 모델: {mode}')
    if mode == 'pcd':
        return {'mode': mode}
    path = TABLES / 'lens_wf_nano.ini'
    return {'mode': mode, 'lens_table': path.name,
            'table_sha256': sha256(path.read_bytes()).hexdigest(),
            'pixel_pitch_mm': .020, 'width': 320, 'height': 240,
            'center_pixels': [159.5, 119.5], 'distance_type': 'radial_range_mm'}


def rt_geometry(calibration):
    spec = calibration.get('source_geometry', {'mode': 'pcd'})
    expected = geometry_spec(spec.get('mode', 'pcd'))
    if spec != expected:
        raise ValueError('RT의 NSL 거리→XYZ 설정/변환표가 현재 모델과 다릅니다. 해당 모델로 재계산하세요.')
    ray_scale(calibration)
    return expected['mode']


def ray_scale(calibration):
    """An explicit RT-bound estimated ray model, never a change to captured data."""
    correction = calibration.get('source_ray_correction')
    if correction is None:
        return 1.
    if (calibration.get('source_geometry', {}).get('mode') != 'wf_range'
            or correction.get('model') != 'radial_angle_scale'):
        raise ValueError('광선 각도 보정은 명시된 WF 복원 모델에만 적용할 수 있습니다.')
    scale = float(correction['scale'])
    if not np.isfinite(scale) or not .7 <= scale <= 1.3:
        raise ValueError('추정 광선 각도 계수가 유효 범위 밖입니다.')
    return scale


def scale_ray_angles(points, scale=1.):
    """Scale polar angles, preserving radial range, azimuth, invalid points and input."""
    points = np.asarray(points, dtype=np.float64)
    if scale == 1.:
        return points
    rho = np.linalg.norm(points[..., :2], axis=-1)
    distance = np.linalg.norm(points, axis=-1)
    theta = np.arctan2(rho, points[..., 2]) * scale
    factor = np.divide(np.sin(theta), rho, out=np.zeros_like(rho), where=rho > 0)
    return np.concatenate((points[..., :2]*factor[..., None], np.cos(theta)[..., None]), axis=-1)*distance[..., None]


def corrected_points(points, calibration):
    return scale_ray_angles(points, ray_scale(calibration))


def check_geometry(frame, calibration):
    mode = frame.get('source_geometry', {'mode': 'pcd'}).get('mode', 'pcd')
    if mode != rt_geometry(calibration):
        raise ValueError('PCD 원본 RT와 WF 복원 RT는 혼용할 수 없습니다. 점군 모델에 맞는 RT를 선택하세요.')


def detected_lens(xyz_mm, shape):
    if shape == (240, 320):
        valid = np.isfinite(xyz_mm).all(1) & (xyz_mm[:, 2] > 0)
        if valid.sum() >= 100:
            nf = range_to_xyz(np.full(shape, 1000, np.uint16), 'nf')[valid] / 1000
            unit = xyz_mm[valid] / np.linalg.norm(xyz_mm[valid], axis=1)[:, None]
            if np.median(np.linalg.norm(unit - nf, axis=1)) < 1e-6:
                return 'nf'
    return None


def fusion_block_reason(frame):
    if (frame.get('source_geometry', {}).get('mode') == 'pcd'
            and frame.get('source_lens') == 'nf' and not frame.get('allow_nf_comparison', False)):
        return 'NF 원본 융합 차단: 현재 광각 장비와 맞지 않습니다. WF 복원을 적용하거나 NF 비교를 명시적으로 허용하세요.'
    return ''


def source_geometry_warning(xyz_mm, shape, mode, lens=None):
    if mode == 'wf_range':
        return 'WF 거리맵 복원 모델: 원본 PCD 보존 · 개별 장비 실측 내부 보정값 아님 · 전용 RT 필요'
    if (lens or detected_lens(xyz_mm, shape)) == 'nf':
        return 'NF 원본 좌표: 현재 광각 장비와 불일치합니다. 아래 버튼으로 WF 복원과 전용 RT를 함께 적용하세요.'
    return 'PCD 원본 XYZ 사용 · 목록과 화면은 동일한 NSL 평면 검사 · 초록 점은 실제 피팅 인라이어'


def range_to_xyz(distance_mm, lens='wf'):
    """Manufacturer radial-angle table, with the pitch verified against saved NF XYZ.

    WF is an explicit recovery model, not a per-unit factory calibration.
    Radial distance is preserved; this is NOT scaling previously computed XYZ.
    """
    if distance_mm.shape != (240, 320) or distance_mm.dtype != np.uint16:
        raise ValueError('WF 복원에는 320×240 uint16 원본 거리맵(mm)이 필요합니다.')
    if lens not in ('nf', 'wf'):
        raise ValueError(f'지원하지 않는 렌즈: {lens}')
    lines = (TABLES / f'lens_{lens}_nano.ini').read_text().splitlines()
    a, b = lines.index('angle'), lines.index('radius')
    angles = np.array([float(x) for x in lines[a+1:b] if x.strip()])
    radius = np.array([float(x) for x in lines[b+1:] if x.strip()])
    y, x = np.mgrid[:240, :320]
    x, y = x - 159.5, y - 119.5
    rho = np.hypot(x, y)
    if rho.max() * .020 > radius[-1] or np.any(np.diff(radius) <= 0):
        raise ValueError('렌즈 변환표 범위/순서가 올바르지 않습니다.')
    theta = np.deg2rad(np.interp(rho * .020, radius, angles))
    factor = np.sin(theta) / rho
    xyz = np.stack((x * factor, y * factor, np.cos(theta)), axis=-1) * distance_mm[..., None]
    xyz[(distance_mm == 0) | (distance_mm >= 64000)] = np.nan
    return xyz.reshape(-1, 3)
