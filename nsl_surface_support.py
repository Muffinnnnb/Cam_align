"""Source-only support proposals; no RGB, RT, fixed position or invented points."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from nsl_plane_fit import cell_means, fit_plane, complete_support


def same_surface(grid, first, second):
    """Require neighboring measured samples and a compact common 3D surface."""
    a, b = np.asarray(first['mask'], bool), np.asarray(second['mask'], bool)
    if not a.any() or not b.any():
        return False
    ay, ax = np.where(a); by, bx = np.where(b)
    if (max(ax.min(), bx.min())-min(ax.max(), bx.max()) > 3
            or max(ay.min(), by.min())-min(ay.max(), by.max()) > 3):
        return False
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
    axes = np.linalg.svd(points-center, full_matrices=False)[2][:2]
    size = np.sort(cv2.minAreaRect(((points-center)@axes.T).astype('float32'))[1])
    return bool(np.all(size <= np.array([.25, .28])*1.40))


def range_support(grid, quality_function):
    """Multiscale depth-discontinuity regions supplement narrow RANSAC seeds.

    Smoothing only proposes spatial support. Plane fits and output coordinates
    always use original samples; normals never come from fitted image lines.
    """
    valid = np.isfinite(grid).all(2)
    if valid.sum() < 150:
        return []
    distance = np.linalg.norm(grid, axis=2)
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = distance[tuple(nearest)].astype('float32')
    seeds = []
    for window in (5, 9):
        smooth = median_filter(filled, size=window)
        gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0),
                            cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        for threshold in (.08, .12, .18):
            region = ((gradient < threshold) & valid).astype('uint8')
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((3, 3), 'uint8'))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(region)
            for label in range(1, count):
                if not 150 <= stats[label, cv2.CC_STAT_AREA] <= 12000:
                    continue
                mask = labels == label
                points = grid[mask]
                _, normal, h = fit_plane(cell_means(points))
                quality = quality_function(points, normal)
                if quality['rmse'] > .030 or max(quality['size']) > .6 or min(quality['size']) < .12:
                    continue
                if any(np.count_nonzero(mask & s['mask'])/np.count_nonzero(mask | s['mask']) > .85 for s in seeds):
                    continue
                seeds.append(dict(mask=mask, normal=normal, h=h, **quality))
    return complete_support(grid, seeds, quality_function)


def paired_support(grid, planes, quality_function):
    """Refit independent pairs; never recursively union a chain through a hand."""
    seeds = []
    for i, a in enumerate(planes):
        for b in planes[i+1:]:
            mask = a['mask'] | b['mask']
            if np.count_nonzero(a['mask'] & b['mask'])/max(1, mask.sum()) > .65:
                continue
            if same_surface(grid, a, b):
                if any(np.count_nonzero(mask & s['mask'])/np.count_nonzero(mask | s['mask']) > .90 for s in seeds):
                    continue
                seeds.append(dict(mask=mask, normal=a['normal']))
    # complete_support reads raw points again and applies all existing structural
    # gates, including coverage, size, flatness, normal stability and boundaries.
    return complete_support(grid, seeds, quality_function)
