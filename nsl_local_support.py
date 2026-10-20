"""Supplement missed source objects with metric local plane hypotheses.

Only XYZ and existing SOURCE candidates enter this path. No RGB, RT, frame ID,
fixed image location or missing/checker pattern is used. Primary hypotheses stay
in their original order; supplemental proposals never displace them for RT.
"""
import numpy as np
from scipy.spatial import cKDTree
from nsl_plane_fit import cell_means, fit_plane, support_quality
from nsl_surface_support import plane_conditioned_support


def overlap(a, b):
    return np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b))


def unrepresented_support(grid, planes):
    eligible = [p for p in planes if support_quality(grid[p['mask']], p)['eligible']]
    proposed = []
    for p in planes:
        if support_quality(grid[p['mask']], p)['eligible']:
            continue
        if p.get('edge_support', 0) < .55 or p.get('quad_support', 0) < .70:
            continue
        if min(p['size']) < .10 or max(p['size']) > .6:
            continue
        if any(overlap(p['mask'], a['mask']) > .20 for a in eligible):
            continue
        if any(overlap(p['mask'], a['mask']) > .60 for a in proposed):
            continue
        proposed.append(p)
        if len(proposed) == 8:
            break
    focus = np.logical_or.reduce([p['mask'] for p in proposed]) if proposed else np.zeros(grid.shape[:2], bool)
    return focus, eligible


def local_hypotheses(grid, focus):
    valid = np.isfinite(grid).all(2)
    raw, flat = grid[valid], np.flatnonzero(valid)
    focus = np.asarray(focus, bool) & valid
    if len(raw) < 150 or not focus.any():
        return []
    _, index = np.unique(np.floor(raw/.05).astype('int64'), axis=0, return_index=True)
    centers = raw[index]
    centers = centers[cKDTree(grid[focus]).query(centers)[0] < .05]
    distance, neighbors = cKDTree(raw).query(centers, k=min(160, len(raw)))
    seeds = []
    for radius in (.06, .10, .16):
        for i in range(len(centers)):
            ids = neighbors[i][distance[i] < radius]
            if len(ids) < 100:
                continue
            points = raw[ids]
            center, normal, offset = fit_plane(cell_means(points))
            values = np.linalg.eigvalsh(np.cov(points.T))
            if values[0]/max(values.sum(), 1e-12) > .08:
                continue
            if any(abs(normal@p['normal']) > np.cos(np.deg2rad(4)) and abs(offset-p['h']) < .015
                   and np.linalg.norm(center-p['center']) < .15 for p in seeds):
                continue
            mask = np.zeros(valid.shape, bool)
            mask.ravel()[flat[ids]] = True
            seeds.append(dict(mask=mask, normal=normal, h=offset, center=center))
    return seeds


def local_support(grid, planes, quality_function):
    """Recover only unrepresented objects; all original structural gates apply."""
    focus, eligible = unrepresented_support(grid, planes)
    seeds = local_hypotheses(grid, focus)
    extra = plane_conditioned_support(grid, seeds, quality_function, fast_proposals=True)
    return [dict(p, support_origin='local_metric_planes') for p in extra
            if not any(overlap(p['mask'], a['mask']) > .20 for a in eligible)]
