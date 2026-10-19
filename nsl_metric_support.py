"""Fallback surface proposals from metric XYZ neighborhoods, without RGB or RT."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from nsl_plane_fit import cell_means, fit_plane, complete_support, support_quality


def metric_support(grid, quality_function):
    """Find connected local planes when image-grid proposals have no usable board.

    10 mm voxels and 40/70/100 mm neighborhoods are proposal scales, NOT accuracy
    gates. Voxel centroids never become measured points. Final support is refit
    from original XYZ and must pass the unchanged structural eligibility tests.
    No timestamp, image location, reflectance, missing-pixel pattern, or sensor
    baseline is an input. The input is an organized XYZ grid in metres.
    """
    valid = np.isfinite(grid).all(2)
    points = grid[valid]
    if len(points) < 150:
        return []
    _, inverse = np.unique(np.floor(points/.010).astype('int64'), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centers = np.column_stack([np.bincount(inverse, weights=points[:, axis])/counts
                               for axis in range(3)])
    if len(centers) < 10:
        return []
    k = min(64, len(centers))
    distance, nearest = cKDTree(centers).query(centers, k=k)
    starts = np.repeat(np.arange(len(centers)), min(16, k-1))
    stops = nearest[:, 1:min(17, k)].ravel()
    nearby = distance[:, 1:min(17, k)].ravel() < .035
    starts, stops = starts[nearby], stops[nearby]
    delta = centers[stops]-centers[starts]
    seeds = []
    for radius in (.04, .07, .10):
        # Chunk covariance work so neighborhood tensors do not scale with N.
        eigenvalues, normals = [], []
        for begin in range(0, len(centers), 2048):
            end = begin+2048
            selected = distance[begin:end] < radius
            count = selected.sum(1)
            neighbors = centers[nearest[begin:end]]
            mean = np.sum(neighbors*selected[..., None], axis=1)/count[:, None]
            offsets = (neighbors-mean[:, None])*selected[..., None]
            cov = np.einsum('nki,nkj->nij', offsets, offsets)/np.maximum(count-1, 1)[:, None, None]
            values, axes = np.linalg.eigh(cov)
            eigenvalues.append(values); normals.append(axes[:, :, 0])
        values, normal = np.vstack(eigenvalues), np.vstack(normals)
        curvature = values[:, 0]/np.maximum(values.sum(1), 1e-12)
        usable = ((distance < radius).sum(1) >= 10) & (curvature < .08)
        links = usable[starts] & usable[stops]
        links &= abs(np.sum(normal[starts]*normal[stops], axis=1)) > np.cos(np.deg2rad(20))
        noise = np.sqrt(np.maximum(0, values[:, 0]))
        tolerance = .010+np.minimum(noise[starts]+noise[stops], .020)
        for endpoint in (starts, stops):
            links &= abs(np.sum(delta*normal[endpoint], axis=1)) < tolerance
        graph = coo_matrix((np.ones(links.sum()), (starts[links], stops[links])),
                           shape=(len(centers), len(centers)))
        _, labels = connected_components(graph, directed=False)
        sizes = np.bincount(labels, weights=counts)
        for label in np.flatnonzero((sizes >= 150) & (sizes <= 12000)):
            mask = np.zeros(valid.shape, bool)
            mask[valid] = labels[inverse] == label
            raw = grid[mask]
            _, n, _ = fit_plane(cell_means(raw))
            quality = quality_function(raw, n)
            if quality['rmse'] > .030 or max(quality['size']) > .6 or min(quality['size']) < .12:
                continue
            if any(np.count_nonzero(mask & p['mask'])/np.count_nonzero(mask | p['mask']) > .85 for p in seeds):
                continue
            seeds.append(dict(mask=mask, normal=n))
    result = complete_support(grid, seeds, quality_function)
    return [dict(p, support_origin='metric_neighborhoods') for p in result
            if support_quality(grid[p['mask']], p)['eligible']]
