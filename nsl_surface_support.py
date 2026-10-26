"""Source-only support proposals; no RGB, RT, fixed position or invented points."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter, watershed_ift
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from nsl_plane_fit import cell_means, fit_plane, complete_support, support_quality
from calibrate_d_to_r import BOARD_M


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
    return bool(np.all(size <= np.array(BOARD_M)*1.40))


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


def plane_conditioned_support(grid, planes, quality_function, *, fast_proposals=False):
    """Grow source hypotheses in plane-residual space, not raw range space.

    A tilted plane has a range gradient even when perfectly flat. Subtracting
    its source-only plane model removes that slope before region proposals.
    Nearest filling/median filtering affect proposals ONLY; final fitting reads
    original finite XYZ and retains all existing structural eligibility gates.
    """
    valid = np.isfinite(grid).all(2)
    if valid.sum() < 150:
        return []
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    seeds, models, packed = [], [], []
    for plane in planes:
        normal, offset = plane['normal'], plane['h']
        if any(abs(normal@n) > np.cos(np.deg2rad(3)) and abs(offset-h) < .010 for n, h in models):
            continue
        models.append((normal, offset))
        residual = (grid@normal-offset)[tuple(nearest)].astype('float32')
        for window in (5, 9):
            # OpenCV's float median supports kernels <=5; SciPy preserves float
            # residuals for the larger window without quantizing measured data.
            if window <= 5:
                smooth = cv2.medianBlur(residual, window)
            elif fast_proposals:
                # Millimetre quantization speeds PROPOSALS only; fits/exports
                # still read the original unquantized XYZ below.
                quantized = np.clip(np.rint(residual*1000)+128, 0, 255).astype('uint8')
                smooth = (cv2.medianBlur(quantized, window).astype('float32')-128)*.001
            else:
                smooth = median_filter(residual, size=window)
            gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
            for boundary in (None, .12):
                region = abs(smooth) < .030
                if boundary is not None:
                    region &= gradient < boundary
                region = cv2.morphologyEx(region.astype('uint8'), cv2.MORPH_OPEN, np.ones((3, 3), 'uint8'))
                _, labels, stats, _ = cv2.connectedComponentsWithStats(region)
                overlap = np.bincount(labels[plane['mask']])
                for label in np.flatnonzero(overlap >= max(100, .30*plane['mask'].sum())):
                    if label == 0 or not 150 <= stats[label, cv2.CC_STAT_AREA] <= 12000:
                        continue
                    mask = (labels == label) & valid
                    # Exact bit-set IoU avoids repeatedly scanning full grids.
                    bits = int.from_bytes(np.packbits(mask).tobytes(), 'little')
                    if mask.sum() < 100 or any((bits & b).bit_count()/(bits | b).bit_count() > .90 for b in packed):
                        continue
                    raw = grid[mask]
                    _, n, _ = fit_plane(cell_means(raw))
                    quality = quality_function(raw, n)
                    if max(quality['size']) <= .6 and min(quality['size']) >= .12:
                        seeds.append(dict(mask=mask, normal=n))
                        packed.append(bits)
    result = complete_support(grid, seeds, quality_function)
    return [dict(p, support_origin='plane_conditioned') for p in result
            if support_quality(grid[p['mask']], p)['eligible']]


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


def bounded_surface(grid, plane):
    """Reject an arbitrary patch of a plane that continues into its surroundings.

    Perspective alone creates a range gradient on an unbounded flat wall. At
    least 55% of measured surrounding samples must disagree with this source
    plane at the existing 30 mm source-outlier scale. A small inner gap avoids
    mixed edge pixels and the final fit's eroded rim. Missing samples are not
    boundary evidence; this supplements, never replaces, shape gates.
    """
    y, x = np.where(plane['mask'])
    footprint = np.zeros(grid.shape[:2], 'uint8')
    cv2.fillConvexPoly(footprint, cv2.convexHull(np.c_[x, y].astype('int32')), 1)
    gap = max(2, int(np.sqrt(footprint.sum())*.05))
    inner = cv2.dilate(footprint, np.ones((2*gap+1, 2*gap+1), 'uint8'))
    outer = cv2.dilate(footprint, np.ones((4*gap+1, 4*gap+1), 'uint8'))
    ring = (outer > 0) & (inner == 0)
    points = grid[ring & np.isfinite(grid).all(2)]
    return len(points) >= 20 and np.mean(abs(points@plane['normal']-plane['h']) > .030) >= .55


def basin_support(grid, planes, quality_function):
    """Expand incomplete footprints at measured range boundaries, then refit XYZ.

    Noisy points can fragment a plane-residual region. Watershed instead grows
    from an independent source footprint towards range-gradient barriers. Its
    background marker is only a search limit: proposals substantially touching
    that artificial fence are rejected. Missing samples guide proposals only;
    they never become fitted/exported points or count towards support coverage.
    """
    valid = np.isfinite(grid).all(2)
    if valid.sum() < 150:
        return []
    eligibility = [support_quality(grid[p['mask']], p)['eligible'] for p in planes]
    eligible = [p for p, good in zip(planes, eligibility) if good]
    candidates = [p for p, good in zip(planes, eligibility) if not good
                  and p.get('edge_support', 0) >= .55 and p.get('quad_support', 0) >= .70
                  and not any(overlap(p['mask'], a['mask']) > .20 for a in eligible)]
    if not candidates:
        return []
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    distance = np.linalg.norm(grid, axis=2)[tuple(nearest)].astype('float32')
    seeds, kernel = [], np.ones((3, 3), 'uint8')
    for window in (5, 9):
        smooth = median_filter(distance, size=window)
        gradient = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
        # Preserve gradient ordering. A fixed saturated 8-bit cost can turn a
        # smoothly sloped wide-FOV wall into arbitrary flat-cost basins.
        cost = np.rint(gradient/max(float(gradient.max()), 1e-6)*65535).astype('uint16')
        for plane in candidates:
            y, x = np.where(plane['mask'])
            if len(x) < 150:
                continue
            spatial = np.zeros(valid.shape, 'uint8')
            cv2.fillConvexPoly(spatial, cv2.convexHull(np.c_[x, y].astype('int32')), 1)
            interior = distance_transform_edt(spatial)
            cy, cx = np.unravel_index(np.argmax(interior), interior.shape)
            for expansion in (.25, .5, 1.):
                gap = max(3, int(np.sqrt(spatial.sum())*expansion))
                outside = cv2.dilate(spatial, np.ones((2*gap+1, 2*gap+1), 'uint8')) == 0
                markers = np.zeros(valid.shape, 'int32')
                markers[outside] = 2
                markers[max(0, cy-1):cy+2, max(0, cx-1):cx+2] = 1
                basin = (watershed_ift(cost, markers) == 1).astype('uint8')
                if basin[[0, -1], :].any() or basin[:, [0, -1]].any():
                    continue  # A fully supported outline cannot end outside the image.
                measured = (basin > 0) & valid
                if not 150 <= measured.sum() <= 12000:
                    continue
                if any(overlap(measured, s['mask']) > .85 for s in seeds):
                    continue
                # The image boundary is another artificial fence, not evidence
                # of a physical board edge. OpenCV's default erosion border
                # otherwise hides this contact and admits cropped wall patches.
                perimeter = basin-cv2.erode(basin, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                fence = cv2.dilate(outside.astype('uint8'), kernel) > 0
                fence[[0, -1], :] = True
                fence[:, [0, -1]] = True
                contact = np.count_nonzero((perimeter > 0) & fence)/max(1, np.count_nonzero(perimeter))
                if contact > .05:
                    continue
                seeds.append(dict(mask=measured))
    # The hull/gradient is NOT a fitted board pose. Normal, offset, size and
    # eligibility still come from original XYZ via the unchanged fitting path.
    # An expanded footprint can identify a board without giving an unbiased
    # calibration normal. These supplemental proposals must not feed back into
    # RT or source-dimension learning; that path retains its original supports.
    return [dict(p, support_origin='range_gradient_basin', calibration_eligible=False)
            for p in complete_support(grid, seeds, quality_function)
            if support_quality(grid[p['mask']], p)['eligible']
            and bounded_surface(grid, p)
            and not any(overlap(p['mask'], a['mask']) > .20 for a in eligible)]
