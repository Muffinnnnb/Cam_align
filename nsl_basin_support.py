"""Depth-gradient basin proposals; no RGB, RT, manual ROI or synthetic points."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter, watershed_ift
from nsl_plane_fit import complete_support, support_quality
from nsl_local_support import overlap


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
