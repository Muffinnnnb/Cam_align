"""Frozen e413ca3 reduction, used only by lossless optimization regressions."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import numpy as np


def legacy_cell_means(points, bins=8):
    center = points.mean(0)
    axes = np.linalg.svd(points-center, full_matrices=False)[2][:2]
    uv = (points-center) @ axes.T
    lo, hi = np.percentile(uv, [1, 99], axis=0)
    cells = np.clip(((uv-lo)/np.maximum(hi-lo, 1e-9)*bins).astype(int), 0, bins-1)
    key = cells[:, 1]*bins+cells[:, 0]
    return np.array([points[key == k].mean(0) for k in np.unique(key)])


@contextmanager
def legacy_reduction():
    import nsl_color_calibration as nc
    import nsl_plane_fit as fit
    import nsl_local_support as local
    import nsl_metric_support as metric
    import nsl_surface_support as surface
    import nsl_projection_fit as projection
    with ExitStack() as stack:
        for module in (nc, fit, local, metric, surface, projection):
            stack.enter_context(patch.object(module, 'cell_means', legacy_cell_means))
        yield


def assert_exact(a, b, path='result'):
    """Check every field, not just the winning mask or final success count."""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_exact(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            assert_exact(x, y, f'{path}[{index}]')
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype, path
        np.testing.assert_array_equal(a, b, err_msg=path, strict=True)
    else:
        assert a == b or (isinstance(a, (float, np.floating)) and np.isnan(a) and np.isnan(b)), path
