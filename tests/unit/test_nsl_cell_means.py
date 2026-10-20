"""Arithmetic/layout parity with the pre-optimization implementation."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cold_analysis_reference import legacy_cell_means
from nsl_plane_fit import cell_means


class CellMeansTests(unittest.TestCase):
    def check(self, points, bins):
        before = points.copy()
        np.testing.assert_array_equal(cell_means(points, bins), legacy_cell_means(points, bins), strict=True)
        np.testing.assert_array_equal(points, before)

    def test_dtypes_sizes_bins_and_layouts(self):
        rng = np.random.default_rng(51021)
        for dtype in (np.float64, np.float32, np.int32):
            for count in (3, 5, 30, 100, 1000, 12000):
                points = (rng.normal(size=(count, 3))*100).astype(dtype)
                for layout in (points, np.asfortranarray(points), points[::-1], points[::2]):
                    for bins in (1, 2, 8, 16):
                        with self.subTest(dtype=dtype, count=len(layout), bins=bins, strides=layout.strides):
                            self.check(layout, bins)

    def test_planar_degenerate_outliers_and_uneven_density(self):
        rng = np.random.default_rng(923)
        points = rng.normal(size=(1700, 3))*[.1, .12, .003]+[.2, -.1, .8]
        points[:1000] = points[0]  # Occupied cells must still receive equal weight.
        for data in (points, points*1e6, points+1e9, np.ones((100, 3)), points[:, :2], np.c_[points, points[:, 0]]):
            for bins in (1, 8, 16):
                with self.subTest(shape=data.shape, bins=bins):
                    self.check(data, bins)


if __name__ == '__main__':
    unittest.main()
