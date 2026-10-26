"""Metric fallback must not specialize to one camera or alter measured samples."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_surface_support import metric_support
from nsl_plane_fit import support_quality


def synthetic_board():
    x, y = np.meshgrid(np.linspace(-.3, .3, 100), np.linspace(-.3, .3, 100))
    board = (abs(x) < .125) & (abs(y) < .14)
    rng = np.random.default_rng(61)
    z = np.where(board, .8, 1.3)+rng.normal(0, .004, x.shape)
    grid = np.stack((x, y, z), -1)
    grid[rng.random(x.shape) < .2] = np.nan
    return grid, board & np.isfinite(grid).all(2)


class MetricSupportTests(unittest.TestCase):
    def test_empty_tiny_and_duplicate_clouds_do_not_generate_planes(self):
        for grid in (np.full((30, 30, 3), np.nan), np.zeros((2, 2, 3)),
                     np.ones((30, 30, 3))):
            self.assertEqual(metric_support(grid, nc.plane_quality), [])

    def test_original_xyz_and_missing_points_are_preserved(self):
        grid, board = synthetic_board(); original = grid.copy()
        planes = metric_support(grid, nc.plane_quality)
        self.assertTrue(planes)
        p = max(planes, key=lambda p: np.count_nonzero(p['mask'] & board))
        self.assertGreater(np.count_nonzero(p['mask'] & board)/board.sum(), .80)
        self.assertFalse(np.any(p['mask'] & ~board))
        self.assertTrue(support_quality(grid[p['mask']], p)['eligible'])
        np.testing.assert_array_equal(grid, original)

    def test_pose_changes_do_not_require_sensor_baseline_or_rgb(self):
        grid, board = synthetic_board()
        for pose in ((.2, -.4, .1), (-.3, .6, -.5)):
            rotation = cv2.Rodrigues(np.array(pose))[0]
            moved = grid@rotation.T+[.1, -.2, .5]
            planes = metric_support(moved, nc.plane_quality)
            self.assertTrue(planes)
            self.assertGreater(max(np.count_nonzero(p['mask'] & board)/board.sum() for p in planes), .80)
            self.assertTrue(all(np.isfinite(moved[p['mask']]).all() for p in planes))

    def test_successful_image_grid_hypotheses_do_not_call_fallback(self):
        grid, _ = synthetic_board()
        with patch.object(nc, 'metric_support', side_effect=AssertionError('unneeded fallback')):
            planes = nc.nsl_plane_candidates(grid.reshape(-1, 3)*1000, grid.shape[:2])
        self.assertTrue(any(support_quality(grid[p['mask']], p)['eligible'] for p in planes))


if __name__ == '__main__':
    unittest.main()
