"""Local proposals must be source-only, finite and supplemental."""
import unittest
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_local_support import local_hypotheses, local_support, unrepresented_support
from nsl_plane_fit import complete_support, support_quality
from nsl_surface_support import plane_conditioned_support
import test_nsl_surface_support as fixtures


class LocalSupportTests(unittest.TestCase):
    def test_local_hypotheses_follow_3d_geometry_not_image_location(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        for pose in ((.1, .5, -.3), (-.3, -.6, .4)):
            rotation = cv2.Rodrigues(np.array(pose))[0]
            points = grid@rotation.T+[.2, -.1, .5]
            original = points.copy()
            seeds = local_hypotheses(points, board)
            self.assertTrue(seeds)
            self.assertTrue(any(abs(p['normal']@rotation[:, 2]) > .98 for p in seeds))
            self.assertTrue(all(np.isfinite(points[p['mask']]).all() for p in seeds))
            np.testing.assert_array_equal(points, original)

    def test_empty_focus_and_invalid_grid_do_not_create_points(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        self.assertEqual(local_hypotheses(grid, np.zeros_like(board)), [])
        self.assertEqual(local_hypotheses(np.full_like(grid, np.nan), board), [])
        grid[45:65, 45:65] = np.nan
        for seed in local_hypotheses(grid, board):
            self.assertFalse(seed['mask'][45:65, 45:65].any())

    def test_existing_complete_object_is_not_replaced(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        planes = complete_support(grid, [dict(mask=board)], nc.plane_quality)
        self.assertTrue(support_quality(grid[planes[0]['mask']], planes[0])['eligible'])
        old = [p['mask'].copy() for p in planes]
        focus, eligible = unrepresented_support(grid, planes)
        self.assertFalse(focus.any())
        self.assertTrue(eligible)
        self.assertEqual(local_support(grid, planes, nc.plane_quality), [])
        for before, after in zip(old, planes):
            np.testing.assert_array_equal(before, after['mask'])

    def test_fast_residual_proposals_keep_original_coordinates(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        grid[50:60, 50:60] = np.nan
        original = grid.copy()
        seeds = local_hypotheses(grid, board)
        planes = plane_conditioned_support(grid, seeds, nc.plane_quality, fast_proposals=True)
        self.assertTrue(planes)
        self.assertTrue(all(support_quality(grid[p['mask']], p)['eligible'] for p in planes))
        self.assertTrue(all(not p['mask'][50:60, 50:60].any() for p in planes))
        np.testing.assert_array_equal(grid, original)


if __name__ == '__main__':
    unittest.main()
