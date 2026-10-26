"""Range basins must add real support without ROI, RT or invented boundaries."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_surface_support import basin_support, bounded_surface
from nsl_plane_fit import complete_support, support_quality
import test_nsl_surface_support as fixtures


class BasinSupportTests(unittest.TestCase):
    @staticmethod
    def partial(grid, board):
        seed = board & (np.indices(board.shape)[1] > board.shape[1]//2)
        planes = complete_support(grid, [dict(mask=seed)], nc.plane_quality)
        return planes

    def test_grows_partial_source_footprints_under_rigid_pose_changes(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        for pose in ((0., 0., 0.), (.2, .35, -.4), (-.1, -.45, .3)):
            moved = grid@cv2.Rodrigues(np.array(pose))[0].T+[.1, -.1, .3]
            original = moved.copy()
            planes = self.partial(moved, board)
            self.assertTrue(planes)
            self.assertFalse(support_quality(moved[planes[0]['mask']], planes[0])['eligible'])
            # Source boundary evidence locates the incomplete object; no image
            # position or target-camera information enters the expansion.
            extra = basin_support(moved, planes, nc.plane_quality)
            self.assertTrue(extra)
            best = max(extra, key=lambda p: np.count_nonzero(p['mask'] & board))
            self.assertGreater(np.count_nonzero(best['mask'] & board)/board.sum(), .8)
            self.assertFalse(np.any(best['mask'] & ~board))
            self.assertFalse(best['calibration_eligible'])
            np.testing.assert_array_equal(moved, original)

    def test_missing_samples_never_become_measured_support(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        grid[50:65, 50:65] = np.nan
        planes = self.partial(grid, board & np.isfinite(grid).all(2))
        extra = basin_support(grid, planes, nc.plane_quality)
        self.assertTrue(extra)
        for p in extra:
            self.assertFalse(p['mask'][50:65, 50:65].any())
            self.assertTrue(np.isfinite(grid[p['mask']]).all())

    def test_complete_objects_and_empty_data_do_not_need_expansion(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        planes = complete_support(grid, [dict(mask=board)], nc.plane_quality)
        with patch('nsl_surface_support.watershed_ift') as watershed:
            self.assertEqual(basin_support(grid, planes, nc.plane_quality), [])
            self.assertEqual(basin_support(np.full_like(grid, np.nan), [], nc.plane_quality), [])
            watershed.assert_not_called()

    def test_search_fence_cannot_manufacture_a_board_boundary(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        planes = self.partial(grid, board)
        # A basin reaching its artificial outer search fence is NOT a measured
        # object boundary, even if a rectangular final hull would fit a plane.
        with patch('nsl_surface_support.watershed_ift', side_effect=lambda cost, markers: np.where(markers == 2, 2, 1)):
            self.assertEqual(basin_support(grid, planes, nc.plane_quality), [])

    def test_image_border_is_not_a_physical_object_boundary(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        planes = self.partial(grid, board)
        # A wall basin can avoid the search fence yet end at the camera border.
        labels = np.full(board.shape, 2, 'int32')
        labels[20:100, :45] = 1
        with patch('nsl_surface_support.watershed_ift', return_value=labels):
            self.assertEqual(basin_support(grid, planes, nc.plane_quality), [])

    def test_patch_of_continuous_wall_is_not_a_bounded_board(self):
        yy, xx = np.mgrid[:120, :160]
        rays = np.stack([(xx-80)/65, (yy-60)/65, np.ones_like(xx)], -1)
        footprint = (xx > 50) & (xx < 90) & (yy > 35) & (yy < 75)
        for pose in ((0., 0., 0.), (.2, -.4, .1)):
            basis = cv2.Rodrigues(np.array(pose))[0]
            grid = rays*.8@basis.T
            plane = dict(mask=footprint, normal=basis[:, 2], h=.8)
            self.assertFalse(bounded_surface(grid, plane))

    def test_identity_proposals_do_not_change_calibration_candidates(self):
        grid, board = fixtures.SurfaceSupportTests.board()
        original = complete_support(grid, [dict(mask=board)], nc.plane_quality)
        supplemental = dict(original[0], calibration_eligible=False)
        frame = dict(xyz_mm=grid.reshape(-1, 3)*1000, pcd_shape=board.shape,
                     source_candidates=[supplemental]+original)
        self.assertIs(nc.board_candidates(frame)[0], supplemental)
        actual = nc.board_candidates(frame, for_calibration=True)
        self.assertEqual(len(actual), len(original))
        for before, after in zip(original, actual):
            self.assertIs(before, after)
        frame['source_candidates'] = [supplemental]
        self.assertEqual(nc.board_candidates(frame, for_calibration=True), [])


if __name__ == '__main__':
    unittest.main()
