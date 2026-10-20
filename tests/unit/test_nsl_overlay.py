"""Projection glyphs only change display pixels, never measurements or RGB."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from nsl_overlay import point_overlay
from nsl_color_calibration import projection_view, save_projection


class OverlayTests(unittest.TestCase):
    def test_cross_has_exact_arms_and_dot_is_a_disk(self):
        image = np.zeros((15, 15, 3), np.uint8)
        cross = point_overlay(image, [[7, 7]], [[255]*3], [1], size=7, shape='cross', opacity=1)
        disk = point_overlay(image, [[7, 7]], [[255]*3], [1], size=7, opacity=1)
        mask = np.zeros((15, 15), bool)
        mask[7, 4:11] = mask[4:11, 7] = True
        np.testing.assert_array_equal(cross[:, :, 0] != 0, mask)
        self.assertTrue(disk[8, 8].any())
        self.assertFalse(cross[8, 8].any())
        self.assertTrue(np.all(image == 0))

    def test_one_pixel_cross_and_dot_match(self):
        image = np.zeros((5, 5, 3), np.uint8)
        a = point_overlay(image, [[2, 2]], [[17, 23, 97]], [1])
        b = point_overlay(image, [[2, 2]], [[17, 23, 97]], [1], shape='cross')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(np.count_nonzero(a.any(axis=2)), 1)

    def test_nearest_marker_wins_and_blends_only_once(self):
        image = np.full((10, 10, 3), 20, np.uint8)
        pixels, colors, depths = np.array([[4, 4], [5, 4]]), np.array([[0, 100, 0], [100, 0, 0]]), np.array([1., 2.])
        args = [a.copy() for a in (pixels, colors, depths)]
        for shape in ('dot', 'cross'):
            forward = point_overlay(image, pixels, colors, depths, size=5, shape=shape, opacity=.5)
            backward = point_overlay(image, pixels[::-1], colors[::-1], depths[::-1], size=5, shape=shape, opacity=.5)
            np.testing.assert_array_equal(forward, backward)
            np.testing.assert_array_equal(forward[4, 5], [10, 60, 10])
        for before, after in zip(args, (pixels, colors, depths)):
            np.testing.assert_array_equal(before, after)

    def test_clipping_invalid_points_and_empty(self):
        image = np.zeros((5, 5, 3), np.uint8)
        xy = [[0, 0], [-1, 2], [5, 2], [np.nan, 0], [2, 2], [3, 3], [4.8, 4]]
        out = point_overlay(image, xy, [[100]*3]*len(xy), [1, 1, 1, 1, 0, np.inf, 1],
                            size=5, shape='cross', opacity=1)
        self.assertEqual(np.count_nonzero(out.any(axis=2)), 5)
        for pixels, colors, depths in (([], [], []), ([[1, 1]], [[100]*3], [-1])):
            out = point_overlay(image, pixels, colors, depths)
            np.testing.assert_array_equal(out, image)
            self.assertIsNot(out, image)

    def test_validation(self):
        args = (np.zeros((5, 5, 3), np.uint8), [[2, 2]], [[100]*3], [1])
        for kwargs in ({'size': 0}, {'size': 4}, {'size': 23}, {'size': 3.0},
                       {'shape': 'unknown'}, {'opacity': -1}, {'opacity': 2}, {'opacity': np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                point_overlay(*args, **kwargs)
        with self.assertRaises(ValueError):
            point_overlay(args[0], [[1, 1]], [], [1])

    def test_default_projection_is_original_one_pixel_blend(self):
        rng = np.random.default_rng(3)
        original = rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)
        u, v = np.tile(np.arange(40), 30), np.repeat(np.arange(30), 40)
        points = rng.uniform(.1, 2, (len(u), 3)).astype('float32')
        z = points[:, 2]
        lo, hi = np.percentile(z, [2, 98])
        q = np.uint8(np.clip((z-lo)*255/(hi-lo), 0, 255))
        colors = cv2.applyColorMap(q[:, None], cv2.COLORMAP_TURBO).reshape(-1, 3)
        expected = original.copy()
        expected[v, u] = np.rint(expected[v, u]*.3+colors*.7).astype(np.uint8)
        frame = dict(color=original.copy(), aligned_depth=np.zeros((30, 40)))
        with patch('nsl_color_calibration.project_to_color', return_value=(points, None, None, u, v, {})):
            actual, count = projection_view(frame, {})
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(count, len(u))
            np.testing.assert_array_equal(frame['color'], original)
            self.assertFalse(frame['aligned_depth'].any())

    def test_save_forwards_style_and_original_resolution(self):
        image = np.zeros((12, 16, 3), np.uint8)
        with patch('nsl_color_calibration.projection_view', return_value=(image, 3)) as render, \
             patch('nsl_color_calibration.write_image') as write:
            self.assertEqual(save_projection('overlay.png', {}, {}, point_size=7, point_shape='cross'), 3)
            self.assertEqual(render.call_args.kwargs, dict(point_size=7, point_shape='cross'))
            self.assertIs(write.call_args.args[1], image)


if __name__ == '__main__':
    unittest.main()
