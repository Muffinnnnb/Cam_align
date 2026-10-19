"""Board identity is not plane precision; no scene-specific pixel ROI or baseline."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from nsl_board_identity import identify_board, association_model, frame_fingerprint
import nsl_color_calibration as nc


def scene(rotation=(0., 0., 0.), translation=(0., 0., 0.), board_pose=(0., 0., 0.), focal=600, depth=2.):
    R = cv2.Rodrigues(np.array(rotation, float))[0]
    B = cv2.Rodrigues(np.array(board_pose, float))[0]
    t = np.array(translation, float)
    corners = np.array([[-.125, -.14, 0], [.125, -.14, 0], [.125, .14, 0], [-.125, .14, 0]])@B.T + [0, 0, depth]
    x, y = np.meshgrid(np.linspace(-.123, .123, 20), np.linspace(-.138, .138, 20))
    board = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]@B.T + [0, 0, depth]
    # Bigger, perfectly flat off-target surface must not win by point count/flatness.
    body = board + [.02, .02, .35]
    source = (np.vstack([body, board])-t)@R
    mask = np.zeros((40, 20), bool); mask[20:] = True
    candidates = [dict(mask=~mask, support_quality={'eligible': True}),
                  dict(mask=mask, support_quality={'eligible': True})]
    K = np.array([[focal, 0., 320], [0., focal*1.1, 240], [0., 0., 1.]])
    D = np.zeros(5)
    frame = dict(timestamp='arbitrary', xyz_mm=source*1000, pcd_shape=(40, 20),
                 color=np.zeros((480, 640, 3), np.uint8), source_candidates=candidates,
                 source_plane=candidates[0], target_plane={'corners3d': corners})
    model = dict(R=R.tolist(), t=t.tolist(), target_intrinsics=K.tolist(),
                 target_distortion=D.tolist(), target_image_size=[640, 480])
    return frame, model, K, D


class BoardIdentityTests(unittest.TestCase):
    def test_pose_baseline_board_tilt_and_intrinsics_do_not_change_identity(self):
        for yaw in (-1.2, -.4, .6, 1.2):
            for shift in (-1.4, 0., 1.4):
                for focal in (280., 1400.):
                    with self.subTest(yaw=yaw, shift=shift, focal=focal):
                        f, rt, K, D = scene((.2, yaw, -.3), (shift, -.5, .2), (.4, -.7, .5), focal, depth=6.)
                        self.assertTrue(np.all(f['xyz_mm'][:, 2] > 0))
                        old = f['xyz_mm'].copy(); masks = [p['mask'].copy() for p in f['source_candidates']]
                        result = identify_board(f, rt, K, D)
                        self.assertEqual(result['state'], 'matched')
                        self.assertIs(result['plane'], f['source_candidates'][1])
                        np.testing.assert_array_equal(f['xyz_mm'], old)
                        for p, mask in zip(f['source_candidates'], masks):
                            np.testing.assert_array_equal(p['mask'], mask)
                        self.assertIs(f['source_plane'], f['source_candidates'][0])

    def test_same_infinite_plane_outside_board_is_not_board(self):
        f, rt, K, D = scene()
        f['xyz_mm'][:400] = f['xyz_mm'][400:] + [500., 0., 0.]
        f['source_candidates'] = f['source_candidates'][:1]
        r = identify_board(f, rt, K, D)
        self.assertEqual(r['state'], 'unconfirmed'); self.assertIsNone(r['plane'])

    def test_partial_is_not_training_quality_and_never_completes_support(self):
        f, rt, K, D = scene()
        mask = f['source_candidates'][1]['mask']; mask[30:] = False
        f['source_candidates'][1]['support_quality']['eligible'] = False
        before = mask.copy()
        r = identify_board(f, rt, K, D)
        self.assertEqual(r['state'], 'partial')
        np.testing.assert_array_equal(mask, before)
        self.assertFalse(r['plane']['support_quality']['eligible'])

    def test_similarly_supported_disjoint_candidates_abstain(self):
        f, rt, K, D = scene()
        f['xyz_mm'][:400] = f['xyz_mm'][400:]
        r = identify_board(f, rt, K, D)
        self.assertEqual(r['state'], 'ambiguous'); self.assertIsNone(r['plane'])

    def test_no_rt_does_not_pretend_source_candidate_is_identified(self):
        f, rt, K, D = scene()
        f['board_identity'] = identify_board(f, None, K, D)
        self.assertEqual(f['board_identity']['state'], 'unverified')
        self.assertIsNone(nc.displayed_board(f))
        self.assertEqual(nc.plane_status(f)['state'], 'unconfirmed')

    def test_camera_changes_require_recalibration(self):
        for change in ('K', 'distortion', 'resolution'):
            f, rt, K, D = scene()
            if change == 'K': K[0, 0] *= 1.1
            if change == 'distortion': D[0] = .1
            if change == 'resolution': f['color'] = f['color'][::2, ::2]
            model, reason = association_model(f, rt, K, D)
            self.assertIsNone(model); self.assertIn('변경', reason)

    def test_crossfit_requires_exact_input_and_disjoint_training(self):
        f, rt, K, D = scene()
        heldout = dict(R=rt['R'], t=rt['t'], training_frames=['another'],
                       heldout_fingerprints={f['timestamp']: frame_fingerprint(f)})
        rt['board_identity_models'] = [heldout]
        self.assertIs(association_model(f, rt, K, D)[0], heldout)
        heldout['training_frames'].append(f['timestamp'])
        self.assertIsNone(association_model(f, rt, K, D)[0])
        heldout['training_frames'].pop()
        f['xyz_mm'][0, 0] += 1
        model, basis = association_model(f, rt, K, D)
        self.assertIs(model, rt); self.assertIn('독립 정확도 검증 아님', basis)

    def test_behind_camera_is_not_a_valid_identity(self):
        f, rt, K, D = scene()
        rt['t'][2] = -4
        f['target_plane']['corners3d'][:, 2] -= 4
        self.assertEqual(identify_board(f, rt, K, D)['state'], 'unconfirmed')

    def test_rgb_projection_does_not_use_association_to_hide_or_move_points(self):
        f, rt, K, D = scene()
        f.update(z_min=.1, z_max=20., intensity=np.ones(len(f['xyz_mm'])))
        f['color'][:] = [15, 65, 215]
        with patch.object(nc, 'camera', return_value=(K, D)):
            before = nc.project_to_color(f, rt, return_indices=True)
            f['board_identity'] = identify_board(f, rt, K, D)
            f.pop('_color_projection', None)
            after = nc.project_to_color(f, rt, return_indices=True)
        for i in (0, 1, 3, 4, 6):
            np.testing.assert_array_equal(before[i], after[i])


if __name__ == '__main__':
    unittest.main()
