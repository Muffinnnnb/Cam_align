"""Small regressions: python -m unittest test_nsl_color -v"""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import nsl_color_calibration as nc
from fuse_color_pointcloud import fuse_points, load_rt
from nsl_geometry import range_to_xyz, geometry_spec, check_geometry, rt_geometry, fusion_block_reason


class GeometryTests(unittest.TestCase):
    def test_nf_guard_checks_before_projection_cache(self):
        frame={'source_geometry':{'mode':'pcd'},'source_lens':'nf'}
        self.assertTrue(fusion_block_reason(frame))
        with self.assertRaisesRegex(ValueError,'NF 원본 융합 차단'):
            nc.project_to_color(frame, {})
        frame['allow_nf_comparison']=True
        self.assertFalse(fusion_block_reason(frame))
        self.assertFalse(fusion_block_reason({'source_geometry':{'mode':'wf_range'}}))

    def test_bad_rt_does_not_prefer_bad_support_when_all_candidates_fail(self):
        target={'normal':np.array([0.,0.,-1.]),'h':-1.}
        strip={**target,'quality_ok':False,'quality_reasons':['too narrow']}
        board={'normal':cv2.Rodrigues(np.array([.1,0.,0.]))[0] @ target['normal'],
               'h':-1.,'quality_ok':True,'quality_reasons':[]}
        frame={'timestamp':'sample','target_plane':target,'source_candidates':[strip,board]}
        with patch.object(nc,'plane_projection_metrics',return_value={'overlap':1.,'coverage':1.,'iou':1.}):
            selected=nc.select_plane_for_rt(frame,{'R':np.eye(3),'t':np.zeros(3)})
        self.assertIs(selected,board)
        self.assertFalse(frame['plane_valid'])  # Detection does not imply a good RT.

    def test_range_recovery_preserves_distance_and_invalid_codes(self):
        raw = np.full((240,320), 750, np.uint16)
        raw[0,:2] = [0,64000]
        for lens in ('nf','wf'):
            xyz = range_to_xyz(raw,lens)
            self.assertTrue(np.isnan(xyz[:2]).all())
            np.testing.assert_allclose(np.linalg.norm(xyz[2:],axis=1),750,atol=1e-10)
            self.assertLess(xyz[160,1],0)
        with self.assertRaises(ValueError):
            range_to_xyz(raw.astype(np.float32))
        with self.assertRaises(ValueError):
            range_to_xyz(raw[:120])

    def test_rt_geometry_cannot_be_mixed_or_reinterpreted(self):
        rt = {'source_geometry':geometry_spec('wf_range')}
        with self.assertRaises(ValueError):
            check_geometry({},rt)
        check_geometry({'source_geometry':geometry_spec('wf_range')},rt)
        self.assertEqual(rt_geometry({}),'pcd')
        rt['source_geometry']['pixel_pitch_mm']=.010
        with self.assertRaises(ValueError):
            rt_geometry(rt)

    def test_source_z_range_and_empty_projection(self):
        image = np.full((10, 10, 3), [10, 20, 30], np.uint8)
        args = (np.array([[0, 0, 1000.], [0, 0, 3000.]]), np.ones(2), image,
                np.eye(3), np.zeros(5), np.eye(3), np.array([0, 0, 2.]))
        points, rgb, _, _, _, stats = fuse_points(*args, .9, 1.1)
        np.testing.assert_allclose(points, [[0, 0, 3]])
        np.testing.assert_array_equal(rgb, [[30, 20, 10]])
        self.assertEqual(stats["valid_source_points"], 1)
        self.assertEqual(len(fuse_points(*args, 4., 5.)[0]), 0)

    def test_uint8_ply_mask(self):
        frame = {"xyz_mm": np.array([[i, 0, 1000.] for i in range(4)]),
                 "intensity": np.ones(4), "z_min": .1, "z_max": 2,
                 "source_plane": {"mask": np.array([[0, 1], [0, 1]], np.uint8)},
                 "rt_inlier": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.ply"
            self.assertEqual(nc.save_source_cloud(path, frame, include_intensity=True), (4, 2))
            header, payload = path.read_bytes().split(b"end_header\n", 1)
            self.assertIn(b"coordinate_frame NSL camera", header)
            # xyz:12 bytes, RGB:3 bytes, intensity:4 bytes.
            colors = np.frombuffer(payload, np.uint8).reshape(4, 19)[:, 12:15]
            np.testing.assert_array_equal(colors, [[180, 180, 180], nc.PLANE_CANDIDATE_RGB,
                                                   [180, 180, 180], nc.PLANE_CANDIDATE_RGB])
            original=path.read_bytes()
            frame['rt_inlier']=False
            nc.save_source_cloud(path,frame,include_intensity=True)
            self.assertEqual(path.read_bytes(),original)

    def test_legacy_rt_is_revalidated_without_recalibration(self):
        plane = {"normal": np.array([0., 0., -1.]), "h": -1., 'quality_ok': True, 'quality_reasons': []}
        frame = {"timestamp": "sample", "source_candidates": [plane], "target_plane": plane}
        rt = {"R": np.eye(3), "t": np.zeros(3), "ransac_inlier_frames": ["sample"]}
        with patch.object(nc, "plane_projection_metrics", return_value={'overlap':1.,'coverage':1.,'iou':1.}):
            nc.select_plane_for_rt(frame, rt)
            self.assertFalse(frame["rt_inlier"])
            self.assertEqual(nc.plane_status(frame, rt)["state"], "candidate")
            # Missing/stale provenance must not claim confirmed training membership.
            rt["fitted_plane_selection"] = {"sample": {"candidate_index": 99, "normal": [1, 0, 0], "h": -2.}}
            nc.select_plane_for_rt(frame, rt)
            self.assertFalse(frame["fitted_plane_reproduced"])
            self.assertFalse(frame["rt_inlier"])
            self.assertEqual(nc.plane_status(frame, rt)["state"], "candidate")
        with patch.object(nc, "plane_projection_metrics", return_value={'overlap':.2,'coverage':.2,'iou':.2}):
            nc.select_plane_for_rt(frame, rt)
            self.assertEqual(nc.plane_status(frame, rt)["state"], "candidate")

    def test_whole_board_not_narrow_strip(self):
        x,y=np.meshgrid(np.linspace(-.125,.125,30),np.linspace(-.14,.14,30))
        points=np.column_stack((x.ravel(),y.ravel(),np.ones(x.size)))
        self.assertTrue(nc.plane_quality(points,np.array([0,0,-1.]))['quality_ok'])
        self.assertFalse(nc.plane_quality(points[abs(points[:,0])<.04],np.array([0,0,-1.]))['quality_ok'])

    def test_one_way_overlap_cannot_validate_small_patch(self):
        plane={'normal':np.array([0.,0.,-1.]),'h':-1.,'quality_ok':True,'quality_reasons':[]}
        frame={'timestamp':'patch','source_candidates':[plane],'target_plane':plane}
        rt={'R':np.eye(3),'t':np.zeros(3)}
        with patch.object(nc,'plane_projection_metrics',return_value={'overlap':1.,'coverage':.2,'iou':.2}):
            nc.select_plane_for_rt(frame,rt)
        self.assertFalse(frame['plane_valid'])
        self.assertIn('면적 커버',';'.join(frame['quality_failures']))

    def test_surface_occlusion_between_foreground_samples(self):
        xyz=np.array([[-10,-10,1000],[10,-10,1000],[-10,10,1000],[10,10,1000],
                      [np.nan]*3,[np.nan]*3,[0,0,1500],[np.nan]*3])
        k=np.array([[100.,0,20],[0,100.,20],[0,0,1]])
        args=(xyz,np.ones(8),np.zeros((40,40,3),np.uint8),k,np.zeros(5),np.eye(3),np.zeros(3),.1,2.)
        old=fuse_points(*args,return_indices=True)
        new=fuse_points(*args,return_indices=True,organized_shape=(4,2))
        self.assertIn(6,old[-1])
        self.assertNotIn(6,new[-1])
        self.assertEqual(set(new[-1]),{0,1,2,3})
        self.assertEqual(new[5]['surface_occluded_points'],1)
        # Do not bridge a real depth discontinuity with a false foreground wall.
        xyz[2:4,2]=2000
        unconnected=fuse_points(*args,return_indices=True,organized_shape=(4,2))
        self.assertIn(6,unconnected[-1])

    def test_rgb_export_and_preview_do_not_depend_on_planes(self):
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        frame = {"xyz_mm": np.array([[0, 0, 1000.], [1000, 0, 1000.], [0, 1000, 1000.], [1000, 1000, 1000.]]),
                 "intensity": np.arange(4), "pcd_shape": (2, 2), "color": image,
                 "z_min": .1, "z_max": 2, "source_candidates": [], "target_plane": None}
        rt = {"R": np.eye(3).tolist(), "t": [0., 0., 0.]}
        with tempfile.TemporaryDirectory() as directory, patch.object(nc, "camera", return_value=(np.eye(3), np.zeros(5))):
            self.assertEqual(nc.plane_status(frame, rt)["state"], "failed")
            path = Path(directory) / "rgb.ply"
            stats = nc.save_fused_cloud(path, frame, rt)
            header, payload = path.read_bytes().split(b"end_header\n", 1)
            self.assertNotIn(b"property float intensity", header)
            saved = np.frombuffer(payload, np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)]))
            np.testing.assert_array_equal(saved["rgb"], image[:2, :2, ::-1].reshape(-1, 3))
            preview, count = nc.fused_color_view(frame, rt)
            np.testing.assert_array_equal(preview, image[:2, :2])
            self.assertEqual(count, len(saved))
            self.assertTrue(Path(stats["preview"]).exists())
            meta = json.loads(Path(stats["metadata"]).read_text(encoding='utf-8'))
            self.assertEqual(meta["color_source"], "original Color image RGB")
            nc.save_fused_cloud(path, frame, rt, include_intensity=True)
            self.assertIn(b"property float intensity", path.read_bytes().split(b"end_header\n", 1)[0])

    def test_empty_export_does_not_write_a_success_file(self):
        frame = {"xyz_mm": np.array([[0, 0, 1000.]]), "intensity": np.ones(1),
                 "color": np.zeros((4, 4, 3), np.uint8), "pcd_shape": (1, 1), "z_min": 2., "z_max": 3.}
        rt = {"R": np.eye(3).tolist(), "t": [0., 0., 0.]}
        with tempfile.TemporaryDirectory() as directory, patch.object(nc, "camera", return_value=(np.eye(3), np.zeros(5))):
            path = Path(directory) / "empty.ply"
            with self.assertRaises(ValueError):
                nc.save_fused_cloud(path, frame, rt)
            self.assertFalse(path.exists())

    def test_invalid_rt_and_mm_translation(self):
        rt = {"from": "nsl", "to": "color", "R": np.eye(3).tolist(), "t": [100, 0, 0], "translation_unit": "mm"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rt.json"
            path.write_text(json.dumps(rt))
            _, t, normalized = load_rt(path)
            self.assertAlmostEqual(t[0], .1)
            self.assertEqual(normalized["RT"][0][3], .1)
            rt["to"] = "depth_camera"
            path.write_text(json.dumps(rt))
            with self.assertRaises(ValueError):
                load_rt(path)

    def test_occluder_is_kept_until_z_buffer(self):
        x, y = np.meshgrid(np.linspace(-.12, .12, 12), np.linspace(-.12, .12, 12))
        board = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        xyz = np.vstack((board, board * .8)) * 1000
        frame = {"xyz_mm": xyz, "z_min": .1, "z_max": 2, "pcd_shape": (24, 12),
                 "color": np.zeros((40, 40, 3), np.uint8),
                 "target_plane": {"center": np.array([0, 0, 1.]), "normal": np.array([0, 0, -1.]),
                                  "h": -1., "region": np.array([[15, 15], [25, 15], [25, 25], [15, 25]])}}
        k = np.array([[30., 0, 20], [0, 30., 20], [0, 0, 1.]])
        with patch.object(nc, "camera", return_value=(k, np.zeros(5))):
            self.assertIsNone(nc.guided_plane_from_color(frame, {"R": np.eye(3), "t": np.zeros(3)}))

    def test_lm_final_inliers(self):
        rng = np.random.default_rng(41)
        normals = rng.normal(size=(50, 3))
        normals[:, 2] = -abs(normals[:, 2]) - 1
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        r = cv2.Rodrigues(np.array([.05, -.12, .03]))[0]
        t = np.array([.12, -.02, .04])
        transformed = normals @ r.T
        offsets = rng.uniform(-1.2, -.5, 50)
        target_offsets = offsets + transformed @ t + rng.normal(0, .006, 50)
        target_offsets[:8] += .10
        frames = {str(i): {"source_plane": {"normal": n, "h": h},
                           "target_plane": {"normal": nr, "h": hr}}
                  for i, (n, h, nr, hr) in enumerate(zip(normals, offsets, transformed, target_offsets))}
        result = nc.calibrate(frames, None, "source_plane", "target_plane", "nsl", "color")
        self.assertEqual(result["nonlinear_method"], "LM")
        for name in result["ransac_inlier_frames"]:
            source, target = frames[name]["source_plane"], frames[name]["target_plane"]
            predicted = np.array(result["R"]) @ source["normal"]
            angle = np.rad2deg(np.arccos(np.clip(predicted @ target["normal"], -1, 1)))
            distance = abs(source["h"] + predicted @ result["t"] - target["h"])
            self.assertLess(angle, 1.5)
            self.assertLess(distance, .008)


if __name__ == "__main__":
    unittest.main()
