"""Independent NSL dimensions, fixed ray model, pixel diagnostics and real RGB."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import cv2
import numpy as np

import nsl_color_calibration as nc
from nsl_geometry import geometry_spec, ray_scale, scale_ray_angles
from nsl_dimensions import estimate_source_correction, measured_size
from nsl_projection_fit import refine_projection, sample_polygon
from fuse_color_pointcloud import fuse_points


class ProjectionTests(unittest.TestCase):
    def test_ray_correction_preserves_range_invalids_and_input(self):
        p=np.array([[.2,-.1,1.],[0,0,2.],[np.nan,np.nan,np.nan],[0,0,0.]])
        original=p.copy();q=scale_ray_angles(p,.88)
        np.testing.assert_allclose(np.linalg.norm(q,axis=1),np.linalg.norm(p,axis=1),equal_nan=True)
        np.testing.assert_array_equal(p,original)
        self.assertTrue(np.isnan(q[2]).all())
        np.testing.assert_array_equal(q[1],[0,0,2.])

    def test_correction_requires_explicit_wf_metadata(self):
        self.assertEqual(ray_scale({}),1.)
        correction={'model':'radial_angle_scale','scale':.9}
        with self.assertRaises(ValueError):ray_scale({'source_ray_correction':correction})
        meta={'source_geometry':geometry_spec('wf_range'),'source_ray_correction':correction}
        self.assertEqual(ray_scale(meta),.9)
        correction['scale']=float('nan')
        with self.assertRaises(ValueError):ray_scale(meta)

    def test_fusion_samples_actual_pixel_after_correction(self):
        xyz=np.array([[200.,100.,500.],[-100.,50.,500.],[0.,0.,500.]])
        image=np.indices((100,100))[1].astype('uint8')[...,None]*np.array([1,2,1],np.uint8)
        K=np.array([[80.,0,50],[0,80.,50],[0,0,1.]])
        points,rgb,_,u,v,stats,ids=fuse_points(xyz,np.ones(3),image,K,np.zeros(5),np.eye(3),np.zeros(3),
                                           .49,.51,False,True,source_ray_scale=.9)
        expected=scale_ray_angles(xyz*.001,.9)
        np.testing.assert_allclose(points,expected[ids],rtol=1e-6)
        uv=cv2.projectPoints(expected,np.zeros(3),np.zeros(3),K,np.zeros(5))[0].reshape(-1,2)
        np.testing.assert_array_equal(np.c_[u,v],np.rint(uv[ids]))
        np.testing.assert_array_equal(rgb,image[v,u,::-1])
        self.assertEqual(stats['valid_source_points'],3)  # Filter nominal NSL Z, not corrected Z.

    def test_overlap_does_not_expand_board_by_default(self):
        K=np.array([[100.,0,50],[0,100.,50],[0,0,1.]])
        frame={'xyz_mm':np.array([[110.,0.,1000.]]),'pcd_shape':(1,1),
               'color':np.zeros((100,100,3),np.uint8),
               'target_plane':{'region':np.array([[40.,40],[60,40],[60,60],[40,60]])}}
        with patch.object(nc,'camera',return_value=(K,np.zeros(5))):
            q=nc.plane_projection_metrics(frame,{'mask':np.ones((1,1),bool)},dict(R=np.eye(3),t=np.zeros(3)))
        self.assertEqual(q['overlap'],0.)
        self.assertEqual(q['outside_fraction'],1.)
        frame['xyz_mm'][0,0]=2000.  # Entirely outside the Color camera FOV.
        with patch.object(nc,'camera',return_value=(K,np.zeros(5))):
            q=nc.plane_projection_metrics(frame,{'mask':np.ones((1,1),bool)},dict(R=np.eye(3),t=np.zeros(3)))
        self.assertEqual(q['outside_fraction'],1.)

    def test_projection_cache_includes_ray_model_and_checks_intrinsics(self):
        K=np.array([[80.,0,50],[0,80.,50],[0,0,1.]])
        f={'xyz_mm':np.array([[200.,100.,500.]]),'pcd_shape':(1,1),
           'color':np.zeros((100,100,3),np.uint8),'intensity':np.ones(1),
           'z_min':.1,'z_max':2.,'source_geometry':geometry_spec('wf_range')}
        rt={'R':np.eye(3),'t':np.zeros(3),'source_geometry':geometry_spec('wf_range'),
            'source_ray_correction':{'model':'radial_angle_scale','scale':.9}}
        with patch.object(nc,'camera',return_value=(K,np.zeros(5))):
            before=nc.project_to_color(f,rt,visible_only=False)[0].copy()
            rt['source_ray_correction']['scale']=.8
            after=nc.project_to_color(f,rt,visible_only=False)[0]
            self.assertFalse(np.allclose(before,after))
            rt['target_intrinsics']=K*2
            with self.assertRaisesRegex(ValueError,'내부 파라미터'):
                nc.project_to_color(f,rt,visible_only=False)

    def test_source_ply_applies_dimensions_but_not_rt(self):
        f={'xyz_mm':np.array([[200.,100.,500.],[0,0,800.]]),'pcd_shape':(1,2),
           'intensity':np.ones(2),'z_min':.1,'z_max':2.,'source_geometry':geometry_spec('wf_range')}
        rt={'R':-np.eye(3),'t':np.ones(3)*100,'source_geometry':geometry_spec('wf_range'),
            'source_ray_correction':{'model':'radial_angle_scale','scale':.9}}
        with TemporaryDirectory() as folder:
            path=Path(folder)/'source.ply'
            nc.save_source_cloud(path,f,calibration=rt)
            header,data=path.read_bytes().split(b'end_header\n',1)
            self.assertIn(b'before RT',header)
            saved=np.frombuffer(data,np.dtype([('xyz','<f4',3),('rgb','u1',3)]))
            np.testing.assert_allclose(saved['xyz'],scale_ray_angles(f['xyz_mm']*.001,.9),rtol=1e-6)

    def test_nsl_size_estimation_ignores_rgb_and_rt(self):
        frames={};rng=np.random.default_rng(30)
        corners=np.array([[-.125,-.14,0],[.125,-.14,0],[.125,.14,0],[-.125,.14,0]])
        # Enough real boundary samples that the 1% flying-pixel trim retains it.
        board=np.vstack((sample_polygon(corners,200),np.c_[rng.uniform(-.12,.12,200),rng.uniform(-.13,.13,200),np.zeros(200)]))
        for i in range(20):
            R=cv2.Rodrigues(rng.normal(0,.2,3))[0]
            xyz=board@R.T+np.array([rng.uniform(-.15,.15),rng.uniform(-.1,.1),rng.uniform(.6,1.)])
            raw=scale_ray_angles(xyz,1/.88)
            p={'size':measured_size(raw),'quad_support':1.,'edge_support':1.,'support_occupancy':1.,
               'rectangularity':1.,'mask':np.ones((20,20),bool)}
            frames[str(i)]={'source_plane':p,'pcd_shape':(20,20),'xyz_mm':raw*1000}
        correction,audit=estimate_source_correction(frames)
        self.assertIsNotNone(correction)
        self.assertAlmostEqual(correction['scale'],.88,delta=.015)
        self.assertLess(audit['heldout_after_mae_mm'],audit['heldout_before_mae_mm'])
        for f in frames.values():f.update(color='not used',target_plane='not used',RT=np.zeros((4,4)))
        other,_=estimate_source_correction(frames)
        self.assertEqual(correction,other)

    def test_dimension_calibration_refuses_insufficient_data(self):
        correction,audit=estimate_source_correction({})
        self.assertIsNone(correction)
        self.assertFalse(audit['applied'])

    def test_lm_cannot_change_independent_ray_scale(self):
        rng=np.random.default_rng(10);items=[]
        K=np.array([[600.,0,320],[0,600.,240],[0,0,1.]])
        R=cv2.Rodrigues(np.array([.04,-.08,.02]))[0];t=np.array([.20,-.03,.05])
        corners=np.array([[-.125,-.14,0],[.125,-.14,0],[.125,.14,0],[-.125,.14,0]])
        for i in range(12):
            B=cv2.Rodrigues(rng.normal(0,.3,3))[0];c=np.r_[rng.uniform(-.12,.12,2),rng.uniform(.6,1.)]
            edge=sample_polygon(corners,96)@B.T+c
            target=corners@B.T@R.T+c@R.T+t
            normal=-R@B[:,2]
            q={'normal':normal,'h':float(normal@target.mean(0)),'corners3d':target}
            raw=scale_ray_angles(edge,1/.9)
            items.append(dict(name=str(i),points=raw,edge=raw,target=q))
        initial={'R':R,'t':t+[.003,-.002,.003],'ransac_inlier_frames':[str(i) for i in range(12)],
                 'selected_record_indices':{str(i):i for i in range(12)},'source_geometry':geometry_spec('wf_range'),
                 'source_ray_correction':{'model':'radial_angle_scale','scale':.9}}
        result=refine_projection(items,initial,K,np.zeros(5))
        self.assertEqual(result['source_ray_correction']['scale'],.9)
        np.testing.assert_allclose(result['R'],R,atol=1e-5)
        np.testing.assert_allclose(result['t'],t,atol=1e-5)


if __name__=='__main__':unittest.main()
