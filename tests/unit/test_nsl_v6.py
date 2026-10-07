"""Automatic target identity and finite-plane calibration regressions."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_plane_fit import complete_support, fit_point_planes, quadrilateral_support


class AutomaticBoardTests(unittest.TestCase):
    def test_failed_holdout_does_not_overwrite_existing_rt(self):
        x,y=np.meshgrid(np.linspace(-.125,.125,20),np.linspace(-.14,.14,20))
        points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
        n=np.array([0.,0.,-1.]);center=points.mean(0)
        p={**nc.plane_quality(points,n),'normal':n,'h':-1.,'mask':np.ones(x.shape,bool),
           'center':center,'board_center':center,'detector_version':6,'fit_method':'test',
           'support_occupancy':1.,'edge_support':1.,'quad_support':1.}
        frames={str(i):{'timestamp':str(i),'source_candidates':[p.copy()],
                       'target_plane':p,'xyz_mm':points*1000,'pcd_shape':x.shape,'z_min':.1,'z_max':20.}
                for i in range(16)}
        initial={'selected_record_indices':{str(i):i for i in range(16)}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rt.json';path.write_bytes(b'previous calibration')
            with patch.object(nc,'fit_point_planes',side_effect=[initial]+[RuntimeError('degenerate fold')]*4):
                with self.assertRaisesRegex(RuntimeError,'기존 RT/설정 유지'):
                    nc.calibrate_nsl_color(frames,path)
            self.assertEqual(path.read_bytes(),b'previous calibration')

    def test_round_patch_is_not_a_rectangle(self):
        a=np.linspace(0,2*np.pi,128,endpoint=False)
        circle=cv2.convexHull(np.c_[100*np.cos(a),100*np.sin(a)].astype('float32'))
        quad=np.array([[0,0],[110,20],[80,140],[5,100]],np.float32).reshape(-1,1,2)
        self.assertLess(quadrilateral_support(circle),.70)
        self.assertAlmostEqual(quadrilateral_support(quad),1.)

    def test_failed_completion_preserves_original_board(self):
        x,y=np.meshgrid(np.linspace(-.125,.125,60),np.linspace(-.14,.14,60))
        z=np.ones_like(x);z[20:40,20:40]=1.4
        grid=np.stack([x,y,z],axis=-1)
        seed=np.ones_like(z,bool);seed[18:42,18:42]=False
        result=complete_support(grid,[{'mask':seed,'normal':np.array([0.,0.,-1.])}],nc.plane_quality)
        self.assertTrue(result)
        self.assertTrue(result[0]['completion_fallback'])
        self.assertFalse(result[0]['mask'][25,25])
        self.assertGreater(result[0]['full_rmse'], .1)  # Never hide contamination.

    def test_all_invalid_candidates_are_not_forced_to_be_a_board(self):
        x,y=np.meshgrid(np.linspace(-.025,.025,20),np.linspace(-.14,.14,30))
        grid=np.stack([x,y,np.ones_like(x)],axis=-1)
        n=np.array([0.,0.,-1.])
        p={**nc.plane_quality(grid.reshape(-1,3),n),'normal':n,'h':-1.,
           'mask':np.ones(x.shape,bool),'detector_version':6,'fit_method':'test',
           'support_occupancy':1.,'edge_support':1.,'quad_support':1.}
        f={'timestamp':'strip','source_candidates':[p],'source_plane':p,'target_plane':p,
           'pcd_shape':x.shape,'xyz_mm':grid.reshape(-1,3)*1000,'rt_point_rms_m':0.001}
        self.assertIsNone(nc.select_plane_for_rt(f,{'R':np.eye(3),'t':np.zeros(3)}))
        self.assertEqual(nc.plane_status(f)['state'],'failed')
        self.assertNotIn('rt_point_rms_m',f)

    def test_partial_centers_large_baseline_and_one_vote_per_frame(self):
        rng=np.random.default_rng(72)
        R=cv2.Rodrigues(np.array([.6,-.4,.25]))[0];t=np.array([.9,-.35,.8])
        records=[]
        corners=np.array([[-.125,-.14,0],[.125,-.14,0],[.125,.14,0],[-.125,.14,0]])
        for i in range(24):
            B=cv2.Rodrigues(rng.normal(0,.30,3))[0]
            c=np.r_[rng.uniform(-.2,.2,2),rng.uniform(.6,1.2)]
            xy=rng.uniform([-.125,-.14],[.02,.14],(140,2)) # Partial board.
            p=np.c_[xy,np.zeros(len(xy))]@B.T+c
            normal=-R@B[:,2];center=R@c+t
            target={'normal':normal,'h':normal@center,'center':center,
                    'corners3d':corners@B.T@R.T+center}
            # Wrong candidate first; it must not multiply a frame's vote count.
            wrong=p+rng.uniform(-.5,.5,3)
            for q in (wrong,p):
                records.append((str(i),q,target,q.mean(0)+[.1,0,0]))
        result=fit_point_planes(records)
        self.assertEqual(result['ransac_inliers'],24)
        self.assertEqual(result['plane_pairs'],24)
        self.assertEqual(result['candidate_count'],48)
        np.testing.assert_allclose(result['R'],R,atol=1e-6)
        np.testing.assert_allclose(result['t'],t,atol=1e-6)
        self.assertFalse(result['center_constraint'])
        self.assertEqual(result['lm_loss'],'linear')


if __name__=='__main__':unittest.main()
