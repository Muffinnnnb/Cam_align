"""Fixed support and point-to-plane regressions; no hardware needed."""
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_plane_fit import complete_support, fit_point_planes, support_quality


class PlaneFitTests(unittest.TestCase):
    def test_noisy_board_eligible_but_narrow_strip_not(self):
        x,y=np.meshgrid(np.linspace(-.125,.125,40),np.linspace(-.14,.14,40))
        rng=np.random.default_rng(4)
        points=np.c_[x.ravel(),y.ravel(),1+rng.normal(0,.009,x.size)]
        p={**nc.plane_quality(points,np.array([0.,0.,-1.])), 'support_occupancy':1.,'edge_support':1.}
        self.assertFalse(p['quality_ok'])  # 6mm precision warning is NOT exclusion.
        self.assertTrue(support_quality(points,p)['eligible'])
        narrow=points[abs(points[:,0])<.04]
        p.update(nc.plane_quality(narrow,np.array([0.,0.,-1.])))
        self.assertFalse(support_quality(narrow,p)['eligible'])

    def test_training_history_is_separate_from_detection_and_precision(self):
        n=np.array([0.,0.,-1.]);plane={'normal':n,'h':-1.,'quality_ok':False,'quality_reasons':['6mm precision'],
            'mask':np.ones((2,2),bool),'center':np.array([0.,0.,1.])}
        frame={'timestamp':'used','source_candidates':[plane],'target_plane':plane,
            'xyz_mm':np.tile([0.,0.,1000.],(4,1)),'pcd_shape':(2,2)}
        rt={'R':np.eye(3),'t':np.zeros(3),'ransac_inlier_frames':['used'],
            'fitted_plane_selection':{'used':{'normal':n,'h':-1.,'support_signature':nc.support_signature(frame,plane)}}}
        with patch.object(nc,'plane_projection_metrics',return_value={'overlap':1.,'coverage':1.,'iou':1.}):
            nc.select_plane_for_rt(frame,rt)
            self.assertFalse(frame['plane_valid'])
            self.assertTrue(frame['rt_inlier'])
            self.assertEqual(nc.plane_status(frame,rt)['state'],'candidate')
            self.assertEqual(nc.plane_status(frame,rt)['training'],'used')
            frame['xyz_mm'][0,0]=1.  # Same filename/plane equation, different measured points.
            nc.select_plane_for_rt(frame,rt)
            self.assertFalse(frame['rt_inlier'])
            frame['source_path']=Path('other/capture/NSL_pointcloud/cloud.pcd')
            rt['capture_root']='original/capture'
            nc.select_plane_for_rt(frame,rt)
            self.assertFalse(frame['training_record_known'])

    def test_four_corners_cannot_fill_missing_board(self):
        xyz = np.array([[-.12,-.13,1.],[.12,-.13,1.],[.12,.13,1.],[-.12,.13,1.]])*1000
        K = np.array([[100.,0,30],[0,100.,30],[0,0,1.]])
        polygon = np.array([[17.5,16],[42.5,16],[42.5,44],[17.5,44.]])
        frame = {'xyz_mm':xyz, 'pcd_shape':(2,2), 'color':np.zeros((60,60,3),np.uint8),
                 'target_plane':{'region':polygon}}
        with patch.object(nc,'camera',return_value=(K,np.zeros(5))):
            q = nc.plane_projection_metrics(frame, {'mask':np.ones((2,2),bool)}, {'R':np.eye(3),'t':np.zeros(3)})
        self.assertLessEqual(q['coverage'], 4/64)
        self.assertLess(q['iou'], .1)

    def test_complete_support_does_not_hide_warped_middle(self):
        x,y = np.meshgrid(np.linspace(-.125,.125,64),np.linspace(-.14,.14,64))
        z = np.ones_like(x)
        z[16:48,16:48] += .020
        grid = np.stack((x,y,z),axis=-1)
        seed = np.ones_like(z,bool); seed[12:52,12:52]=False
        p = complete_support(grid,[{'mask':seed,'normal':np.array([0.,0.,-1.])}],nc.plane_quality)[0]
        self.assertGreater(p['full_rmse'], .007)
        self.assertGreater(np.count_nonzero(p['mask'][16:48,16:48]), 500)
        self.assertFalse(p['quality_ok'])

    def test_lm_fixed_points_with_bad_frame_rejection(self):
        rng=np.random.default_rng(11)
        R=cv2.Rodrigues(np.array([.08,-.2,.04]))[0];t=np.array([.24,-.04,.07])
        records=[]
        for i in range(36):
            B=cv2.Rodrigues(rng.normal(0,.45,3))[0]
            c=np.r_[rng.uniform(-.25,.25,2),rng.uniform(.6,1.2)]
            xy=rng.uniform([-.125,-.14],[.125,.14],(200,2))
            p=np.c_[xy,np.zeros(len(xy))]@B.T+c
            n=-R@B[:,2];center=R@c+t
            if i<5: center+=np.array([.15,-.18,.12])
            corners=np.array([[-.125,-.14,0],[.125,-.14,0],[.125,.14,0],[-.125,.14,0]])@B.T@R.T+center
            records.append((str(i),p,{'normal':n,'h':n@center,'center':center,'corners3d':corners},c))
        result=fit_point_planes(records)
        np.testing.assert_allclose(result['R'],R,atol=1e-7)
        np.testing.assert_allclose(result['t'],t,atol=1e-7)
        self.assertEqual(result['nonlinear_method'],'LM')
        self.assertEqual(result['lm_loss'],'linear')
        self.assertTrue(result['consensus_rechecked'])
        self.assertTrue(set(map(str,range(5))).isdisjoint(result['ransac_inlier_frames']))

    def test_parallel_planes_are_rejected(self):
        x,y=np.meshgrid(np.linspace(-.125,.125,12),np.linspace(-.14,.14,12))
        records=[]
        for i in range(8):
            c=np.array([0.,0.,1.+i*.1]); p=np.c_[x.ravel(),y.ravel(),np.full(x.size,c[2])]
            records.append((str(i),p,{'normal':np.array([0.,0.,-1.]),'h':-c[2],'center':c},c))
        with self.assertRaises(RuntimeError):fit_point_planes(records)

    def test_rt_cannot_select_a_more_flattering_new_plane(self):
        n=np.array([0.,0.,-1.]); good={'normal':n,'h':-1.,'quality_ok':True,'quality_reasons':[],
            'fit_method':'fixed','board_center':np.array([0,0,1.]),'center':np.array([0,0,1.]),'mask':np.ones((2,2),bool)}
        source={**good,'normal':cv2.Rodrigues(np.array([.2,0,0]))[0]@n}
        frame={'timestamp':'fixed','source_candidates':[source,good], 'target_plane':good,
            'xyz_mm':np.tile([0.,0.,1000.],(4,1)),'pcd_shape':(2,2)}
        with patch.object(nc,'plane_projection_metrics',return_value={'overlap':1.,'coverage':1.,'iou':1.}):
            self.assertIs(nc.select_plane_for_rt(frame,{'R':np.eye(3),'t':np.zeros(3)}),source)
        self.assertFalse(frame['plane_valid'])

    def test_saved_training_support_cannot_replace_independent_detection(self):
        n=np.array([0.,0.,-1.])
        first={'normal':n,'h':-1.,'quality_ok':True,'quality_reasons':[],
               'center':np.array([0.,0.,1.]),'mask':np.array([[1,1],[0,0]],bool)}
        learned={**first,'mask':np.array([[0,0],[1,1]],bool)}
        frame={'timestamp':'sample','source_candidates':[first,learned],'target_plane':first,
               'xyz_mm':np.tile([0.,0.,1000.],(4,1)),'pcd_shape':(2,2)}
        rt={'R':np.eye(3),'t':np.zeros(3),'ransac_inlier_frames':['sample'],
            'fitted_plane_selection':{'sample':{'support_signature':nc.support_signature(frame,learned)}}}
        with patch.object(nc,'plane_projection_metrics',return_value={'overlap':1.,'coverage':1.,'iou':1.}):
            for translation,used in (([0,0,0],['sample']),([1,-2,3],[])):
                rt.update(t=translation,ransac_inlier_frames=used)
                self.assertIs(nc.select_plane_for_rt(frame,rt),first)
                self.assertIs(frame['calibration_plane'],learned)
                self.assertEqual(nc.plane_status(frame,rt)['state'],'candidate')
                self.assertEqual(nc.plane_status(frame,rt)['training'],'used' if used else 'excluded')


if __name__=='__main__':unittest.main()
