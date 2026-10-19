"""Range proposals must add measured board support, not relabel partials."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import nsl_color_calibration as nc
from nsl_board_identity import identify_board
from nsl_surface_support import range_support, paired_support
from nsl_plane_fit import support_quality
from test_nsl_board_identity import scene


class SurfaceSupportTests(unittest.TestCase):
    @staticmethod
    def board():
        x,y=np.meshgrid(np.linspace(-.3,.3,120),np.linspace(-.3,.3,120))
        selected=(abs(x)<.125)&(abs(y)<.14)
        rng=np.random.default_rng(8)
        z=np.where(selected,1.,1.4)+rng.normal(0,.005,x.shape)
        return np.stack([x,y,z],-1),selected

    def test_range_region_recovers_measured_board_without_rgb_or_rt(self):
        grid,board=self.board();before=grid.copy()
        candidates=range_support(grid,nc.plane_quality)
        complete=[p for p in candidates if support_quality(grid[p['mask']],p)['eligible']]
        self.assertTrue(complete)
        best=max(complete,key=lambda p:np.count_nonzero(p['mask']&board))
        self.assertGreater(np.count_nonzero(best['mask']&board)/board.sum(),.80)
        self.assertLess(np.count_nonzero(best['mask']&~board)/best['mask'].sum(),.01)
        np.testing.assert_array_equal(grid,before)

    def test_missing_samples_are_never_filled(self):
        grid,board=self.board();grid[50:65,50:65]=np.nan
        for p in range_support(grid,nc.plane_quality):
            self.assertFalse(p['mask'][50:65,50:65].any())
            self.assertTrue(np.isfinite(grid[p['mask']]).all())

    def test_paired_support_refits_actual_full_board_without_chaining(self):
        grid,board=self.board()
        left=board&(grid[...,0]<-.003);right=board&(grid[...,0]>.003)
        seeds=[dict(mask=mask,normal=np.array([0.,0.,-1.])) for mask in (left,right)]
        old=[p['mask'].copy() for p in seeds]
        result=paired_support(grid,seeds,nc.plane_quality)
        self.assertTrue(result)
        self.assertTrue(support_quality(grid[result[0]['mask']],result[0])['eligible'])
        self.assertGreater(result[0]['mask'].sum(),max(left.sum(),right.sum()))
        for p,a in zip(seeds,old):np.testing.assert_array_equal(p['mask'],a)
        # Adjacent but offset depth layer cannot be pulled into this board.
        grid[right,2]+=.080
        self.assertFalse(paired_support(grid,seeds,nc.plane_quality))

    def test_invalid_grid_has_no_hypothesis(self):
        self.assertEqual(range_support(np.full((30,30,3),np.nan),nc.plane_quality),[])

    def test_complete_same_object_can_outrank_a_quieter_partial(self):
        f,rt,K,D=scene()
        full=f['source_candidates'][1]
        mask=full['mask'].copy();mask[20:25]=False
        part=dict(mask=mask,support_quality={'eligible':False})
        f['source_candidates']=[part,full]
        def evidence(points,target,model):
            return dict(plausible=True,score=.10 if len(points)<400 else .12,coverage=.7,
                        overlap=1.,near_fraction=1.,finite_board_rms_m=.01)
        with patch('nsl_board_identity.candidate_evidence',side_effect=evidence):
            result=identify_board(f,rt,K,D)
        self.assertEqual(result['state'],'matched')
        self.assertIs(result['plane'],full)
        self.assertFalse(part['support_quality']['eligible'])

    def test_partial_only_is_never_promoted(self):
        f,rt,K,D=scene()
        f['source_candidates']=[f['source_candidates'][1]]
        f['source_candidates'][0]['support_quality']['eligible']=False
        self.assertEqual(identify_board(f,rt,K,D)['state'],'partial')


if __name__=='__main__':unittest.main()
