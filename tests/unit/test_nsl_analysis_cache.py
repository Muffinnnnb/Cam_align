import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from nsl_analysis_cache import AnalysisCache


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        # The fixture files are fingerprinted, not interpreted by this test.
        for name in ('color', 'cloud', 'range', 'Intrinsic.txt', 'Distortion.txt'):
            (self.folder/name).write_bytes(name.encode())
        self.pair = {k: self.folder/k for k in ('color', 'cloud', 'range')}
        self.pair['timestamp'] = 'example'
        self.args = self.pair, self.folder, .1, 20., 'wf_range'
        plane = dict(mask=np.array([[True, False]]), normal=np.array([0., 0., -1.]))
        self.original = dict(pcd_shape=(1, 2), xyz_mm=np.array([[1.23456789, np.nan, 0.]]),
                             source_candidates=[plane], source_plane=plane, path=self.folder)
        self.analyzer = Mock(side_effect=lambda *args: __import__('copy').deepcopy(self.original))
        self.cache = AnalysisCache(self.folder/'cache')

    def test_memory_and_disk_are_lossless_and_isolated_from_rt_mutation(self):
        first = self.cache.analyze(*self.args, self.analyzer)
        first['source_candidates'][0]['mask'][:] = False
        first['aligned_depth'] = np.ones((1, 2))
        second = self.cache.analyze(*self.args, self.analyzer)
        self.assertTrue(second['source_plane']['mask'][0, 0])
        self.assertNotIn('aligned_depth', second)
        cold = AnalysisCache(self.cache.folder)
        third = cold.analyze(*self.args, self.analyzer)
        np.testing.assert_array_equal(third['xyz_mm'], self.original['xyz_mm'])
        self.assertIs(third['source_plane'], third['source_candidates'][0])
        self.assertEqual(third['pcd_shape'], (1, 2))
        self.analyzer.assert_called_once()

    def test_content_model_range_and_code_edits_invalidate(self):
        original = self.cache.key(*self.args)
        import os
        for name in ('cloud', 'color', 'range', 'Intrinsic.txt', 'Distortion.txt'):
            path = self.folder/name
            data, stat = path.read_bytes(), path.stat()
            path.write_bytes(b'x'*len(data))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(self.cache.key(*self.args), original, name)
            path.write_bytes(data)
        self.assertNotEqual(self.cache.key(*self.args[:-1], 'pcd'), original)
        self.assertNotEqual(self.cache.key(self.pair, self.folder, .2, 20., 'wf_range'), original)
        with patch('nsl_analysis_cache.ROOT', self.folder):
            before = self.cache.key(*self.args)
            (self.folder/'algorithm.py').write_text('changed')
            self.assertNotEqual(self.cache.key(*self.args), before)

    def test_corrupt_cache_is_recomputed_not_reported_as_detection_failure(self):
        self.cache.analyze(*self.args, self.analyzer)
        next(self.cache.folder.glob('*.npz')).write_bytes(b'not an npz')
        self.cache.memory.clear()
        self.cache.analyze(*self.args, self.analyzer)
        self.assertEqual(self.analyzer.call_count, 2)

    def test_disk_restores_exact_candidate_not_first_candidate_with_same_mask(self):
        from copy import deepcopy
        first = self.original['source_candidates'][0]
        second = dict(deepcopy(first), edge_support=.9)
        self.original['source_candidates'].append(second)
        self.original['source_plane'] = second
        self.cache.analyze(*self.args, self.analyzer)
        self.cache.memory.clear()
        restored = self.cache.analyze(*self.args, self.analyzer)
        self.assertIs(restored['source_plane'], restored['source_candidates'][1])

    def test_cache_failure_does_not_prevent_analysis(self):
        with patch.object(self.cache, 'write', side_effect=OSError('disk full')):
            frame = self.cache.analyze(*self.args, self.analyzer)
        np.testing.assert_array_equal(frame['xyz_mm'], self.original['xyz_mm'])

    def test_cache_is_bounded_and_does_not_remove_unrelated_files(self):
        self.cache.memory_items = 1
        self.cache.disk_bytes = 1
        self.cache.folder.mkdir()
        unrelated = self.cache.folder/'keep.txt'
        unrelated.write_text('preserve')
        self.cache.analyze(*self.args, self.analyzer)
        self.cache.analyze(self.pair, self.folder, .2, 20., 'wf_range', self.analyzer)
        self.assertEqual(len(self.cache.memory), 1)
        self.assertFalse(list(self.cache.folder.glob('*.npz')))
        self.assertEqual(unrelated.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
