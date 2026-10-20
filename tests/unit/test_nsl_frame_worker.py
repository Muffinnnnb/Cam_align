"""Single-frame background work must finish, report errors and reject stale views."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nsl_color_gui import Controller, FrameWorker


class FrameWorkerTests(unittest.TestCase):
    def test_read_failure_reports_and_finishes(self):
        worker = FrameWorker({'timestamp': 'sample'}, '.', .1, 20., 'pcd')
        errors, finished = [], []
        worker.failed.connect(lambda stamp, error: errors.append((stamp, error)))
        worker.done.connect(lambda: finished.append(True))
        with patch('nsl_color_gui.analyze_pair', side_effect=ValueError('bad input')):
            worker.run()
        self.assertEqual(errors, [('sample', 'bad input')])
        self.assertEqual(finished, [True])

    def test_stale_result_is_cached_but_never_replaces_latest_selection(self):
        controller = SimpleNamespace(result=None, frames={}, requested_stamp='latest',
                                     show_frame=Mock(), frame_failed=Mock())
        data = {'original': True}
        Controller.frame_ready(controller, 'older', data)
        self.assertIs(controller.frames['older'], data)
        controller.show_frame.assert_not_called()
        Controller.frame_ready(controller, 'latest', data)
        controller.show_frame.assert_called_once_with('latest')

    def test_association_failure_is_reported_not_retried_in_a_loop(self):
        controller = SimpleNamespace(result={'RT': []}, color_cal='.', frames={},
                                     requested_stamp='sample', show_frame=Mock(), frame_failed=Mock())
        with patch('nsl_color_gui.select_plane_for_rt', side_effect=ValueError('bad RT')):
            Controller.frame_ready(controller, 'sample', {})
        controller.frame_failed.assert_called_once_with('sample', 'bad RT')
        self.assertFalse(controller.frames)
        controller.show_frame.assert_not_called()


if __name__ == '__main__':
    unittest.main()
