"""Single-frame background work must finish, report errors and reject stale views."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nsl_color_gui import Controller, FrameWorker, CalibrationWorker


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
        worker = FrameWorker({'timestamp': 'sample'}, '.', .1, 20., 'pcd', calibration={'RT': []})
        errors, results, finished = [], [], []
        worker.failed.connect(lambda stamp, error: errors.append((stamp, error)))
        worker.succeeded.connect(lambda *args: results.append(args))
        worker.done.connect(lambda: finished.append(True))
        with patch('nsl_color_gui.analyze_pair', return_value={}), \
             patch('nsl_color_gui.select_plane_for_rt', side_effect=ValueError('bad RT')):
            worker.run()
        self.assertEqual(errors, [('sample', 'bad RT')])
        self.assertFalse(results)
        self.assertEqual(finished, [True])

    def test_batch_transfers_selected_frame_instead_of_discarding_it(self):
        for inspect_only in (True, False):
            frame = dict(timestamp='sample', source_plane={}, target_plane={})
            cache = Mock()
            cache.analyze.return_value = frame
            worker = CalibrationWorker([{'timestamp': 'sample'}], '.', .1, 20., 'rt.json', '.',
                calibration={'RT': []}, inspect_only=inspect_only, cache=cache, preview_stamp='sample')
            outputs = []
            worker.succeeded.connect(lambda *args: outputs.append(args))
            with patch('nsl_color_gui.calibrate_nsl_color', return_value={'RT': []}), \
                 patch('nsl_color_gui.select_plane_for_rt') as associate, \
                 patch('nsl_color_gui.plane_status', return_value={'state': 'candidate'}):
                worker.run()
            self.assertEqual(len(outputs), 1)
            self.assertIs(outputs[0][2], frame)
            cache.analyze.assert_called_once()
            self.assertEqual(associate.call_count, int(inspect_only))

    def test_loading_message_is_distinct_from_empty_list(self):
        controller = SimpleNamespace(current_depth=object(), current_stamp='sample', images={},
                                     window=Mock(), update_controls=Mock())
        Controller.clear_view(controller, '촬영쌍 로딩 중: sample')
        controller.window.pickedDepthLabel.setText.assert_called_with('촬영쌍 로딩 중: sample')
        self.assertIsNone(controller.current_stamp)
        Controller.clear_view(controller)
        controller.window.pickedDepthLabel.setText.assert_called_with('표시할 촬영쌍이 없습니다.')

    def test_failed_batch_is_not_silently_retried_or_reported_as_empty(self):
        controller = SimpleNamespace(set_busy=Mock(), window=Mock(), worker_thread=object(),
            worker=object(), batch_error='failed fit', statuses={}, frames={}, clear_view=Mock(),
            refresh_view=Mock(), update_controls=Mock())
        Controller.calibration_finished(controller)
        controller.refresh_view.assert_not_called()
        controller.clear_view.assert_called_once_with('처리 실패: failed fit')
        self.assertIsNone(controller.worker_thread)


if __name__ == '__main__':
    unittest.main()
