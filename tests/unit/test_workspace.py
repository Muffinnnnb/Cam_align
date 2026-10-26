"""Workflow selection and lifecycle without captures, hardware or shared settings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
import app
import calibration_gui as depth_gui


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'workspace.ini'
        self.created = []

        def factory(mode):
            window = QMainWindow()
            window.openNslColorButton = QPushButton(window)
            c = SimpleNamespace(window=window, working=False, selection='kept', result=object())
            c.is_busy = lambda: c.working
            self.created.append((mode, c))
            return c

        self.factory = factory
        self.workspace = app.Workspace(settings_path=self.path, factory=factory, auto_open=False)
        self.addCleanup(self.workspace.window.close)

    def test_lazy_switch_preserves_instances_results_and_selection(self):
        w = self.workspace
        self.assertFalse(self.created)
        w.switch_mode(0)
        first = w.controllers['images']
        result = first.result
        self.assertTrue(first.window.openNslColorButton.isHidden())
        w.window.workflowCombo.setCurrentIndex(1)
        self.assertEqual(w.active, 'pointcloud')
        w.window.workflowCombo.setCurrentIndex(0)
        self.assertIs(w.controllers['images'], first)
        self.assertIs(first.result, result)
        self.assertEqual(first.selection, 'kept')
        self.assertEqual(len(self.created), 2)

    def test_busy_switch_and_close_are_rejected(self):
        w = self.workspace
        w.switch_mode(0)
        w.controllers['images'].working = True
        w.window.show()
        w.window.workflowCombo.setCurrentIndex(1)
        self.assertEqual(w.active, 'images')
        self.assertEqual(w.window.workflowCombo.currentIndex(), 0)
        self.assertFalse(w.window.close())
        self.assertEqual(len(self.created), 1)
        w.controllers['images'].working = False
        self.assertTrue(w.window.close())

    def test_failed_new_environment_does_not_discard_current(self):
        w = self.workspace
        w.switch_mode(0)
        first = w.controllers['images'].window
        with patch.object(w, 'factory', side_effect=ValueError('invalid settings')):
            w.window.workflowCombo.setCurrentIndex(1)
        self.assertEqual(w.active, 'images')
        self.assertIs(w.window.workspaceStack.currentWidget(), first)
        self.assertTrue(w.window.workflowCombo.isEnabled())
        self.assertIn('실패', w.window.statusbar.currentMessage())
        self.assertFalse(w.opening)

    def test_mode_persistence_and_invalid_setting_fallback(self):
        self.workspace.switch_mode(1)
        restored = app.Workspace(settings_path=self.path, factory=self.factory, auto_open=False)
        self.addCleanup(restored.window.close)
        self.assertEqual(restored.window.workflowCombo.currentIndex(), 1)
        restored.settings.setValue('workflow', 'obsolete')
        restored.settings.sync()
        other = app.Workspace(settings_path=self.path, factory=self.factory, auto_open=False)
        self.addCleanup(other.window.close)
        self.assertEqual(other.window.workflowCombo.currentIndex(), 0)

    def test_depth_folder_restore_and_busy_release_on_reload_error(self):
        path = Path(self.folder.name)/'depth.ini'
        from PySide6.QtCore import QSettings
        settings = QSettings(str(path), QSettings.IniFormat)
        settings.setValue('input/depth_dir', str(Path(self.folder.name)/'depth'))
        settings.setValue('input/target_dir', str(Path(self.folder.name)/'color'))
        settings.sync()
        with patch.object(depth_gui.Controller, 'reload'), patch.object(depth_gui.Controller, 'load_calibrations'):
            c = depth_gui.Controller(settings_path=path)
        self.addCleanup(c.window.close)
        self.assertEqual(c.depth_dir.name, 'depth')
        self.assertEqual(c.target_dir.name, 'color')
        with patch.object(c, '_reload', side_effect=ValueError('missing input')):
            c.reload()
        self.assertFalse(c.loading)
        self.assertFalse(c.is_busy())
        self.assertFalse(c.window.fileList.signalsBlocked())
        self.assertIn('실패', c.window.statusbar.currentMessage())

    def test_legacy_entrypoints_share_the_workspace(self):
        import nsl_color_gui
        with patch.object(app, 'main') as launch:
            depth_gui.main()
            nsl_color_gui.main()
        self.assertEqual([c.args for c in launch.call_args_list], [('images',), ('pointcloud',)])

    def test_close_before_startup_timer_does_not_launch_analysis(self):
        w = app.Workspace(settings_path=self.path, factory=self.factory)
        w.window.close()
        self.qt.processEvents()
        self.assertFalse(self.created)


if __name__ == '__main__':
    unittest.main()
