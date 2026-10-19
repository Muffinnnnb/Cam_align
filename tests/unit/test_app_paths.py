"""Output-location regressions; dialogs are cancelled and camera devices are not opened."""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings, QSignalBlocker
from PySide6.QtWidgets import QApplication

import app_paths as paths
import calibration_gui as depth_gui
import nsl_color_gui as nsl_gui
from app_logging import LOG_PATH
from fuse_color_pointcloud import build_parser
from nsl_geometry import geometry_spec


class OutputPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.settings = QSettings(str(self.directory / "settings.ini"), QSettings.IniFormat)
        self.rt_path = self.directory / "rt.json"
        self.rt_path.write_text(json.dumps({
            "RT": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "source_geometry": geometry_spec("wf_range"),
        }), encoding="utf-8")

    def test_cli_and_log_defaults(self):
        args = build_parser().parse_args(["--rt", "example.json"])
        self.assertEqual(args.output, paths.OUTPUT_DIR / "fusion")
        self.assertEqual(LOG_PATH, paths.OUTPUT_DIR / "logs" / "cam_align.log")

    def test_legacy_lookup_does_not_redirect_explicit_custom_paths(self):
        default = paths.OUTPUT_DIR / "calibration" / "d_to_r_rt.json"
        with patch.object(Path, "exists", return_value=False), patch.object(Path, "is_file", return_value=True):
            self.assertEqual(paths.calibration_input(default), paths.APP_DIR / default.name)
            explicit = paths.OUTPUT_DIR / "validation" / default.name
            self.assertEqual(paths.calibration_input(explicit), explicit)
        with patch.object(Path, "exists", return_value=True):
            self.assertEqual(paths.calibration_input(default), default)

    def test_depth_gui_load_and_save_defaults(self):
        with patch.object(depth_gui.Controller, "reload"), \
             patch.object(depth_gui.Controller, "refresh_view"), \
             patch.object(depth_gui.Controller, "show_matrix"), \
             patch.object(depth_gui, "QSettings", return_value=self.settings), \
             patch.object(depth_gui, "calibration_input", return_value=self.rt_path):
            controller = depth_gui.Controller()
        self.addCleanup(controller.window.close)
        self.assertIsNotNone(controller.window)
        self.assertIsNotNone(controller.result)
        self.assertIsNotNone(controller.factory_result)
        self.assertEqual(Path(controller.settings.fileName()), self.directory / "settings.ini")
        self.assertEqual(controller.result_path, paths.OUTPUT_DIR / "calibration" / "d_to_r_rt.json")
        self.assertEqual(controller.calibration_file(True), paths.OUTPUT_DIR / "calibration" / "factory_d_to_r.json")
        with QSignalBlocker(controller.window.fileList):
            controller.window.fileList.addItem("path_test")
            controller.window.fileList.setCurrentRow(0)
        for method, name in ((controller.save_point_cloud, "depth"), (controller.save_ir_point_cloud, "target")):
            with patch.object(depth_gui.QFileDialog, "getSaveFileName", return_value=("", "")) as dialog:
                method()
                self.assertEqual(Path(dialog.call_args.args[2]), paths.OUTPUT_DIR / "pointclouds" / f"{name}_path_test.ply")
        with patch.object(depth_gui.QFileDialog, "getExistingDirectory", return_value="") as dialog:
            controller.save_alignment_comparison()
            self.assertEqual(Path(dialog.call_args.args[2]), paths.OUTPUT_DIR / "comparison")
        controller.window.close()

    def test_nsl_gui_load_and_save_defaults(self):
        self.settings.setValue("input/geometry", "wf_range")
        self.settings.setValue("rt/wf_range", str(self.rt_path))
        self.settings.sync()
        with patch.object(nsl_gui.Controller, "reload"):
            controller = nsl_gui.Controller(settings_path=self.directory / "settings.ini", auto_validate=False)
        self.addCleanup(controller.window.close)
        self.assertIsNotNone(controller.result)
        self.assertEqual(controller.geometry, 'wf_range')
        self.assertEqual(controller.result_path, Path(controller.settings.value('rt/wf_range',
                         str(paths.OUTPUT_DIR / "calibration" / "nsl_to_color_wf_rt.json"))))
        self.assertEqual(controller.point_plane_result_path(), paths.OUTPUT_DIR / 'calibration' / 'nsl_to_color_wf_range_auto_v8_rt.json')
        self.assertEqual(Path(controller.settings.fileName()), self.directory / "settings.ini")
        methods = ((controller.save_source, "pointclouds/nsl_wf_range_path_test.ply"),
                   (controller.save_fused, "pointclouds/nsl_color_wf_range_path_test.ply"),
                   (controller.save_projection_image, "images/nsl_color_projection_path_test.png"))
        with patch.object(controller, "selected_frame", return_value=("path_test", {})):
            for method, relative in methods:
                with patch.object(nsl_gui.QFileDialog, "getSaveFileName", return_value=("", "")) as dialog:
                    method()
                    self.assertEqual(Path(dialog.call_args.args[2]), paths.OUTPUT_DIR / relative)
        controller.window.close()


if __name__ == "__main__":
    unittest.main()
