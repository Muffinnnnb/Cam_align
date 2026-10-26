"""Opt-in real-data workflow switch with isolated settings and read-only RTs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
from hashlib import sha256
from time import monotonic, sleep
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from app import Workspace
import calibration_gui as depth_gui
import nsl_color_gui as nsl_gui
from calibrate_d_to_r import timestamps
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT
from nsl_color_calibration import project_to_color


def main():
    qt = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf')
    qt.setFont(QFont('Malgun Gothic', 9))
    out = ROOT/'output/validation/cleanup_20261026'
    out.mkdir(parents=True, exist_ok=True)
    rt = ROOT/'output/calibration/nsl_to_color_wf_range_auto_v10_rt.json'
    paths = list((ROOT/'output/calibration').glob('*.json'))+list(ROOT.glob('*settings.ini'))
    original = {p: sha256(p.read_bytes()).hexdigest() for p in paths}
    ns = QSettings(str(out/'nsl.ini'), QSettings.IniFormat)
    ns.setValue('input/capture_root', str(DEFAULT_CAPTURE_ROOT))
    ns.setValue('input/geometry', 'wf_range')
    ns.sync()
    stamps = timestamps()[:2]
    assert len(stamps) == 2

    def factory(mode):
        if mode == 'pointcloud':
            return nsl_gui.Controller(rt, out/'nsl.ini', auto_validate=False)
        with patch.object(depth_gui, 'timestamps', return_value=stamps):
            return depth_gui.Controller(settings_path=out/'depth.ini')

    workspace = Workspace(settings_path=out/'workspace.ini', factory=factory, auto_open=False)
    workspace.window.show()
    workspace.switch_mode(0)
    qt.processEvents()
    depth = workspace.controllers['images']
    assert depth.window.fileList.count() == 2
    assert depth.current_stamp in stamps
    assert not depth.window.openNslColorButton.isVisible()
    old_depth = depth.aligned_depth.copy() if depth.aligned_depth is not None else None
    selection = depth.current_stamp
    workspace.window.grab().save(str(out/'workspace_depth.png'))
    # Real offline rs.align() and Qt background save (no camera acquisition).
    with patch.object(depth_gui.QFileDialog, 'getExistingDirectory', return_value=str(out/'comparison')):
        depth.save_alignment_comparison()
    assert depth.save_thread is not None
    workspace.window.workflowCombo.setCurrentIndex(1)
    assert workspace.active == 'images'
    deadline = monotonic()+120
    while depth.save_thread is not None and monotonic() < deadline:
        qt.processEvents(); sleep(.005)
    assert depth.save_thread is None, 'Comparison save timed out'
    assert '저장 완료' in depth.window.comparisonStatus.text(), depth.window.comparisonStatus.text()
    assert len(list((out/'comparison').glob('*.ply'))) == 3
    workspace.window.workflowCombo.setCurrentIndex(1)
    nsl = workspace.controllers['pointcloud']
    # A live selected-frame worker must prevent switching and closing.
    if nsl.is_busy():
        workspace.window.workflowCombo.setCurrentIndex(0)
        assert workspace.active == 'pointcloud'
        assert not workspace.window.close()
    deadline = monotonic()+120
    while nsl.is_busy() and monotonic() < deadline:
        qt.processEvents(); sleep(.005)
    assert not nsl.is_busy() and nsl.current_stamp
    nsl.window.viewComboBox.setCurrentIndex(2)
    nsl.window.pointShapeComboBox.setCurrentIndex(1)
    nsl.window.pointSizeSpinBox.setValue(5)
    qt.processEvents()
    image = nsl.images[nsl.window.targetImageLabel].copy()
    cloud_selection = nsl.current_stamp
    workspace.window.grab().save(str(out/'workspace_pointcloud.png'))
    for index in (0, 1, 0, 1):
        workspace.window.workflowCombo.setCurrentIndex(index)
        qt.processEvents()
    assert depth.current_stamp == selection
    assert nsl.current_stamp == cloud_selection
    assert len(workspace.controllers) == 2
    assert not workspace.busy()
    np.testing.assert_array_equal(image, nsl.images[nsl.window.targetImageLabel])
    ply = out/'workspace_rgb.ply'
    nsl.window.intensityCheckBox.setChecked(False)
    with patch.object(nsl_gui.QFileDialog, 'getSaveFileName', return_value=(str(ply), 'PLY')):
        nsl.save_fused()
    _, payload = ply.read_bytes().split(b'end_header\n', 1)
    saved = np.frombuffer(payload, np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3)]))
    points, rgb = project_to_color(nsl.frames[nsl.current_stamp], nsl.result, nsl.color_cal)[:2]
    np.testing.assert_array_equal(saved['xyz'], points.astype('float32'))
    np.testing.assert_array_equal(saved['rgb'], rgb)
    if old_depth is not None:
        np.testing.assert_array_equal(old_depth, depth.aligned_depth)
    assert workspace.window.close()
    qt.processEvents()
    assert all(sha256(p.read_bytes()).hexdigest() == h for p, h in original.items())
    print('PASS: real Depth/NSL panels, lazy loading, selection/RT/display preserved, '
          'busy switch/close rejected, offline rs.align + 3 PLY save, RGB PLY exact, '
          'clean close, original settings/RT files unchanged')


if __name__ == '__main__':
    main()
