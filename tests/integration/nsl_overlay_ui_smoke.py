"""Real Qt + real capture test; display only, no calibration or source changes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import sys
import time
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT
from nsl_color_gui import Controller
from nsl_color_calibration import project_to_color, projection_view


def main():
    out = ROOT/'output/validation/point_overlay'
    out.mkdir(parents=True, exist_ok=True)
    rt = ROOT/'output/calibration/nsl_to_color_wf_range_auto_v10_rt.json'
    if not DEFAULT_CAPTURE_ROOT.is_dir() or not rt.is_file():
        raise SystemExit('Set CAM_ALIGN_CAPTURE_ROOT and provide the local v10 WF RT.')
    originals = [rt, ROOT/'nsl_color_settings.ini']
    hashes = {p: sha256(p.read_bytes()).hexdigest() for p in originals if p.exists()}
    settings = QSettings(str(out/'settings.ini'), QSettings.IniFormat)
    settings.clear()
    settings.setValue('input/capture_root', str(DEFAULT_CAPTURE_ROOT))
    settings.setValue('input/geometry', 'wf_range')
    settings.sync()
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf')
    app.setFont(QFont('Malgun Gothic', 9))
    c = Controller(rt, out/'settings.ini', auto_validate=False)
    c.window.show()

    def wait(controller):
        deadline = time.monotonic()+120
        while (controller.frame_thread is not None or controller.worker_thread is not None) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert controller.frame_thread is None and controller.worker_thread is None
        app.processEvents()
        assert controller.current_stamp

    def click_pixel(u, v):
        label = c.window.targetImageLabel
        rect, pixmap = label.contentsRect(), label.pixmap()
        h, w = c.current_depth.shape
        x = rect.x()+(rect.width()-pixmap.width())/2+(u+.5)*pixmap.width()/w
        y = rect.y()+(rect.height()-pixmap.height())/2+(v+.5)*pixmap.height()/h
        c.pick_depth(QPointF(x, y))

    wait(c)
    c.window.fileList.setCurrentRow(2)
    wait(c)
    frame = c.frames[c.current_stamp]
    depth = c.current_depth.copy()
    image, xyz = frame['color'].copy(), frame['xyz_mm'].copy()
    projection = [a.copy() for a in project_to_color(frame, c.result, c.color_cal)[:5]]
    rt_before = json.dumps(c.result, sort_keys=True)
    misses = c.analysis_cache.misses
    assert not c.window.projectionControls.isVisible()
    assert not c.window.pointSizeSpinBox.keyboardTracking()
    times = []
    for mode in (1, 2, 3, 0, 2):
        start = time.perf_counter()
        c.window.viewComboBox.setCurrentIndex(mode)
        app.processEvents()
        times.append(time.perf_counter()-start)
        assert c.window.projectionControls.isVisible() == (mode in (2, 3))
        assert c.frame_thread is None
    for size, shape in ((1, 0), (7, 0), (7, 1), (21, 1), (5, 1)):
        start = time.perf_counter()
        c.window.pointSizeSpinBox.setValue(size)
        c.window.pointShapeComboBox.setCurrentIndex(shape)
        app.processEvents()
        times.append(time.perf_counter()-start)
        expected, count = projection_view(frame, c.result, c.color_cal, **c.projection_style())
        np.testing.assert_array_equal(c.images[c.window.targetImageLabel], expected)
    # Typed even sizes normalize to odd; test zero-argument Qt signal handler too.
    c.window.pointSizeSpinBox.setValue(4)
    assert c.window.pointSizeSpinBox.value() == 5
    assert c.projection_style() == dict(point_size=5, point_shape='cross')
    app.processEvents()
    assert c.current_stamp == c.window.fileList.item(2).text()
    assert c.analysis_cache.misses == misses
    assert json.dumps(c.result, sort_keys=True) == rt_before
    np.testing.assert_array_equal(frame['color'], image)
    np.testing.assert_array_equal(frame['xyz_mm'], xyz)
    np.testing.assert_array_equal(c.current_depth, depth)
    for before, after in zip(projection, project_to_color(frame, c.result, c.color_cal)[:5]):
        np.testing.assert_array_equal(before, after)
    v, u = np.argwhere(depth > 0)[len(np.argwhere(depth > 0))//2]
    click_pixel(u, v)
    assert f'{depth[v, u]*1000:.1f} mm' in c.window.pickedDepthLabel.text()
    # A glyph arm with no measured center must not invent a depth sample.
    marked = np.any(c.images[c.window.targetImageLabel] != image, axis=2)
    empty = np.argwhere(marked & (depth == 0))
    assert len(empty)
    v, u = empty[len(empty)//2]
    click_pixel(u, v)
    assert 'point 없음' in c.window.pickedDepthLabel.text()
    png = out/'cross_5px.png'
    with patch('nsl_color_gui.QFileDialog.getSaveFileName', return_value=(str(png), 'PNG')):
        c.save_projection_image()
    np.testing.assert_array_equal(cv2.imread(str(png)), c.images[c.window.targetImageLabel])
    c.window.grab().save(str(out/'qt_overlay.png'))
    settings.sync()
    assert settings.value('view/point_shape') == 'cross'
    assert int(settings.value('view/point_size')) == 5
    c.window.close()
    restored = Controller(rt, out/'settings.ini', auto_validate=False)
    wait(restored)
    assert restored.projection_style() == dict(point_size=5, point_shape='cross')
    saved_result, restored.result = restored.result, None
    restored.update_controls()
    assert not restored.window.viewComboBox.model().item(2).isEnabled()
    assert not restored.window.projectionControls.isEnabled()
    restored.result = saved_result
    restored.window.close()
    assert all(sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(selected=c.current_stamp, projected_points=count,
                  display_refresh_seconds=times, checks='glyph controls, right image, PNG exact, '
                  'settings restore, no RT, sparse depth clicks, no analysis rerun, XYZ/RGB/RT unchanged')
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
