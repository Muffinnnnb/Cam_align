"""Fast real-Qt cache handoff and loading/empty/failure regression after timing audit."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QSettings
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from nsl_color_gui import Controller
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT


def main():
    out = ROOT/'output/validation/cache_ui'
    out.mkdir(parents=True, exist_ok=True)
    settings = QSettings(str(out/'settings.ini'), QSettings.IniFormat)
    settings.setValue('input/capture_root', str(DEFAULT_CAPTURE_ROOT))
    settings.setValue('input/geometry', 'wf_range')
    settings.sync()
    app = QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf')
    app.setFont(QFont('Malgun Gothic', 9))
    c = Controller(ROOT/'output/calibration/nsl_to_color_wf_range_auto_v10_rt.json',
                   out/'settings.ini', auto_validate=False)
    c.window.show()
    def wait():
        deadline = time.monotonic()+120
        while (c.frame_thread is not None or c.worker_thread is not None) and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.005)
        assert c.frame_thread is None and c.worker_thread is None
        app.processEvents()
    assert '로딩 중' in c.window.pickedDepthLabel.text()
    wait()
    stamp = c.current_stamp
    c.pairs = [c.pairs[i] for i in np.linspace(0, len(c.pairs)-1, 5).astype(int)]
    misses = c.analysis_cache.misses
    c.validate_all()
    wait()
    assert c.current_stamp == stamp and c.analysis_cache.misses == misses
    assert all(c.statuses[p['timestamp']]['state'] == 'candidate' for p in c.pairs)
    assert '촬영쌍이 없습니다' not in c.window.pickedDepthLabel.text()
    c.window.grab().save(str(out/'completed.png'))
    # A broken input is a failure, never an empty list or automatic retry loop.
    c.frames.clear()
    c.pair_by_stamp[stamp] = dict(c.pair_by_stamp[stamp], cloud=out/'missing.pcd')
    c.show_frame(stamp)
    wait()
    assert '실패' in c.window.pickedDepthLabel.text()
    assert c.requested_stamp is None
    c.capture_root = out/'empty'
    c.reload()
    assert c.window.fileList.count() == 0
    assert c.window.pickedDepthLabel.text() == '표시할 촬영쌍이 없습니다.'
    c.window.close()
    print('PASS: warm handoff, latest completed view, loading, failure without retry, real empty list')


if __name__ == '__main__':
    main()
