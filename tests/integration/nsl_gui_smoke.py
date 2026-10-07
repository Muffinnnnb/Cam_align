"""Opt-in GUI regression for the 76-frame 260922 dataset, not an accuracy test."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
import sys
import time
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, QTimer, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication
from app_paths import APP_DIR, output_dir
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT
from nsl_color_gui import Controller
from nsl_color_calibration import project_to_color


def main():
    rt_path=output_dir('calibration')/'nsl_to_color_wf_range_auto_v5_rt.json'
    if not DEFAULT_CAPTURE_ROOT.is_dir() or not rt_path.is_file():
        raise SystemExit('260922 촬영 폴더를 CAM_ALIGN_CAPTURE_ROOT로 지정하고 '
                         'output/calibration/nsl_to_color_wf_range_auto_v5_rt.json을 준비하세요.')
    out=output_dir('validation')/'gui_smoke'
    out.mkdir(parents=True,exist_ok=True)
    originals=list(output_dir('calibration').glob('*.json'))+list(output_dir('annotations').glob('*.json'))+[APP_DIR/'nsl_color_settings.ini']
    hashes={p:sha256(p.read_bytes()).hexdigest() for p in originals if p.is_file()}
    app=QApplication([])
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/malgun.ttf')
    app.setFont(QFont('Malgun Gothic',9))
    settings=QSettings(str(out/'settings.ini'),QSettings.IniFormat)
    settings.clear();settings.setValue('input/geometry','wf_range');settings.sync()
    settings.setValue('input/capture_root', str(DEFAULT_CAPTURE_ROOT));settings.sync()
    c=Controller(result_path=rt_path,settings_path=out/'settings.ini',auto_validate=False)
    c.window.show();app.processEvents()
    for name in ('manualBoardButton','clearBoardButton','pointPlaneRtButton','rgbCheckBox','projectionCheckBox'):
        assert c.window.findChild(QObject,name) is None,name
    assert not hasattr(c,'pick_manual_board')
    # Exercise the real threaded automatic path and its signal/status handling.
    c.point_plane_result_path=lambda:out/'automatic_rt.json'
    errors=[];ticks=[]
    timer=QTimer();timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start(100)
    c.run_calibration();c.worker.failed.connect(errors.append)
    deadline=time.monotonic()+360
    while c.worker_thread is not None and time.monotonic()<deadline:
        app.processEvents();time.sleep(.01)
    timer.stop()
    assert c.worker_thread is None,'worker timeout'
    assert not errors,errors
    assert len(ticks)>20,'GUI event loop was not responsive'
    used=set(c.result['ransac_inlier_frames'])
    history_used={n for n,s in c.statuses.items() if s['training']=='used'}
    assert history_used==used and len(used)>0
    assert all(s['state'] in ('candidate','failed') for s in c.statuses.values())
    assert sum(s['state']=='candidate' for s in c.statuses.values())==66
    assert len(c.statuses)==76
    assert c.window.inputGroup.isEnabled() and c.window.calibrateButton.isEnabled()
    stamp='20260922_093209_224'
    c.window.fileList.setCurrentItem(c.window.fileList.findItems(stamp,Qt.MatchExactly)[0])
    f=c.frame(stamp)
    assert f['rt_inlier']==(stamp in used)
    assert not f.get('manual_roi')
    assert 'RT 학습 여부와 무관' in c.window.sourceGroup.title()
    app.processEvents();c.window.grab().save(str(out/'automatic_planes.png'))
    points,rgb,_,u,v,stats,indices=project_to_color(f,c.result,c.color_cal,return_indices=True)
    np.testing.assert_array_equal(rgb,f['color'][v,u,::-1])
    c.window.viewComboBox.setCurrentIndex(1)
    np.testing.assert_array_equal(c.images[c.window.sourceImageLabel].reshape(-1,3)[indices],rgb[:,::-1])
    path=out/'rgb.ply'
    with patch('nsl_color_gui.QFileDialog.getSaveFileName',return_value=(str(path),'')):
        c.window.saveFusedButton.click()
    header,payload=path.read_bytes().split(b'end_header\n',1)
    assert b'property float intensity' not in header
    saved=np.frombuffer(payload,np.dtype([('xyz','<f4',3),('rgb','u1',3)]))
    np.testing.assert_array_equal(saved['xyz'],points)
    np.testing.assert_array_equal(saved['rgb'],rgb)
    app.processEvents();c.window.grab().save(str(out/'automatic_rgb.png'))
    label=c.window.targetImageLabel;pix=label.pixmap();rect=label.contentsRect()
    x=rect.x()+(rect.width()-pix.width())/2+(u[0]+.5)*pix.width()/f['color'].shape[1]
    y=rect.y()+(rect.height()-pix.height())/2+(v[0]+.5)*pix.height()/f['color'].shape[0]
    c.pick_depth(QPointF(x,y))
    assert f'{points[0,2]*1000:.1f} mm' in c.window.pickedDepthLabel.text()
    for name,filename in [('saveSourceButton','plane.ply'),('saveProjectionButton','projection.png')]:
        with patch('nsl_color_gui.QFileDialog.getSaveFileName',return_value=(str(out/filename),'')):
            getattr(c.window,name).trigger()
        assert (out/filename).is_file()
    c.window.intensityCheckBox.setChecked(True)
    with patch('nsl_color_gui.QFileDialog.getSaveFileName',return_value=(str(out/'intensity.ply'),'')):
        c.window.saveFusedButton.click()
    assert b'property float intensity' in (out/'intensity.ply').read_bytes().split(b'end_header\n',1)[0]
    for index in (2,3,0):
        c.window.viewComboBox.setCurrentIndex(index)
        assert c.current_depth is not None
    issue='20260922_094624_361'
    c.window.fileList.setCurrentItem(c.window.fileList.findItems(issue,Qt.MatchExactly)[0])
    assert c.statuses[issue]['state']=='candidate'
    assert c.statuses[issue]['training']=='excluded'
    assert c.window.fileList.currentItem().background().color().name()=='#dceeff'
    assert c.frame(issue)['rt_point_rms_m']>.020  # Do not hide the actual disagreement.
    app.processEvents();c.window.grab().save(str(out/'reported_frame.png'))
    # Menus still support explicitly loading an existing RT.
    with patch('nsl_color_gui.QFileDialog.getOpenFileName',return_value=(str(rt_path),'')):
        c.window.loadRtButton.trigger()
    assert c.result_path==rt_path
    c.window.geometryComboBox.setCurrentIndex(0)
    assert c.current_depth is None and not c.window.saveFusedButton.isEnabled()
    c.window.nfCompareCheckBox.setChecked(True)
    assert c.current_depth is not None and c.window.saveFusedButton.isEnabled()
    c.window.nfCompareCheckBox.setChecked(False)
    c.window.applyWfButton.click()
    assert c.geometry=='wf_range' and c.result_path==rt_path
    assert all(sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    report={'input':len(c.pairs),'used':len(used),'detection_candidates':66,'orange_detection_states':0,
            'training_history_matches_rt':True,'rgb_exact_points':len(saved),
            'event_loop_ticks':len(ticks),'no_manual_controls':True,'old_rt_and_annotations_preserved':True}
    (out/'report.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(report,flush=True)
    c.window.close()


if __name__=='__main__':main()
