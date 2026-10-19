from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog

from app_logging import get_log_path, get_logger, install_exception_hook
from app_paths import calibration_input, output_dir
from fuse_color_pointcloud import DEFAULT_CAPTURE_ROOT, DEFAULT_COLOR_CAL, capture_pairs, load_camera, load_rt
from nsl_geometry import GEOMETRY_MODES, GEOMETRY_LABELS, rt_geometry, fusion_block_reason
from nsl_color_calibration import (
    RESULT_PATH, aligned_depth, analyze_pair, calibrate_nsl_color,
    color_plane_view, pointcloud_view, projection_view, save_fused_cloud,
    save_projection, save_source_cloud, select_plane_for_rt,
    fused_color_view, plane_status, board_error_view, displayed_board,
    PLANE_CANDIDATE_RGB,
)


LOGGER = get_logger("nsl_color_gui")
UI_PATH = Path(__file__).with_name("nsl_color_viewer.ui")
SETTINGS_PATH = Path(__file__).with_name("nsl_color_settings.ini")


class CalibrationWorker(QObject):
    progress = Signal(int, int, str, bool, bool)
    stage = Signal(str)
    status = Signal(str, object)
    succeeded = Signal(object, object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, pairs, color_cal, z_min, z_max, result_path, capture_root,
                 calibration=None, inspect_only=False, geometry='pcd'):
        super().__init__()
        self.pairs = pairs
        self.color_cal = Path(color_cal)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.result_path = Path(result_path)
        self.capture_root = Path(capture_root)
        self.calibration = calibration
        self.inspect_only = inspect_only
        self.geometry = geometry

    def run(self):
        try:
            frames, statuses = {}, {}
            total = len(self.pairs)
            for index, pair in enumerate(self.pairs, 1):
                try:
                    frame = analyze_pair(
                        pair, self.color_cal, self.z_min, self.z_max, self.geometry
                    )
                    if not self.inspect_only:
                        frames[pair["timestamp"]] = frame
                    source_ok = frame['source_plane'] is not None
                    target_ok = frame["target_plane"] is not None
                    if self.inspect_only and self.calibration:
                        select_plane_for_rt(frame, self.calibration, self.color_cal)
                    statuses[pair["timestamp"]] = plane_status(frame, self.calibration if self.inspect_only else None)
                except Exception:
                    LOGGER.exception(
                        "bulk analysis failed stamp=%s", pair["timestamp"]
                    )
                    source_ok = target_ok = False
                    statuses[pair["timestamp"]] = {"state": "failed", "message": "파일 읽기/분석 실패 (로그 참조)"}
                if self.inspect_only:
                    self.status.emit(pair["timestamp"], statuses[pair["timestamp"]])
                self.progress.emit(
                    index, total, pair["timestamp"], source_ok, target_ok
                )

            result = self.calibration
            if not self.inspect_only:
                result = calibrate_nsl_color(
                    frames, self.result_path, self.capture_root, self.color_cal,
                    progress=self.stage.emit,
                )
                for stamp, frame in frames.items():
                    select_plane_for_rt(frame, result, self.color_cal)
                    statuses[stamp] = plane_status(frame, result)
            self.succeeded.emit(result, statuses)
        except Exception as exc:
            LOGGER.exception("NSL->Color calibration worker failed")
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class Controller(QObject):
    def __init__(self, result_path=None, settings_path=SETTINGS_PATH, auto_validate=True):
        super().__init__()
        self.window = QUiLoader().load(str(UI_PATH))
        if self.window is None:
            raise RuntimeError(f"UI load failed: {UI_PATH}")
        for name in ('validateButton', 'loadRtButton', 'saveSourceButton', 'saveProjectionButton', 'intensityCheckBox'):
            setattr(self.window, name, self.window.findChild(QAction, name))

        self.settings = QSettings(str(settings_path), QSettings.IniFormat)
        self.auto_validate = auto_validate
        self.capture_root = Path(
            self.settings.value("input/capture_root", str(DEFAULT_CAPTURE_ROOT))
        )
        self.color_cal = Path(
            self.settings.value("input/color_cal", str(DEFAULT_COLOR_CAL))
        )
        self.z_min = float(self.settings.value("input/z_min", .10))
        self.z_max = float(self.settings.value("input/z_max", 20.0))
        self.geometry = self.settings.value('input/geometry', 'pcd')
        if self.geometry not in GEOMETRY_MODES:
            self.geometry = 'pcd'
        self.result_path = Path(result_path) if result_path else self.geometry_result_path()
        self.result = None
        try:
            read_path = calibration_input(self.result_path)
            if read_path.exists():
                self.result = load_rt(read_path)[2]
                if rt_geometry(self.result) != self.geometry:
                    self.result = None
        except Exception:
            LOGGER.exception("RT load failed path=%s", self.result_path)

        self.pairs = []
        self.pair_by_stamp = {}
        self.frames = {}
        self.statuses = {}
        self.images = {}
        self.current_stamp = None
        self.current_depth = None
        self.worker_thread = None
        self.worker = None

        self.window.captureRootEdit.setText(str(self.capture_root))
        self.window.colorCalEdit.setText(str(self.color_cal))
        self.window.fileLabel.setText(
            "초록=보드 대응 / 파랑=부분 대응 / 회색=미확인·보류\n빨강=검출 없음 / 대응 확인 ≠ 정합 정확도 합격"
        )
        self.window.zMinSpinBox.setValue(self.z_min)
        self.window.zMaxSpinBox.setValue(self.z_max)
        self.window.geometryComboBox.addItems(GEOMETRY_LABELS)
        self.window.geometryComboBox.setCurrentIndex(GEOMETRY_MODES.index(self.geometry))

        for label in (self.window.sourceImageLabel, self.window.targetImageLabel):
            label.installEventFilter(self)
        self.window.installEventFilter(self)
        self.window.fileList.currentItemChanged.connect(self.show_item)
        self.window.captureRootButton.clicked.connect(self.choose_capture_root)
        self.window.colorCalButton.clicked.connect(self.choose_color_cal)
        self.window.reloadButton.clicked.connect(self.reload)
        self.window.applyButton.clicked.connect(self.apply_settings)
        self.window.geometryComboBox.currentIndexChanged.connect(lambda _: self.apply_settings())
        self.window.applyWfButton.clicked.connect(lambda: self.window.geometryComboBox.setCurrentIndex(1))
        self.window.nfCompareCheckBox.toggled.connect(self.nf_comparison_changed)
        self.window.calibrateButton.clicked.connect(self.run_calibration)
        self.window.validateButton.triggered.connect(self.validate_all)
        self.window.loadRtButton.triggered.connect(lambda: self.choose_rt())
        self.window.viewComboBox.currentIndexChanged.connect(self.refresh_view)
        self.window.saveSourceButton.triggered.connect(self.save_source)
        self.window.saveFusedButton.clicked.connect(self.save_fused)
        self.window.saveProjectionButton.triggered.connect(self.save_projection_image)

        if self.result:
            self.show_matrix()
        self.reload()
        LOGGER.info(
            "NSL GUI ready capture=%s color_cal=%s geometry=%s rt=%s",
            self.capture_root, self.color_cal, self.geometry, self.result_path if self.result else None,
        )

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Close and obj == self.window and self.worker_thread is not None:
                event.ignore()
                self.window.statusbar.showMessage("실행 중인 검사/계산 완료 후 창을 닫아주세요.")
                return True
            if event.type() == QEvent.Resize and obj in self.images:
                self.set_image(obj, self.images[obj])
            elif (
                event.type() == QEvent.MouseButtonPress
                and obj == self.window.targetImageLabel
            ):
                self.pick_depth(event.position())
                return True
            return super().eventFilter(obj, event)
        except RuntimeError:
            return False

    def choose_capture_root(self):
        path = QFileDialog.getExistingDirectory(
            self.window, "Capture Root 선택", str(self.capture_root)
        )
        if path:
            self.apply_settings(capture_root=Path(path))

    def choose_color_cal(self):
        path = QFileDialog.getExistingDirectory(
            self.window, "Color 내부 파라미터 폴더 선택", str(self.color_cal)
        )
        if not path:
            return
        folder = Path(path)
        missing = [
            name for name in ("Intrinsic.txt", "Distortion.txt")
            if not (folder / name).exists()
        ]
        if missing:
            self.window.statusbar.showMessage(
                "필요한 파일이 없습니다: " + ", ".join(missing)
            )
            return
        self.apply_settings(color_cal=folder)

    def choose_rt(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self.window, "NSL → Color RT 선택", str(self.result_path), "RT JSON (*.json)")
        if not path:
            return
        try:
            result = load_rt(Path(path))[2]
        except Exception as exc:
            LOGGER.exception("RT selection failed path=%s", path)
            self.window.statusbar.showMessage(f"RT 읽기 실패: {exc}")
            return
        self.result, self.result_path = result, Path(path)
        self.geometry = rt_geometry(result)
        self.window.nfCompareCheckBox.blockSignals(True)
        self.window.nfCompareCheckBox.setChecked(False)
        self.window.nfCompareCheckBox.blockSignals(False)
        self.window.geometryComboBox.blockSignals(True)
        self.window.geometryComboBox.setCurrentIndex(GEOMETRY_MODES.index(self.geometry))
        self.window.geometryComboBox.blockSignals(False)
        self.settings.setValue('input/geometry', self.geometry)
        self.settings.setValue(f'rt/{self.geometry}', str(self.result_path))
        self.settings.sync()
        self.reload()
        self.show_matrix()
        self.window.calibrationStatus.setText('비교용 RT 적용 · 전체 장면 정합 미검증' if result.get('calibration_version',0) >= 4 else '선택한 RT 적용')

    def geometry_result_path(self):
        name = 'nsl_to_color_rt.json' if self.geometry == 'pcd' else 'nsl_to_color_wf_rt.json'
        return Path(self.settings.value(f'rt/{self.geometry}', str(output_dir('calibration') / name)))

    def point_plane_result_path(self):
        return output_dir('calibration') / f'nsl_to_color_{self.geometry}_auto_v8_rt.json'

    def read_range(self):
        z_min = self.window.zMinSpinBox.value()
        z_max = self.window.zMaxSpinBox.value()
        if z_min >= z_max:
            raise ValueError("z_min은 z_max보다 작아야 합니다.")
        return float(z_min), float(z_max)

    def apply_settings(self, *, capture_root=None, color_cal=None):
        try:
            z_min, z_max = self.read_range()
            folder = color_cal if color_cal is not None else self.color_cal
            root = capture_root if capture_root is not None else self.capture_root
            load_camera(folder)
            capture_pairs(root)
        except Exception as exc:
            self.window.statusbar.showMessage(str(exc))
            return
        self.z_min, self.z_max = z_min, z_max
        geometry = GEOMETRY_MODES[self.window.geometryComboBox.currentIndex()]
        if geometry != self.geometry:
            LOGGER.info('NSL geometry changed %s -> %s; invalidate RT/cache', self.geometry, geometry)
            self.geometry = geometry
            self.window.nfCompareCheckBox.blockSignals(True)
            self.window.nfCompareCheckBox.setChecked(False)
            self.window.nfCompareCheckBox.blockSignals(False)
            self.result_path = self.geometry_result_path()
            self.result = None
            if self.result_path.is_file():
                try:
                    result = load_rt(self.result_path)[2]
                    if rt_geometry(result) == geometry:
                        self.result = result
                except Exception:
                    LOGGER.exception('Geometry-specific RT load failed')
            self.show_matrix()
            self.window.calibrationStatus.setText('점군 모델 변경: 해당 모델의 RT만 사용합니다. RT가 없으면 계산하세요.')
        self.capture_root, self.color_cal = root, folder
        self.window.captureRootEdit.setText(str(root))
        self.window.colorCalEdit.setText(str(folder))
        self.settings.setValue("input/capture_root", str(self.capture_root))
        self.settings.setValue("input/color_cal", str(self.color_cal))
        self.settings.setValue("input/z_min", self.z_min)
        self.settings.setValue("input/z_max", self.z_max)
        self.settings.setValue('input/geometry', self.geometry)
        self.settings.sync()
        self.frames.clear()
        self.reload()

    def reload(self):
        current = (
            self.window.fileList.currentItem().text()
            if self.window.fileList.currentItem() else ""
        )
        try:
            self.pairs = capture_pairs(self.capture_root)
            if self.result:
                same_root = Path(self.result.get("capture_root", self.capture_root)).resolve() == self.capture_root.resolve()
                k, dist = load_camera(self.color_cal)
                same_camera = (
                    np.array_equal(k, self.result.get("target_intrinsics", k))
                    and np.array_equal(dist, self.result.get("target_distortion", dist))
                )
                if "target_intrinsics" not in self.result and "target_intrinsic_source" in self.result:
                    same_camera &= Path(self.result["target_intrinsic_source"]).parent.resolve() == self.color_cal.resolve()
                if not same_camera:
                    LOGGER.warning("Previous RT deactivated: Color intrinsics changed")
                    self.result = None
                    self.show_matrix()
                    self.window.calibrationStatus.setText("Color 내부 파라미터 변경: 일치하는 RT를 불러오세요.")
                elif not same_root:
                    LOGGER.info("Reusing RT for another capture folder; training membership will not be reused")
        except Exception as exc:
            LOGGER.exception("capture pair loading failed")
            self.window.statusbar.showMessage(str(exc))
            return

        self.frames.clear()
        self.current_depth = None
        self.current_stamp = None
        self.statuses.clear()
        self.pair_by_stamp = {p["timestamp"]: p for p in self.pairs}
        self.window.fileList.blockSignals(True)
        self.window.fileList.clear()
        for pair in self.pairs:
            stamp = pair["timestamp"]
            self.window.fileList.addItem(stamp)
            self.apply_status(stamp, {"state": "pending", "message": "파일 쌍 연결 완료 / 현재 평면 검사 대기"})
        self.window.fileList.blockSignals(False)
        matches = self.window.fileList.findItems(current, Qt.MatchExactly)
        if matches:
            self.window.fileList.setCurrentItem(matches[0])
        elif self.window.fileList.count():
            self.window.fileList.setCurrentRow(0)
        else:
            self.clear_view()
        self.window.statusbar.showMessage(
            f"manifest 기준 Color + NSL PCD 매칭 {len(self.pairs)}개"
        )
        self.update_controls()
        if self.auto_validate and self.pairs:
            QTimer.singleShot(0, self.validate_all)

    def frame(self, stamp):
        frame = self.frames.get(stamp)
        if frame is None:
            pair = self.pair_by_stamp[stamp]
            frame = analyze_pair(pair, self.color_cal, self.z_min, self.z_max, self.geometry)
            if self.result:
                select_plane_for_rt(frame, self.result, self.color_cal)
            if len(self.frames) >= 12:
                self.frames.pop(next(iter(self.frames)))
            self.frames[stamp] = frame
        frame['allow_nf_comparison'] = self.window.nfCompareCheckBox.isChecked()
        return frame

    def apply_status(self, stamp, status):
        self.statuses[stamp] = status
        items = self.window.fileList.findItems(stamp, Qt.MatchExactly)
        if not items:
            return
        background, foreground = {
            "failed": ("#ffd6d6", "#8b0000"), "pending": ("#eeeeee", "#555555"),
            "candidate": ("#e0f4df", "#166534"),
            "partial": ("#dceefa", "#005a9c"), "unconfirmed": ("#eeeeee", "#555555"),
        }[status["state"]]
        items[0].setBackground(QColor(background))
        items[0].setForeground(QColor(foreground))
        items[0].setToolTip(status["message"] + "\n현재 RT는 정답이 아닙니다. 보드 대응과 RT 학습 사용/제외는 별개입니다.")

    def mark_item(self, stamp, frame):
        self.apply_status(stamp, plane_status(frame, self.result))

    def show_item(self, item, previous):
        if item:
            self.show_frame(item.text())

    def refresh_view(self):
        item = self.window.fileList.currentItem()
        if item:
            self.show_frame(item.text())

    def nf_comparison_changed(self, checked):
        LOGGER.warning('NF comparison opt-in=%s capture=%s rt=%s', checked, self.capture_root, self.result_path)
        self.refresh_view()

    def clear_view(self):
        self.current_depth = self.current_stamp = None
        self.images.clear()
        for label in (self.window.sourceImageLabel, self.window.targetImageLabel):
            label.clear()
        self.window.planeText.clear()
        self.window.rtCheckStatus.clear()
        self.window.pickedDepthLabel.setText("표시할 촬영쌍이 없습니다.")
        self.update_controls()

    def update_controls(self):
        idle = self.worker_thread is None
        ready = self.current_stamp is not None
        frame = self.frames.get(self.current_stamp, {})
        allowed = self.result is not None and not fusion_block_reason(frame)
        for name in ("saveFusedButton", "saveProjectionButton"):
            getattr(self.window, name).setEnabled(idle and ready and allowed)
        self.window.viewComboBox.setEnabled(idle and ready)
        for index in range(1, 4):
            self.window.viewComboBox.model().item(index).setEnabled(allowed)
        self.window.saveSourceButton.setEnabled(idle and ready)

    def show_frame(self, stamp):
        self.current_depth = None
        try:
            frame = self.frame(stamp)
            self.mark_item(stamp, frame)
            self.window.geometryNoticeLabel.setText(frame.get('geometry_warning',''))
            nf = self.geometry == 'pcd' and frame.get('source_lens') == 'nf'
            self.window.applyWfButton.setVisible(nf)
            self.window.nfCompareCheckBox.setVisible(nf)
            blocked = fusion_block_reason(frame)
            can_fuse = self.result is not None and not blocked
            self.window.activeModelLabel.setText(
                f"적용 중: {GEOMETRY_LABELS[GEOMETRY_MODES.index(self.geometry)]} | RT: {self.result_path.name if self.result else '없음'}"
                + (' | 독립 정확도 미검증' if self.result else '')
                + (' | NF 비교 전용' if nf and not blocked else '')
                + (f" | NSL 독립 치수 보정 ×{self.result['source_ray_correction']['scale']:.5f}"
                   if self.result and self.result.get('source_ray_correction') else ''))
            self.window.activeModelLabel.setStyleSheet('font-weight:bold;color:' + ('#b00020' if nf else '#005a9c'))
            view_mode = self.window.viewComboBox.currentIndex()
            rgb_view = view_mode == 1 and can_fuse
            if rgb_view:
                source_view, count = fused_color_view(frame, self.result, self.color_cal)
                frame["visible_rgb_count"] = count
                self.window.sourceGroup.setTitle(f"실제 Color RGB 점군 · NSL 시점 ({count:,}점, 검정=매칭 없음)")
            elif view_mode == 2 and can_fuse:
                source_view, _ = projection_view(
                    frame, self.result, self.color_cal
                )
                self.window.sourceGroup.setTitle("검사용 거리 색상 투영 — RGB PLY 색상이 아님")
            else:
                displayed_plane = displayed_board(frame)
                partial = (frame.get('board_identity') or {}).get('state') == 'partial'
                plane_color = (255, 190, 0) if partial else PLANE_CANDIDATE_RGB[::-1]
                if displayed_plane is None:
                    title = "NSL — 보드 대응 미확인 / 모호한 후보는 선택 보류"
                else:
                    title = "NSL 보드 부분 대응 (파랑 · RT 학습 적합성 별도)" if partial else "NSL 보드 대응 (초록 · 정합 정확도 합격 아님)"
                source_view = pointcloud_view(
                    frame["xyz_mm"], frame["pcd_shape"],
                    displayed_plane, self.z_min, self.z_max,
                    plane_color=plane_color,
                )
                self.window.sourceGroup.setTitle(title)

            target_view = frame["color"] if rgb_view else color_plane_view(
                frame["color"], frame["target_plane"]
            )
            if view_mode == 3 and can_fuse:
                target_view = board_error_view(frame, self.result, self.color_cal)
            self.window.targetGroup.setTitle('Color 점-평면 잔차 / 자홍=보드 밖 투영 (정답 검증 아님)' if view_mode == 3 and can_fuse else 'RealSense Color 체스보드 평면')
            self.set_image(self.window.sourceImageLabel, source_view)
            self.set_image(self.window.targetImageLabel, target_view)
            self.current_stamp = stamp
            if can_fuse and "aligned_depth" not in frame:
                frame["aligned_depth"] = aligned_depth(frame, self.result, self.color_cal)
            self.current_depth = frame.get("aligned_depth") if can_fuse else None
            self.window.detectionStatus.setText(
                self.status_text(frame)
            )
            self.window.planeText.setPlainText(self.plane_text(frame))
            self.window.rtCheckStatus.setText(self.rt_status_text(frame))
            self.window.pickedDepthLabel.setText(
                blocked or "Color 클릭: 투영된 NSL 점의 Color 좌표계 Z (빈 픽셀은 보간하지 않음)"
            )
            self.update_controls()
        except Exception as exc:
            LOGGER.exception("show frame failed stamp=%s", stamp)
            self.window.detectionStatus.setText(f"표시 실패: {exc}")
            self.clear_view()
            self.apply_status(stamp, {"state": "failed", "message": f"표시 실패: {exc}"})

    def status_text(self, frame):
        return plane_status(frame, self.result)['message']

    def rt_status_text(self, frame):
        if identity := frame.get('board_identity'):
            e = identity.get('evidence')
            return (f"보드 대응 검사: 유한 보드 RMS {e['finite_board_rms_m']*1000:.1f} mm / "
                    f"면적 커버 {e['coverage']:.0%} / {identity['basis']}" if e else identity['basis'])
        if self.result is None:
            return 'RT 정합: 미검사 (RT 없음). 평면 검출과 정합 검사는 별개입니다.'
        if 'rt_normal_error_deg' not in frame:
            return 'RT 정합: 평면쌍 없음 — 검사 불가'
        return (f"추정 RT와의 일치도 (정답 검증 아님): 법선 {frame['rt_normal_error_deg']:.2f}° / "
                f"평면 오프셋 차이 {1000*frame['rt_plane_error_m']:.1f} mm / 면적 커버 {frame.get('projection_coverage',0):.0%}"
                f" / 전체 점 RMS {1000*frame.get('rt_point_rms_m', float('nan')):.1f} mm"
                f" / 보드 밖 {frame.get('projection_outside', 0):.1%}"
                f" / 경계 평균 {frame.get('boundary_mean_px', float('nan')):.1f} px")

    @staticmethod
    def one_plane_text(name, plane):
        if plane is None:
            return f"{name}: not detected"
        n = plane["normal"]
        text = (
            f"{name}: n=[{n[0]:+.5f}, {n[1]:+.5f}, {n[2]:+.5f}], "
            f"h={plane['h']:+.5f} m"
        )
        if "rmse" in plane:
            unit = "px" if "region" in plane else "mm"
            value = plane["rmse"] if unit == "px" else plane["rmse"] * 1000
            text += f", fit={value:.3f} {unit}"
        if "size" in plane:
            text += (
                f", size={plane['size'][0] * 1000:.0f}×"
                f"{plane['size'][1] * 1000:.0f} mm"
            )
        if 'full_rmse' in plane:
            text += f", 전체={plane['full_rmse']*1000:.2f}mm, 실측 셀={plane['support_occupancy']:.0%}"
        if 'support_inlier_fraction' in plane:
            text += f", 원본 지지점 유지={plane['support_inlier_fraction']:.0%}"
        return text

    def plane_text(self, frame):
        if self.window.viewComboBox.currentIndex() == 1 and self.result and not fusion_block_reason(frame):
            return (f"원본 사진 RGB → 투영점 {frame.get('visible_rgb_count', 0):,}개 (거리 색상표 미사용)\n"
                    "미리보기: NSL 시점 / RGB PLY의 XYZ: Color 카메라 좌표계, 미터\n"
                    "평면 판정과 RGB 저장은 별개입니다. 검정 영역은 색을 가져올 수 없는 점입니다.\n"
                    f"인접 점의 표면 가림 처리: {frame.get('projection_stats', {}).get('surface_occluded_points', 0):,}점 제외.\n"
                    "검정=시야 밖/가림/유효점 없음. 보드 밖 점을 숨기거나 RGB를 보드 안으로 당기지 않습니다.")
        if identity := frame.get('board_identity'):
            text = self.one_plane_text('NSL 대응 후보 (보정 전 원본 기하)', displayed_board(frame))
            text += '\n' + self.one_plane_text('Color', frame['target_plane'])
            text += '\n' + identity.get('reason', identity['basis'])
            if e := identity.get('evidence'):
                text += (f"\n보드 내부 점 {e['overlap']:.1%} / 유한 보드 근접 점 {e['near_fraction']:.1%}. "
                         f"물체 대응 허용폭 {e['identity_tolerance_m']*1000:.1f} mm = 보드 대각선의 15% (정밀도 기준 아님).")
            text += '\n' + identity['basis'] + '. 표시 후보는 RT 재학습/NSL 치수 보정에 피드백하지 않습니다.'
            return text
        plane = frame["source_plane"]
        text = (
            self.one_plane_text("NSL (RT 독립 검출점)", plane)
            + "\n"
            + self.one_plane_text("Color", frame["target_plane"])
        )
        if 'corrected_source_size_m' in frame:
            a, b = np.array(frame['corrected_source_size_m'])*1000
            text += f"\nNSL 독립 각도 보정 후 같은 지지점 치수: {a:.1f}×{b:.1f} mm (원본 파일 보존 / NSL PLY에도 적용)"
        if self.result and "rt_normal_error_deg" in frame:
            text += (
                f"\nAfter RT: normal error="
                f"{frame['rt_normal_error_deg']:.3f} deg, "
                f"plane offset difference={frame['rt_plane_error_m'] * 1000:.3f} mm (비평행 평면의 거리 아님)"
            )
        if "projection_overlap" in frame:
            text += (
                f"\nColor 실제 보드 안 투영 (여유 0%)="
                f"{frame['projection_overlap'] * 100:.1f}% / 실측 셀 커버 {frame.get('projection_coverage', 0):.0%} / 셀 IoU {frame.get('projection_iou', 0):.0%}"
            )
            if 'boundary_mean_px' in frame:
                text += f"\n양방향 경계 오차: 평균 {frame['boundary_mean_px']:.2f} px / P90 {frame['boundary_p90_px']:.2f} px. 자홍색 점=보드 밖 투영."
        elif "projection_overlap_best" in frame:
            text += (
                f"\nBest Color checkerboard overlap="
                f"{frame['projection_overlap_best'] * 100:.1f}%"
            )
        reasons = frame.get('quality_failures') or (plane or {}).get('quality_reasons', [])
        if 'rt_point_rms_m' in frame:
            text += f"\n실측 점→Color 평면 RMS={frame['rt_point_rms_m']*1000:.1f}mm / 관측 지지영역 중심 차이={frame['rt_center_error_m']*1000:.1f}mm (부분 보드에서는 참고값)"
        if reasons:
            text += '\n모델 일치도 참고 (검출 판정 아님): ' + '; '.join(reasons)
        learned = frame.get('calibration_plane')
        if learned is not None and learned is not plane:
            text += '\n저장된 RT 학습 후보와 현재 독립 검출 후보가 다릅니다. 학습 이력은 저장 지지점을 별도 대조합니다.'
        return text

    def set_image(self, label, image):
        self.images[label] = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimage = QImage(
            rgb.data, w, h, c * w, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def pick_depth(self, position):
        if self.current_depth is None:
            self.window.pickedDepthLabel.setText(
                "먼저 NSL → Color RT를 계산하세요."
            )
            return
        label = self.window.targetImageLabel
        pixmap = label.pixmap()
        if pixmap is None:
            return
        rect = label.contentsRect()
        left = rect.x() + (rect.width() - pixmap.width()) / 2
        top = rect.y() + (rect.height() - pixmap.height()) / 2
        x, y = position.x() - left, position.y() - top
        if not (0 <= x < pixmap.width() and 0 <= y < pixmap.height()):
            return
        h, w = self.current_depth.shape
        u = min(int(x * w / pixmap.width()), w - 1)
        v = min(int(y * h / pixmap.height()), h - 1)
        depth = float(self.current_depth[v, u])
        if depth > 0:
            text = f"Color pixel ({u}, {v}) → Color 좌표계 Z = {depth * 1000:.1f} mm (NSL 투영점)"
        else:
            text = f"Color pixel ({u}, {v}) → 매칭 NSL point 없음"
        self.window.pickedDepthLabel.setText(text)

    def run_calibration(self):
        self.start_job(inspect_only=False)

    def validate_all(self):
        self.start_job(inspect_only=True)

    def start_job(self, inspect_only):
        if self.worker_thread is not None:
            return
        if not self.pairs or (not inspect_only and len(self.pairs) < 6):
            self.window.statusbar.showMessage("RT 계산에는 최소 6개 촬영쌍이 필요합니다.")
            return
        try:
            self.z_min, self.z_max = self.read_range()
        except ValueError as exc:
            self.window.statusbar.showMessage(str(exc))
            return
        self.settings.setValue("input/z_min", self.z_min)
        self.settings.setValue("input/z_max", self.z_max)
        self.settings.sync()
        self.frames.clear()
        thread = QThread(self)
        worker = CalibrationWorker(
            list(self.pairs), self.color_cal, self.z_min, self.z_max,
            self.result_path if inspect_only else self.point_plane_result_path(), self.capture_root, self.result, inspect_only, self.geometry)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.calibration_progress)
        worker.status.connect(self.apply_status)
        worker.stage.connect(self.calibration_stage)
        worker.succeeded.connect(self.calibration_succeeded)
        worker.failed.connect(self.calibration_failed)
        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.calibration_finished)
        self.worker_thread, self.worker = thread, worker
        self.set_busy(True)
        self.window.progressBar.setRange(0, len(self.pairs))
        self.window.progressBar.setValue(0)
        self.window.progressBar.setVisible(True)
        self.window.calibrationStatus.setText("현재 RT로 평면 검사 중..." if inspect_only else "RT 계산용 평면 검출 중...")
        LOGGER.info("NSL job requested inspect_only=%s pairs=%d", inspect_only, len(self.pairs))
        thread.start()

    def set_busy(self, busy):
        for name in ("inputGroup", "reloadButton", "calibrateButton", "fileList",
                     "saveSourceButton", "saveFusedButton", "saveProjectionButton",
                     "viewComboBox", "validateButton", "loadRtButton", "intensityCheckBox", "applyWfButton", "nfCompareCheckBox"):
            getattr(self.window, name).setEnabled(not busy)

    def calibration_stage(self, message):
        self.window.progressBar.setRange(0, 0)
        self.window.calibrationStatus.setText(message)

    def calibration_progress(self, index, total, stamp, source_ok, target_ok):
        self.window.progressBar.setValue(index)
        self.window.calibrationStatus.setText(
            f"평면 검출 {index}/{total}: {stamp}"
        )
        items = self.window.fileList.findItems(stamp, Qt.MatchExactly)
        if items and not (source_ok and target_ok):
            items[0].setBackground(QColor("#ffd6d6"))
            items[0].setForeground(QColor("#8b0000"))

    def calibration_succeeded(self, result, statuses):
        self.result = result
        if not self.worker.inspect_only:
            self.result_path = self.worker.result_path
            self.settings.setValue(f'rt/{self.geometry}', str(self.result_path))
            self.settings.sync()
        self.frames.clear()
        for stamp, status in statuses.items():
            self.apply_status(stamp, status)
        self.show_matrix()
        counts = {state: sum(s["state"] == state for s in statuses.values())
                  for state in ("failed", "candidate", "partial", "unconfirmed")}
        used = sum(s.get('training') == 'used' for s in statuses.values())
        excluded = sum(s.get('training') == 'excluded' for s in statuses.values())
        action = "RT 유지 · 검사 완료" if self.worker.inspect_only else "RT 계산 + 검사 완료"
        self.window.calibrationStatus.setText(
            f"{action}: 대응 {counts['candidate']} / 부분 {counts['partial']} / 보류 {counts['unconfirmed']} / 검출 없음 {counts['failed']} | RT 사용 {used} / 제외 {excluded}")
        LOGGER.info("%s counts=%s", action, counts)

    def calibration_failed(self, error):
        self.window.calibrationStatus.setText(
            f"실패: {error} / 로그: {get_log_path()}"
        )

    def calibration_finished(self):
        self.set_busy(False)
        self.window.progressBar.setVisible(False)
        self.worker_thread = self.worker = None
        self.window.statusbar.showMessage(f"계산 종료 / 로그: {get_log_path()}")
        self.refresh_view()
        self.update_controls()

    def show_matrix(self):
        if not self.result:
            self.window.matrixText.clear()
            return
        rt = np.asarray(self.result["RT"])
        text = (
            "NSL PointCloud → RealSense Color RT\n"
            + GEOMETRY_LABELS[GEOMETRY_MODES.index(self.geometry)] + '\n'
            + np.array2string(rt, precision=9, suppress_small=False)
            + f"\n\nplanes={self.result.get('plane_pairs', 0)}, "
            + f"inliers={self.result.get('ransac_inliers', 0)}, "
            + f"normal RMSE={self.result.get('normal_rmse_deg', float('nan')):.3f} deg, "
            + f"plane RMSE={self.result.get('plane_rmse_m', float('nan')) * 1000:.3f} mm, "
            + f"optimizer={self.result.get('nonlinear_method', 'N/A')}"
        )
        if "ransac_angle_threshold_deg" in self.result:
            text += (
                f"\nRANSAC threshold="
                f"{self.result['ransac_angle_threshold_deg']:.1f} deg / "
                f"{self.result['ransac_plane_threshold_m'] * 1000:.0f} mm"
            )
        total = self.result.get('input_frame_count', self.result.get('plane_pairs', 0))
        used = self.result.get('ransac_inliers', 0)
        text += (f"\n전체 {total} / 학습 후보 {self.result.get('eligible_frame_count', self.result.get('plane_pairs', 0))}"
                 f" / 실제 사용 {used} / 제외 {total-used}")
        text += "\n미검증 추정 RT: 내부 일관성 검사는 절대 정확도 검증이 아닙니다. 검출색은 학습 사용 여부와 무관합니다."
        if self.result.get('calibration_version', 0) < 3:
            text += '\n주의: 구버전 RT는 전체 보드 크기·면적 검증 없이 계산됨. 원본 점군 기하 확인 후 재계산 필요.'
        if self.geometry == 'wf_range':
            text += '\nWF는 제조사 표 기반 복원 모델이며 개별 장비 실측 내부 보정값은 아닙니다. 원본 PCD는 변경하지 않습니다.'
        if correction := self.result.get('source_ray_correction'):
            text += (f"\n보드 기반 추정 광선 각도 ×{correction['scale']:.6f} → 위 RT 적용 (원본 거리 보존)."
                     " RT만 따로 적용하면 동일 결과가 나오지 않습니다. 공장 내부 보정값이 아닙니다.")
            audit = self.result.get('nsl_dimension_audit', {})
            if audit.get('applied'):
                text += (f"\nNSL 자체 치수 검사 {audit['eligible_frames']}장 (RT/RGB 미사용): "
                         f"분리 검증 치수 오차 {audit['heldout_before_mae_mm']:.1f} → {audit['heldout_after_mae_mm']:.1f} mm")
        if self.result.get('calibration_warning'):
            warning = self.result['calibration_warning']
            # Old saved results describe the retired learning-membership colors.
            if warning.startswith('초록='):
                warning = warning.partition('정합 오차는 별도로 표시합니다.')[2].strip()
            if warning:
                text += '\n주의: ' + warning
        if self.result.get('independent_validation'):
            validation = self.result['independent_validation']
            text += f"\n별도 학습/검증 분할 시험: {validation['passed_count']}/{validation['heldout_count']} 통과 (현 RT의 독립 성적과 구분)."
        if self.result.get('cross_validation'):
            v = self.result['cross_validation']
            text += f"\n4분할 교차검증 {v['evaluated']}개 (새 촬영 검증 아님): 점-평면 중앙값 {v['point_rms_median_mm']:.1f}mm / 중심 {v['center_median_mm']:.1f}mm" if v.get('evaluated') else '\n교차검증 결과 없음'
            if v.get('evaluated'):
                text += f" / P90 {v['point_rms_p90_mm']:.1f}mm, {v['center_p90_mm']:.1f}mm"
            if 'boundary_mean_px_median' in v:
                text += (f"\n분리 검증 경계 오차 중앙값: {v['baseline_boundary_mean_px_median']:.1f} → {v['boundary_mean_px_median']:.1f} px"
                         f" / 보드 밖 투영: {v['baseline_outside_fraction_median']:.1%} → {v['outside_fraction_median']:.1%}")
            if self.result.get('calibration_version', 0) >= 6:
                objective = ('점-평면·양방향 영상 경계' if self.result.get('calibration_version', 0) >= 7
                             else '점-평면·판 범위 이탈')
                text += f'\n시간 구간 분리 검증. RANSAC+LM: 고정 후보 대응·{objective}. 부분 중심 강제 일치 없음.'
                text += '\n20mm는 잠정 정합 허용치이며 장비 실측 신뢰한계가 아닙니다. 학습 사용 ≠ 검출/정합 정확도 보증.'
            else:
                text += '\n구버전: 셀 평균 RMS <20mm + 중심 <60mm (장비 실측 신뢰한계 아님). 새 검출/계산 방식으로 재계산 권장.'
        if not self.result.get("fitted_plane_selection"):
            text += "\n구버전 RT: 현재 후보 평면을 재검사합니다. 과거 학습 평면과 동일하다는 뜻은 아닙니다."
        text += f"\nRT 파일: {self.result_path}"
        if self.result.get("capture_root") and Path(self.result["capture_root"]).resolve() != self.capture_root.resolve():
            text += "\n다른 촬영 폴더에 RT 재사용 중: 동일한 센서 장착 상태인지 확인하세요."
        self.window.matrixText.setPlainText(text)

    def selected_frame(self):
        item = self.window.fileList.currentItem()
        if item is None:
            raise RuntimeError("촬영쌍을 선택하세요.")
        return item.text(), self.frame(item.text())

    def save_source(self):
        try:
            stamp, frame = self.selected_frame()
        except Exception as exc:
            self.window.statusbar.showMessage(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self.window, "NSL PLY 저장",
            str(output_dir("pointclouds") / f"nsl_{self.geometry}_{stamp}.ply"),
            "PLY point cloud (*.ply)"
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        try:
            total, selected = save_source_cloud(
                Path(path), frame,
                include_intensity=self.window.intensityCheckBox.isChecked(),
                calibration=self.result,
            )
            self.window.statusbar.showMessage(
                f"저장 완료: {total:,} points / 평면 {selected:,} points — {path}"
            )
        except Exception as exc:
            LOGGER.exception("source PLY save failed")
            self.window.statusbar.showMessage(f"저장 실패: {exc}")

    def save_fused(self):
        if not self.result:
            self.window.statusbar.showMessage("먼저 RT를 계산하세요.")
            return
        try:
            stamp, frame = self.selected_frame()
            if reason := fusion_block_reason(frame):
                raise ValueError(reason)
        except Exception as exc:
            self.window.statusbar.showMessage(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Color RGB 융합 PLY 저장",
            str(output_dir("pointclouds") / f"nsl_color_{self.geometry}_{stamp}.ply"),
            "PLY point cloud (*.ply)"
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        try:
            stats = save_fused_cloud(
                Path(path), frame, self.result, self.color_cal,
                include_intensity=self.window.intensityCheckBox.isChecked(),
            )
            self.window.statusbar.showMessage(
                f"원본 사진 RGB 저장: {stats['output_points']:,}점 — {path} / 미리보기: {stats['preview']}"
            )
        except Exception as exc:
            LOGGER.exception("fused PLY save failed")
            self.window.statusbar.showMessage(f"융합 저장 실패: {exc}")

    def save_projection_image(self):
        if not self.result:
            self.window.statusbar.showMessage("먼저 RT를 계산하세요.")
            return
        try:
            stamp, frame = self.selected_frame()
            if reason := fusion_block_reason(frame):
                raise ValueError(reason)
        except Exception as exc:
            self.window.statusbar.showMessage(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Color 투영 PNG 저장",
            str(output_dir("images") / f"nsl_color_projection_{stamp}.png"),
            "PNG image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            count = save_projection(
                Path(path), frame, self.result, self.color_cal
            )
            self.window.statusbar.showMessage(
                f"투영 이미지 저장 완료: {count:,} points — {path}"
            )
        except Exception as exc:
            LOGGER.exception("projection save failed")
            self.window.statusbar.showMessage(f"투영 저장 실패: {exc}")


def main():
    install_exception_hook(LOGGER)
    app = QApplication(sys.argv)
    controller = Controller()
    controller.window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
