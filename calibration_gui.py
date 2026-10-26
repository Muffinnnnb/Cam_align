import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QThread, QSignalBlocker, Signal, Slot, Qt, QEventLoop
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog

from app_paths import APP_DIR, calibration_input, output_dir
from app_logging import LOG_PATH, get_logger, start_hang_watchdog, stop_hang_watchdog
from calibrate_d_to_r import (DEFAULT_DEPTH_DIR, DEFAULT_TARGET_DIR, RESULT_PATH,
                              align_depth_to_right, analyze_frame, calibrate,
                              save_ir_point_cloud, save_point_cloud, timestamps)
from realsense_compare import (comparison_metrics, rs_align_depth,
                               save_comparison, save_factory_calibration)


LOGGER = get_logger("gui")


def plane_marked(image, plane, roi=None):
    view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if plane is None:
        cv2.putText(view, "PLANE NOT DETECTED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)
    else:
        overlay = view.copy()
        if "mask" in plane:
            overlay[plane["mask"] > 0] = (0, 220, 80)
        else:
            cv2.fillConvexPoly(overlay, np.round(plane["region"]).astype(int), (0, 220, 80))
        view = cv2.addWeighted(view, .65, overlay, .35, 0)
        start, end = np.round(plane["axis"]).astype(int)
        cv2.arrowedLine(view, tuple(start), tuple(end), (0, 255, 255), 3, tipLength=.2)
        cv2.putText(view, "normal", tuple(end + (5, -5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
    if roi is not None:
        polygon = np.round(roi["polygon"]).astype(np.int32)
        cv2.polylines(view, [polygon], True, (255, 255, 0), 2)
        # cv2.putText(view, "Target ROI", tuple(polygon[0]), cv2.FONT_HERSHEY_SIMPLEX,
        #             .5, (255, 255, 0), 2)

        ys, xs = np.where(roi["mask"] > 0)

        if len(xs) > 0:
            text_x = int(xs.min()) + 5
            text_y = int(ys.min()) + 20

            cv2.putText(
                view,
                "Target ROI",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
                (255, 255, 0),
                2
            )

    return view


def depth_colormap(depth):
    valid = depth > 0
    view = np.zeros((*depth.shape, 3), np.uint8)
    if not valid.any():
        return view
    low, high = np.percentile(depth[valid], (2, 98))
    gray = np.zeros(depth.shape, np.uint8)
    gray[valid] = np.clip((depth[valid] - low) * 255 / max(high - low, 1e-6), 0, 255)
    view = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    view[~valid] = 0
    return view


class ComparisonSaveWorker(QObject):
    progress = Signal(str)
    succeeded = Signal(object, str)
    failed = Signal(str)
    done = Signal()

    def __init__(self, folder, stamp, frame, calibration, source, custom=None, realsense=None):
        super().__init__()
        self.args = folder, stamp, frame, calibration, source
        self.custom, self.realsense = custom, realsense

    @Slot()
    def run(self):
        folder, stamp, frame, calibration, source = self.args
        try:
            if self.custom is None:
                self.progress.emit("현재 방식 Depth 정렬 중")
                LOGGER.info("comparison custom align start stamp=%s", stamp)
                self.custom = align_depth_to_right(frame, calibration)
            if self.realsense is None:
                self.progress.emit("rs.align() 계산 중")
                LOGGER.info("comparison rs.align start stamp=%s", stamp)
                self.realsense = rs_align_depth(frame, calibration)
            metrics = save_comparison(folder, stamp, frame, calibration,
                                      self.custom, self.realsense, source,
                                      self.progress.emit)
            self.succeeded.emit(metrics, str(folder))
        except Exception as error:
            LOGGER.exception("comparison save failed stamp=%s folder=%s", stamp, folder)
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            self.done.emit()


class Controller(QObject):
    def __init__(self, settings_path=None):
        super().__init__()
        self.window = QUiLoader().load(str(APP_DIR / "calibration_viewer.ui"))
        self.images = {}
        self.frames = {}
        self.result = None
        self.factory_result = None
        self.settings = QSettings(str(settings_path or APP_DIR / "viewer_settings.ini"), QSettings.IniFormat)
        self.depth_dir = Path(self.settings.value('input/depth_dir', str(DEFAULT_DEPTH_DIR)))
        self.target_dir = Path(self.settings.value('input/target_dir', str(DEFAULT_TARGET_DIR)))
        self.loading = False
        self.result_path = RESULT_PATH
        self.aligned_depth = None
        self.rs_depth = None
        self.rs_depth_stamp = None
        self.current_stamp = None
        self.save_thread = self.save_worker = None
        self.nsl_controller = None
        self.depth_base = self.right_base = None
        self.window.depthFolderEdit.setText(str(self.depth_dir))
        self.window.targetFolderEdit.setText(str(self.target_dir))
        self.window.zMinSpinBox.setValue(float(self.settings.value("detection/z_min", .15)))
        self.window.zMaxSpinBox.setValue(float(self.settings.value("detection/z_max", 2.0)))
        self.window.useTargetRoiCheckBox.setChecked(
            self.settings.value("detection/use_target_roi", True, type=bool))
        self.window.roiMarginSpinBox.setValue(int(self.settings.value("detection/roi_margin_percent", 20)))
        for label in (self.window.depthImageLabel, self.window.rightImageLabel):
            label.installEventFilter(self)
        self.window.installEventFilter(self)
        self.window.fileList.currentItemChanged.connect(self.show_item)
        self.window.reloadButton.clicked.connect(self.reload)
        self.window.savePointCloudButton.clicked.connect(self.save_point_cloud)
        self.window.saveIrPointCloudButton.clicked.connect(self.save_ir_point_cloud)
        self.window.calibrateButton.clicked.connect(self.run_calibration)
        self.window.showAlignedDepthCheckBox.toggled.connect(
            self.show_aligned_toggled
        )
        self.window.compareRsCheckBox.toggled.connect(
            self.compare_rs_toggled
        )


        self.window.calibrationSourceCombo.currentIndexChanged.connect(self.calibration_changed)
        self.window.depthFolderButton.clicked.connect(lambda: self.choose_folder(True))
        self.window.targetFolderButton.clicked.connect(lambda: self.choose_folder(False))
        self.window.saveFactoryButton.clicked.connect(self.save_factory)
        self.window.saveComparisonButton.clicked.connect(self.save_alignment_comparison)
        self.window.applyDetectionSettingsButton.clicked.connect(self.apply_detection_settings)
        self.window.openNslColorButton.clicked.connect(self.open_nsl_color)
        self.load_calibrations()
        self.reload()
        LOGGER.info("GUI ready depth=%s target=%s log=%s", self.depth_dir,
                    self.target_dir, LOG_PATH)

    def open_nsl_color(self):
        try:
            from nsl_color_gui import Controller as NslColorController
            if self.nsl_controller is None:
                self.nsl_controller = NslColorController()
            self.nsl_controller.window.show()
            self.nsl_controller.window.raise_()
            self.nsl_controller.window.activateWindow()
            LOGGER.info("NSL PointCloud -> Color window opened")
        except Exception as error:
            LOGGER.exception("NSL PointCloud -> Color window open failed")
            self.window.statusbar.showMessage(f"NSL 융합 창 열기 실패: {error}")

    def is_busy(self):
        return self.loading or self.save_thread is not None or any(
            not getattr(self.window, name).isEnabled() for name in
            ('calibrateButton', 'savePointCloudButton', 'saveIrPointCloudButton'))

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Resize and obj in self.images:
                self.set_image(obj, self.images[obj])
            elif event.type() == QEvent.MouseButtonPress and obj == self.window.rightImageLabel:
                self.pick_depth(event.position())
                return True
            elif event.type() == QEvent.Close and obj == self.window and self.is_busy():
                self.window.statusbar.showMessage("분석·계산·저장이 끝난 후 종료할 수 있습니다.")
                event.ignore()
                return True
            return super().eventFilter(obj, event)
        except RuntimeError:
            return False

    def calibration_file(self, factory=False):
        if self.depth_dir.resolve() == DEFAULT_DEPTH_DIR.resolve() and self.target_dir.resolve() == DEFAULT_TARGET_DIR.resolve():
            return RESULT_PATH.with_name("factory_d_to_r.json") if factory else RESULT_PATH
        name = f"{self.depth_dir.name}_to_{self.target_dir.name}"
        return RESULT_PATH.with_name(f"{'factory_' if factory else ''}{name}_rt.json")

    def load_calibrations(self):
        self.result_path = self.calibration_file()
        factory_path = self.calibration_file(True)
        read_path, factory_path = calibration_input(self.result_path), calibration_input(factory_path)
        self.result = json.loads(read_path.read_text(encoding="utf-8")) if read_path.exists() else None
        self.factory_result = json.loads(factory_path.read_text(encoding="utf-8")) if factory_path.exists() else None
        self.calibration_changed()

    def active_calibration(self):
        return self.factory_result if self.window.calibrationSourceCombo.currentIndex() else self.result

    def calibration_changed(self):
        self.rs_depth = None
        self.rs_depth_stamp = None
        
        calibration = self.active_calibration()
        source = "공장값" if self.window.calibrationSourceCombo.currentIndex() else "직접 측정값"
        if calibration:
            self.show_matrix(calibration, source)
        else:
            self.window.calibrationStatus.setText(f"{source} 없음")
            self.window.matrixText.clear()
        self.refresh_view()

    def choose_folder(self, depth):
        current = self.depth_dir if depth else self.target_dir

        kind = "depth" if depth else "target"

        LOGGER.info(
            "FILE_DIALOG OPEN start type=folder kind=%s current=%s save_thread=%s frames=%d",
            kind,
            current,
            self.save_thread is not None and self.save_thread.isRunning(),
            len(self.frames),
        )


        start_hang_watchdog(20)

        try:
            path = QFileDialog.getExistingDirectory(
                self.window,
                "Depth 폴더 선택" if depth else "정렬 대상 폴더 선택",
                str(current),
            )
        finally:
            stop_hang_watchdog()


        LOGGER.info(
            "FILE_DIALOG OPEN returned type=folder kind=%s path=%r",
            kind,
            path,
        )

        if not path:
            LOGGER.info("FILE_DIALOG cancelled type=folder kind=%s", kind)
            return

        LOGGER.info("folder validation start path=%s", path)
        
        folder = Path(path)

        missing = [name for name in ("Intrinsic.txt", "Distortion.txt") if not (folder / name).exists()]

        LOGGER.info(
            "folder validation done path=%s missing=%s",
            path,
            missing,
        )


        if missing:
            self.window.statusbar.showMessage("선택 폴더에 필요한 파일이 없습니다: " + ", ".join(missing))
            return
        if depth:
            self.depth_dir = folder
            self.window.depthFolderEdit.setText(path)
        else:
            self.target_dir = folder
            self.window.targetFolderEdit.setText(path)
        self.settings.setValue('input/depth_dir', str(self.depth_dir))
        self.settings.setValue('input/target_dir', str(self.target_dir))
        self.settings.sync()
        self.frames.clear()

        self.rs_depth = None
        self.rs_depth_stamp = None

        LOGGER.info("folder load_calibrations start path=%s", path)
        self.load_calibrations()
        LOGGER.info("folder load_calibrations done path=%s", path)

        LOGGER.info("folder reload start path=%s", path)
        self.reload()
        LOGGER.info("folder reload done path=%s", path)


    def target_kind(self, frame):
        choice = self.window.targetTypeCombo.currentIndex()
        return "right_ir" if choice == 1 or (choice == 0 and frame["right"].ndim == 2) else "color"

    def detection_options(self):
        z_min, z_max = self.window.zMinSpinBox.value(), self.window.zMaxSpinBox.value()
        if z_min >= z_max:
            raise ValueError("z_min은 z_max보다 작아야 합니다.")
        return {"z_min": z_min, "z_max": z_max,
                "roi_calibration": self.active_calibration()
                if self.window.useTargetRoiCheckBox.isChecked() else None,
                "roi_margin": self.window.roiMarginSpinBox.value() / 100}

    def analyze(self, stamp):
        return analyze_frame(stamp, self.depth_dir, self.target_dir, **self.detection_options())

    def apply_detection_settings(self):
        try:
            options = self.detection_options()
        except ValueError as error:
            self.window.statusbar.showMessage(str(error))
            return
        self.settings.setValue("detection/z_min", options["z_min"])
        self.settings.setValue("detection/z_max", options["z_max"])
        self.settings.setValue("detection/use_target_roi", self.window.useTargetRoiCheckBox.isChecked())
        self.settings.setValue("detection/roi_margin_percent", self.window.roiMarginSpinBox.value())
        self.settings.sync()
        LOGGER.info("settings saved z=%.3f..%.3f roi=%s margin=%d%%",
                    options["z_min"], options["z_max"],
                    self.window.useTargetRoiCheckBox.isChecked(),
                    self.window.roiMarginSpinBox.value())
        self.reload()

    def save_factory(self):
        item = self.window.fileList.currentItem()
        if item is None:
            self.window.statusbar.showMessage("먼저 파일을 선택하세요.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            frame = self.frames[item.text()]
            height, width = frame["right"].shape[:2]
            path = self.calibration_file(True)
            self.factory_result = save_factory_calibration(path, self.target_kind(frame), width, height)
            if self.window.calibrationSourceCombo.currentIndex() == 1:
                self.calibration_changed()
            else:
                self.window.calibrationSourceCombo.setCurrentIndex(1)
            self.window.statusbar.showMessage(f"공장 보정값 저장 완료: {path}")
        except Exception as error:
            LOGGER.exception("factory calibration save failed")
            self.window.statusbar.showMessage(f"공장 보정값 저장 실패: {error}")
        finally:
            QApplication.restoreOverrideCursor()

    def pick_depth(self, position):
        if self.aligned_depth is None or self.window.rightImageLabel.pixmap() is None:
            self.window.pickedDepthLabel.setText("RT 결과와 정렬 Depth가 필요합니다.")
            return
        label, pixmap = self.window.rightImageLabel, self.window.rightImageLabel.pixmap()
        rect = label.contentsRect()
        left = rect.x() + (rect.width() - pixmap.width()) / 2
        top = rect.y() + (rect.height() - pixmap.height()) / 2
        x, y = position.x() - left, position.y() - top
        if not (0 <= x < pixmap.width() and 0 <= y < pixmap.height()):
            return
        height, width = self.aligned_depth.shape
        u, v = min(int(x * width / pixmap.width()), width - 1), min(int(y * height / pixmap.height()), height - 1)
        depth = float(self.aligned_depth[v, u])
        text = (f"Target pixel ({u}, {v}) → Custom = {depth * 1000:.1f} mm"
                if depth > 0 else f"Target pixel ({u}, {v}) → Custom Depth 없음")
        if self.window.compareRsCheckBox.isChecked() and self.rs_depth is not None:
            rs_depth = float(self.rs_depth[v, u])
            text += (f", rs.align = {rs_depth * 1000:.1f} mm"
                     if rs_depth > 0 else ", rs.align Depth 없음")
            if depth > 0 and rs_depth > 0:
                text += f", Δ = {(depth - rs_depth) * 1000:+.1f} mm"
        self.window.pickedDepthLabel.setText(text)
        views = [(self.window.rightImageLabel, self.right_base)]
        if self.window.showAlignedDepthCheckBox.isChecked() or self.window.compareRsCheckBox.isChecked():
            views.append((self.window.depthImageLabel, self.depth_base))
        for label, base in views:
            marked = base.copy()
            cv2.drawMarker(marked, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            self.set_image(label, marked)

    def set_image(self, label, image):
        self.images[label] = image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimage).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reload(self):
        if self.loading:
            return
        self.loading = True
        try:
            self._reload()
        except Exception as error:
            LOGGER.exception('Depth input reload failed')
            self.window.statusbar.showMessage(f'입력 불러오기 실패: {error}')
        finally:
            self.window.fileList.blockSignals(False)
            self.loading = False

    def _reload(self):
        try:
            options = self.detection_options()
        except ValueError as error:
            self.window.statusbar.showMessage(str(error))
            return
        current = self.window.fileList.currentItem().text() if self.window.fileList.currentItem() else ""
        self.window.fileList.blockSignals(True)
        self.window.fileList.clear()
        self.frames.clear()
        LOGGER.info("reload start depth=%s target=%s z=%.3f..%.3f roi=%s",
                    self.depth_dir, self.target_dir, options["z_min"], options["z_max"],
                    options["roi_calibration"] is not None)
        failed = roi_used = 0
        for stamp in timestamps(self.depth_dir, self.target_dir):
            self.window.fileList.addItem(stamp)
            item = self.window.fileList.item(self.window.fileList.count() - 1)
            try:
                frame = analyze_frame(stamp, self.depth_dir, self.target_dir, **options)
                self.frames[stamp] = frame
                roi_used += frame["depth_roi"] is not None
                missing = [name for name, plane in (("Depth", frame["depth_plane"]),
                                                     ("Target", frame["right_plane"])) if plane is None]
            except Exception:
                LOGGER.exception("frame analysis failed stamp=%s", stamp)
                missing = ["file"]
            if missing:
                failed += 1
                item.setBackground(QColor("#ffd6d6"))
                item.setForeground(QColor("#8b0000"))
                item.setToolTip("검출 실패: " + ", ".join(missing))
            else:
                item.setToolTip("양쪽 검출 성공" + (" / Depth ROI 적용" if frame["depth_roi"] is not None else ""))
            self.window.statusbar.showMessage(f"검출 확인 중: {self.window.fileList.count()}개")
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
        self.window.fileList.blockSignals(False)
        matches = self.window.fileList.findItems(current, Qt.MatchExactly)
        self.window.fileList.setCurrentItem(matches[0] if matches else self.window.fileList.item(0))
        roi_text = (f" / ROI {roi_used}개 적용" if self.window.useTargetRoiCheckBox.isChecked()
                    else " / ROI 미사용")
        if self.window.useTargetRoiCheckBox.isChecked() and options["roi_calibration"] is None:
            roi_text = " / ROI 미적용(RT 없음)"
        self.window.statusbar.showMessage(
            f"매칭 {self.window.fileList.count()}개 / 검출 실패 {failed}개{roi_text} — "
            f"{self.depth_dir.name} → {self.target_dir.name}")
        LOGGER.info("reload done matched=%d failed=%d roi=%d", self.window.fileList.count(),
                    failed, roi_used)

    def show_item(self, item, previous):
        if item:
            self.show_frame(item.text())

    def refresh_view(self):
        item = self.window.fileList.currentItem()
        if item:
            self.show_frame(item.text())

    def show_frame(self, stamp):
        if not stamp:
            return
        self.current_stamp = None
        try:
            # frame = self.frames.get(stamp) or self.analyze(stamp)
            # calibration = self.active_calibration()
            # self.aligned_depth = align_depth_to_right(frame, calibration) if calibration else None
            # self.rs_depth = None
            # compare = calibration is not None and self.window.compareRsCheckBox.isChecked()
            
            frame = self.frames.get(stamp) or self.analyze(stamp)
            calibration = self.active_calibration()

            self.aligned_depth = (
                align_depth_to_right(frame, calibration)
                if calibration else None
            )

            compare = (
                calibration is not None
                and self.window.compareRsCheckBox.isChecked()
)
            
            
            
            if compare:
                # self.rs_depth = rs_align_depth(frame, calibration)
                if self.rs_depth is None or self.rs_depth_stamp != stamp:
                    LOGGER.info(
                        "rs.align cache miss stamp=%s previous_stamp=%s",
                        stamp,
                        self.rs_depth_stamp,
                    )
                    self.rs_depth = rs_align_depth(
                        frame,
                        calibration,
                    )

                    self.rs_depth_stamp = stamp
                else:
                    LOGGER.info(
                        "rs.align cache hit stamp=%s",
                        stamp,
                    )
                self.depth_base = depth_colormap(self.aligned_depth)
                self.right_base = depth_colormap(self.rs_depth)
                self.window.depthGroup.setTitle("현재 방식 정렬 Depth")
                self.window.rightGroup.setTitle("RealSense rs.align() Depth")
                metrics = comparison_metrics(self.aligned_depth, self.rs_depth)
                self.window.comparisonStatus.setText(
                    f"IoU={metrics['overlap_iou_percent']:.2f}% / MAE={metrics['mae_mm']:.3f} mm / "
                    f"RMSE={metrics['rmse_mm']:.3f} mm / P95={metrics['p95_abs_mm']:.3f} mm")
            else:
                show_aligned = calibration is not None and self.window.showAlignedDepthCheckBox.isChecked()
                if show_aligned:
                    self.depth_base = depth_colormap(self.aligned_depth)
                    self.window.depthGroup.setTitle("대상 픽셀로 정렬된 Depth")
                else:
                    depth_view = frame["color"]
                    if depth_view is None:
                        depth_view = cv2.applyColorMap(
                            cv2.convertScaleAbs(frame["depth"], alpha=.05), cv2.COLORMAP_JET)
                    self.depth_base = plane_marked(depth_view, frame["depth_plane"], frame.get("depth_roi"))
                    self.window.depthGroup.setTitle("원본 Depth 평면 (초록색 인라이어 + 법선)")
                self.right_base = plane_marked(frame["right"], frame["right_plane"])
                self.window.rightGroup.setTitle("정렬 대상 영상 (클릭하여 Depth 조회)")
                self.window.comparisonStatus.clear()
            self.set_image(self.window.depthImageLabel, self.depth_base)
            self.set_image(self.window.rightImageLabel, self.right_base)
            self.window.pickedDepthLabel.setText("대상 영상에서 픽셀을 클릭하세요.")
            ok = frame["depth_plane"] is not None and frame["right_plane"] is not None
            coverage = (100 * np.count_nonzero(self.aligned_depth) / self.aligned_depth.size
                        if self.aligned_depth is not None else 0)
            status = "양쪽 평면 검출 성공" if ok else "평면 검출 실패: RT 계산에서 제외됩니다."
            roi = frame.get("depth_roi")
            roi_text = (f" / Depth ROI {roi['coverage'] * 100:.1f}%" if roi is not None else "")
            self.window.detectionStatus.setText(
                f"{status} / z={frame['z_min']:.3f}~{frame['z_max']:.3f} m{roi_text} / "
                f"정렬 Depth 유효 픽셀 {coverage:.1f}%")
            text = self.plane_text("Depth", frame["depth_plane"]) + "\n" + self.plane_text("Target", frame["right_plane"])
            if ok and calibration:
                R, t = np.asarray(calibration["R"]), np.asarray(calibration["t"])
                predicted = R @ frame["depth_plane"]["normal"]
                angle = np.degrees(np.arccos(np.clip(predicted @ frame["right_plane"]["normal"], -1, 1)))
                offset = frame["depth_plane"]["h"] + predicted @ t - frame["right_plane"]["h"]
                text += f"\nAfter RT: normal error={angle:.3f} deg, plane error={offset * 1000:+.3f} mm"
            self.window.cornerText.setPlainText(text)
            self.current_stamp = stamp
        except Exception as error:
            LOGGER.exception("frame display failed stamp=%s", stamp)
            self.window.detectionStatus.setText(str(error))

    @staticmethod
    def plane_text(name, plane):
        if plane is None:
            return f"{name} plane: not detected"
        normal, h = plane["normal"], plane["h"]
        quality = (f", fit={plane['rmse'] * 1000:.2f} mm, ratio={plane['ratio']:.2f}"
                   if "ratio" in plane else f", reprojection={plane['rmse']:.3f} px")
        if "size" in plane:
            quality += f", size={plane['size'][0] * 1000:.0f}×{plane['size'][1] * 1000:.0f} mm"
        return (f"{name}: n=[{normal[0]:+.5f}, {normal[1]:+.5f}, {normal[2]:+.5f}], "
                f"h={h:+.5f} m, distance={abs(h):.5f} m{quality}")

    def run_calibration(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.window.calibrateButton.setEnabled(False)
        self.window.calibrationStatus.setText("계산 중...")
        QApplication.processEvents()
        try:
            self.show_result(calibrate(self.frames, self.result_path))
            self.window.calibrationSourceCombo.setCurrentIndex(0)
        except Exception as error:
            LOGGER.exception("RT calibration failed")
            self.window.calibrationStatus.setText(f"실패: {error}")
        finally:
            self.window.calibrateButton.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def save_point_cloud(self):
        item = self.window.fileList.currentItem()
        if item is None:
            self.window.statusbar.showMessage("저장할 Depth 파일을 선택하세요.")
            return
        stamp = item.text()

        default_path = str(
            output_dir("pointclouds") / f"depth_{stamp}.ply"
        )

        LOGGER.info(
            "FILE_DIALOG OPEN start type=save_point_cloud stamp=%s default=%s",
            stamp,
            default_path,
        )

        # path, _ = QFileDialog.getSaveFileName(
        #     self.window, "포인트 클라우드 저장",
        #     str(RESULT_PATH.with_name(f"depth_{stamp}.ply")),
        #     "PLY point cloud (*.ply)")

        # path, selected_filter = QFileDialog.getSaveFileName(
        #     self.window,
        #     "포인트 클라우드 저장",
        #     default_path,
        #     "PLY point cloud (*.ply)",
        # )

        path, selected_filter = QFileDialog.getSaveFileName(
            self.window,
            "대상 영상 색상 포인트 클라우드 저장",
            default_path,
            "PLY point cloud (*.ply)",
            # options=QFileDialog.Option.DontUseNativeDialog,
        )

        LOGGER.info(
            "FILE_DIALOG OPEN returned type=save_point_cloud "
            "stamp=%s path=%r filter=%r",
            stamp,
            path,
            selected_filter,
        )
        
        if not path:
            LOGGER.info(
                "FILE_DIALOG cancelled type=save_point_cloud stamp=%s",
                stamp,
            )
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"

        self.window.savePointCloudButton.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            frame = self.frames.get(stamp) or self.analyze(stamp)
            points, plane = save_point_cloud(path, frame)
            self.window.statusbar.showMessage(
                f"저장 완료: 전체 {points:,} / 평면(초록색) {plane:,} points — {path}")
        except Exception as error:
            LOGGER.exception("Depth point cloud save failed path=%s", path)
            self.window.statusbar.showMessage(f"포인트 클라우드 저장 실패: {error}")
        finally:
            QApplication.restoreOverrideCursor()
            self.window.savePointCloudButton.setEnabled(True)

    def save_ir_point_cloud(self):
        item = self.window.fileList.currentItem()
        calibration = self.active_calibration()

        if item is None or calibration is None:
            self.window.statusbar.showMessage("파일과 RT 결과가 필요합니다.")
            return

        stamp = item.text()

        default_path = str(
            output_dir("pointclouds") / f"target_{stamp}.ply"
        )

        LOGGER.info(
            "FILE_DIALOG OPEN start type=save_ir_point_cloud "
            "stamp=%s default=%s save_thread=%s frames=%d",
            stamp,
            default_path,
            self.save_thread is not None and self.save_thread.isRunning(),
            len(self.frames),
        )

        start_hang_watchdog(20)

        try:
            path, selected_filter = QFileDialog.getSaveFileName(
                self.window,
                "대상 영상 색상 포인트 클라우드 저장",
                default_path,
                "PLY point cloud (*.ply)",
            )
        finally:
            stop_hang_watchdog()

        LOGGER.info(
            "FILE_DIALOG OPEN returned type=save_ir_point_cloud "
            "stamp=%s path=%r filter=%r",
            stamp,
            path,
            selected_filter,
        )

        if not path:
            LOGGER.info(
                "FILE_DIALOG cancelled type=save_ir_point_cloud stamp=%s",
                stamp,
            )
            return

        if not path.lower().endswith(".ply"):
            path += ".ply"

        self.window.saveIrPointCloudButton.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            LOGGER.info(
                "Target point cloud save start stamp=%s path=%s",
                stamp,
                path,
            )

            frame = self.frames.get(stamp) or self.analyze(stamp)

            LOGGER.info(
                "Target point cloud projection/write start stamp=%s",
                stamp,
            )

            points = save_ir_point_cloud(
                path,
                frame,
                calibration,
            )

            LOGGER.info(
                "Target point cloud save done stamp=%s path=%s points=%d",
                stamp,
                path,
                points,
            )

            self.window.statusbar.showMessage(
                f"대상 영상 색상 {points:,} points 저장 완료 — {path}"
            )

        except Exception as error:
            LOGGER.exception(
                "Target point cloud save failed path=%s",
                path,
            )
            self.window.statusbar.showMessage(
                f"IR 포인트 클라우드 저장 실패: {error}"
            )

        finally:
            QApplication.restoreOverrideCursor()
            self.window.saveIrPointCloudButton.setEnabled(True)


    def save_alignment_comparison(self):
        item, calibration = self.window.fileList.currentItem(), self.active_calibration()
        if item is None or calibration is None:
            self.window.statusbar.showMessage("파일과 보정값을 먼저 선택하세요.")
            return
        if self.save_thread is not None and self.save_thread.isRunning():
            self.window.statusbar.showMessage("비교 결과를 이미 저장 중입니다.")
            return
        stamp = item.text()
        folder = QFileDialog.getExistingDirectory(
            self.window, "비교 결과 저장 폴더",
            str(output_dir("comparison")))
        if not folder:
            return
        frame = self.frames.get(stamp)
        if frame is None:
            self.window.statusbar.showMessage("선택한 프레임을 불러올 수 없습니다.")
            return
        source = "factory" if self.window.calibrationSourceCombo.currentIndex() else "measured"
        cached = self.current_stamp == stamp
        custom = self.aligned_depth.copy() if cached and self.aligned_depth is not None else None
        realsense = self.rs_depth.copy() if cached and self.rs_depth is not None else None
        calibration = json.loads(json.dumps(calibration))

        thread = QThread(self)
        worker = ComparisonSaveWorker(folder, stamp, frame.copy(), calibration, source,
                                      custom, realsense)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.comparison_save_progress)
        worker.succeeded.connect(self.comparison_save_succeeded)
        worker.failed.connect(self.comparison_save_failed)
        worker.done.connect(thread.quit)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.comparison_save_finished)
        self.save_thread, self.save_worker = thread, worker
        self.window.saveComparisonButton.setEnabled(False)
        self.window.saveComparisonButton.setText("저장 중...")
        LOGGER.info("comparison save requested stamp=%s source=%s z=%.3f..%.3f "
                    "folder=%s cached_custom=%s cached_rs=%s", stamp, source,
                    frame["z_min"], frame["z_max"], folder,
                    custom is not None, realsense is not None)
        thread.start()

    def comparison_save_progress(self, message):
        self.window.comparisonStatus.setText(message)
        self.window.statusbar.showMessage(message)

    def comparison_save_succeeded(self, metrics, folder):
        mae = metrics.get("mae_mm")
        mae_text = f"{mae:.3f} mm" if mae is not None else "N/A"
        message = (f"비교 저장 완료: IoU {metrics['overlap_iou_percent']:.2f}%, "
                   f"MAE {mae_text} — {folder}")
        self.window.comparisonStatus.setText(message)
        self.window.statusbar.showMessage(message)
        LOGGER.info("comparison save succeeded folder=%s", folder)

    def comparison_save_failed(self, error):
        message = f"비교 저장 실패: {error} / 로그: {LOG_PATH}"
        self.window.comparisonStatus.setText(message)
        self.window.statusbar.showMessage(message)

    def comparison_save_finished(self):
        self.window.saveComparisonButton.setEnabled(True)
        self.window.saveComparisonButton.setText("정렬 비교 결과 저장")
        self.save_thread = self.save_worker = None

    def show_result(self, result):
        self.result = result
        if not self.window.calibrationSourceCombo.currentIndex():
            self.show_matrix(result, "직접 측정값")
            self.refresh_view()

    def show_matrix(self, result, source):
        matrix = np.asarray(result["RT"])
        text = f"{source} RT (Depth → Target)\n" + np.array2string(matrix, precision=9, suppress_small=False)
        if "plane_pairs" in result:
            text += (f"\n\nplanes={result['plane_pairs']}, "
                     f"inliers={result['ransac_inliers']}, normal RMSE={result['normal_rmse_deg']:.3f} deg, "
                     f"plane RMSE={result['plane_rmse_m'] * 1000:.3f} mm, "
                     f"condition={result.get('normal_condition', float('nan')):.1f}")
            if result.get("nonlinear_method"):
                text += f", optimizer={result['nonlinear_method']}"
            if "board_size_m" in result:
                text += (f"\nboard={result['board_size_m'][0] * 1000:.0f} × "
                         f"{result['board_size_m'][1] * 1000:.0f} mm, square={result['square_size_m'] * 1000:.0f} mm")
        elif result.get("source") == "factory":
            text += (f"\n\ndevice={result['device']} / serial={result['serial']}\n"
                     f"target={result['target_stream']} / scale={result['depth_scale']:.9f} m")
        self.window.matrixText.setPlainText(text)
        self.window.calibrationStatus.setText(f"{source} 적용 중")


    def show_aligned_toggled(self, checked):
        LOGGER.info(
            "VIEW toggle show_aligned=%s compare_rs=%s",
            checked,
            self.window.compareRsCheckBox.isChecked(),
        )

        # 현재 방식 정렬 Depth를 선택하면 비교 모드는 해제
        if checked and self.window.compareRsCheckBox.isChecked():
            blocker = QSignalBlocker(self.window.compareRsCheckBox)
            self.window.compareRsCheckBox.setChecked(False)
            del blocker

            LOGGER.info(
                "VIEW compare_rs automatically unchecked"
            )

        self.refresh_view()


    def compare_rs_toggled(self, checked):
        LOGGER.info(
            "VIEW toggle compare_rs=%s show_aligned=%s",
            checked,
            self.window.showAlignedDepthCheckBox.isChecked(),
        )

        if checked:
            # 비교 모드에서는 첫 번째 체크박스가 의미가 없으므로 해제
            if self.window.showAlignedDepthCheckBox.isChecked():
                blocker = QSignalBlocker(
                    self.window.showAlignedDepthCheckBox
                )
                self.window.showAlignedDepthCheckBox.setChecked(False)
                del blocker

                LOGGER.info(
                    "VIEW show_aligned automatically unchecked"
                )

            # 비교 중에는 첫 체크박스 조작 금지
            self.window.showAlignedDepthCheckBox.setEnabled(False)

        else:
            self.window.showAlignedDepthCheckBox.setEnabled(True)
        self.refresh_view()

def main():
    from app import main as launch
    launch('images')


if __name__ == "__main__":
    main()
