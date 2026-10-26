"""One entry point, two isolated workflows. Existing controllers stay reusable."""
import argparse
import sys
from PySide6.QtCore import QObject, QEvent, QSettings, QSignalBlocker, QTimer, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid
from app_paths import APP_DIR
from app_logging import get_logger, install_exception_hook

LOGGER = get_logger('workspace')
MODES = ('images', 'pointcloud')


def create_controller(mode):
    if mode == 'images':
        from calibration_gui import Controller
    elif mode == 'pointcloud':
        from nsl_color_gui import Controller
    else:
        raise ValueError(f'Unknown workflow: {mode}')
    return Controller()


class Workspace(QObject):
    def __init__(self, mode=None, settings_path=None, factory=create_controller, auto_open=True):
        super().__init__()
        self.window = QUiLoader().load(str(APP_DIR/'workspace.ui'))
        if self.window is None:
            raise RuntimeError('workspace.ui load failed')
        self.settings = QSettings(str(settings_path or APP_DIR/'workspace_settings.ini'), QSettings.IniFormat)
        self.factory, self.controllers = factory, {}
        self.active = None
        self.opening = False
        self.closed = False
        mode = mode or self.settings.value('workflow', 'images')
        index = MODES.index(mode) if mode in MODES else 0
        self.window.workflowCombo.setCurrentIndex(index)
        self.window.workflowCombo.currentIndexChanged.connect(self.switch_mode)
        self.window.installEventFilter(self)
        if auto_open:
            QTimer.singleShot(0, lambda: self.switch_mode(index))

    def busy(self):
        return self.opening or any(c.is_busy() for c in self.controllers.values())

    def reset_selection(self):
        with QSignalBlocker(self.window.workflowCombo):
            self.window.workflowCombo.setCurrentIndex(MODES.index(self.active) if self.active else 0)

    def switch_mode(self, index):
        if self.closed or not isValid(self.window):
            return
        mode = MODES[index]
        if self.active == mode:
            return
        if self.busy():
            self.reset_selection()
            self.window.statusbar.showMessage('현재 분석·계산·저장이 끝나면 작업 환경을 전환할 수 있습니다.', 5000)
            return
        self.opening = True
        self.window.workflowCombo.setEnabled(False)
        self.window.statusbar.showMessage('작업 환경 불러오는 중…')
        QApplication.processEvents()
        try:
            if mode not in self.controllers:
                controller = self.factory(mode)
                controller.window.setWindowFlags(Qt.Widget)
                self.window.workspaceStack.addWidget(controller.window)
                if mode == 'images':
                    # Superseded by the shared selector; standalone entry still works.
                    controller.window.openNslColorButton.hide()
                self.controllers[mode] = controller
            self.window.workspaceStack.setCurrentWidget(self.controllers[mode].window)
            self.active = mode
            with QSignalBlocker(self.window.workflowCombo):
                self.window.workflowCombo.setCurrentIndex(index)
            self.settings.setValue('workflow', mode)
            self.settings.sync()
            self.window.statusbar.clearMessage()
            LOGGER.info('workflow selected mode=%s', mode)
        except Exception as error:
            self.reset_selection()
            LOGGER.exception('workflow open failed mode=%s', mode)
            self.window.statusbar.showMessage(f'작업 환경 열기 실패: {error}')
        finally:
            self.opening = False
            self.window.workflowCombo.setEnabled(True)

    def eventFilter(self, obj, event):
        if not isValid(self.window):
            return False  # Qt may deliver teardown events after deleting the window.
        if obj == self.window and event.type() == QEvent.Close:
            if self.busy():
                event.ignore()
                self.window.statusbar.showMessage('분석·계산·저장이 끝난 후 종료할 수 있습니다.', 5000)
                return True
            self.closed = True
            for controller in self.controllers.values():
                controller.window.close()
            if 'images' in self.controllers:
                from realsense_compare import close_rs_align_sessions
                close_rs_align_sessions()
        return super().eventFilter(obj, event)


def main(initial_mode=None):
    parser = argparse.ArgumentParser(description='Cam_align 작업 환경 선택')
    parser.add_argument('--mode', choices=MODES, default=initial_mode,
                        help='images: Depth→IR/Color, pointcloud: NSL→Color')
    args = parser.parse_args()
    install_exception_hook(LOGGER)
    app = QApplication(sys.argv)
    workspace = Workspace(args.mode)
    workspace.window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
