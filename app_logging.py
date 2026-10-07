import logging
import sys
from logging.handlers import RotatingFileHandler
import faulthandler
from app_paths import output_dir


LOG_PATH = output_dir("logs") / "cam_align.log"


def get_logger(name):
    root = logging.getLogger("cam_align")
    if not root.handlers:
        LOG_PATH.parent.mkdir(exist_ok=True)
        handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3,
                                      encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    return root.getChild(name)


def install_exception_hook(logger):
    original = sys.excepthook

    def hook(kind, value, traceback):
        logger.critical("Unhandled exception", exc_info=(kind, value, traceback))
        original(kind, value, traceback)

    sys.excepthook = hook


HANG_LOG_PATH = LOG_PATH.with_name("cam_align_hang.log")

_hang_file = None


def start_hang_watchdog(timeout=20):
    global _hang_file

    if _hang_file is None:
        HANG_LOG_PATH.parent.mkdir(exist_ok=True)

        _hang_file = open(
            HANG_LOG_PATH,
            "a",
            encoding="utf-8",
            buffering=1,
        )

        faulthandler.enable(
            file=_hang_file,
            all_threads=True,
        )

    faulthandler.dump_traceback_later(
        timeout,
        repeat=True,
        file=_hang_file,
    )


def stop_hang_watchdog():
    faulthandler.cancel_dump_traceback_later()
