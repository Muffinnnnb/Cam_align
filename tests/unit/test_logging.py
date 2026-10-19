import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app_logging import ProcessSafeRotatingHandler


class LoggingTests(unittest.TestCase):
    def test_locked_shared_log_falls_back_without_losing_record(self):
        with TemporaryDirectory() as folder:
            handler = ProcessSafeRotatingHandler(Path(folder)/'cam_align.log', maxBytes=1, backupCount=3)
            handler.emit(logging.makeLogRecord({'msg': 'initial record'}))
            with patch('logging.handlers.RotatingFileHandler.doRollover', side_effect=PermissionError('locked')):
                handler.emit(logging.makeLogRecord({'msg': 'record after rollover failure'}))
            handler.close()
            text = (Path(folder)/f'cam_align.{os.getpid()}.log').read_text()
            self.assertIn('Shared log locked', text)
            self.assertIn('record after rollover failure', text)


if __name__ == '__main__':
    unittest.main()
