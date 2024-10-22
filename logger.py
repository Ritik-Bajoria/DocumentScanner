import os
import sys
import threading
from datetime import datetime

class Logger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking to ensure thread safety
                    cls._instance = super().__new__(cls)
                    cls._instance.log_file_path = os.path.join('.', 'combined.log')
                    cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        self.info(f"App started on process id: {os.getpid()}")

    def _log(self, message, level):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"{timestamp} [{level}]: {message}\n"

        try:
            with open(self.log_file_path, 'a') as log_file:
                log_file.write(log_message)
        except IOError as e:
            print(f"Error writing to log file: {e}", file=sys.stderr)

    def info(self, message):
        self._log(message, 'INFO')

    def warn(self, message):
        self._log(message, 'WARN')

    def error(self, message):
        self._log(message, 'ERROR')
