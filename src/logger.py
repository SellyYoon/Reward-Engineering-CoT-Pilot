# src/logger.py
# Handles all structured logging for the experiment.

import json
import os
from datetime import datetime
from configs import settings
import logging
import sys

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Main Process Logger ---

class MainLogger:
    """A simple logger for high-level process events (start, stop)."""
    LOG_PATH = os.path.join(settings.LOG_DIR, 'main.log')

    @staticmethod
    def _log(event: str, details: dict):
        """Appends a new event to the main log file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **details
        }
        with open(MainLogger.LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    @staticmethod
    def log_process_start(sbx_id: int):
        MainLogger._log("PROCESS_START", {"sbx_id": sbx_id})

    @staticmethod
    def log_process_finish(sbx_id: int, total_time: str):
        MainLogger._log("PROCESS_FINISH", {"sbx_id": sbx_id, "total_time": total_time})

# --- Trial-Specific Data Logger ---

class TrialLogger:
    """
    Logs detailed, structured data for a single trial (e.g., submissions and evaluations).
    Creates filenames based on the session configuration.
    """
    def __init__(self, config: dict):
        """Initializes the logger with the configuration for the current trial."""
        self.config = config
        self.log_dir = settings.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self, role: str) -> str:
        """Generates a log file path based on the user-defined convention."""
        run_id = self.config['run_id']
        session_id = self.config['session_id']
        sbx_id = self.config['sbx_id']
        model_id = self.config.get('model_id', 'unknown').replace("/", "_") # Sanitize name
        if role == 'events':
            filename = f"{run_id}_{sbx_id}_{model_id}_{role}.jsonl"
        else:
            filename = f"{run_id}_{session_id}_{model_id}_{role}.jsonl"
        return os.path.join(self.log_dir, filename)

    def _append_to_file(self, file_path: str, data: dict):
        """Appends a single JSON line to the specified file."""
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    def log_event(self, event: str, details: dict = None):
        """
        Logs a general event for the current trial.
        This method is called by main.py for TRIAL_START and TRIAL_FINISH events.
        """
        if details is None:
            details = {}
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **details,
            "session_id": self.config.get('session_id'),
            "trial_num": self.config.get('trial_num'),
            "model_id": self.config.get('model_id')
        }
        path = self._get_log_path("events") # Save general events to a separate 'events' log
        self._append_to_file(path, log_entry)

    def log_submit(self, submission_data: dict):
        """Logs a raw submission from the solver model."""
        path = self._get_log_path("submit")
        self._append_to_file(path, submission_data)

    def log_eval(self, evaluation_data: dict):
        """Logs the final evaluation and reward results for a turn."""
        path = self._get_log_path("eval")
        self._append_to_file(path, evaluation_data)