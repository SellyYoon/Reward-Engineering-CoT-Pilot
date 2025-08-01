# main.py
# The main entrypoint for the Reward Engineering Pilot experiment.
# This script orchestrates the entire process from setup to completion.

import os
import time
import logging
import sys
from datetime import datetime

print("--- INSPECTING src/model_caller.py FROM INSIDE THE CONTAINER ---")
try:
    with open("/app/src/model_caller.py", "r", encoding="utf-8") as f:
        print(f.read())
except Exception as e:
    print(f"Could not read the file: {e}")
print("------------------- INSPECTION END -------------------")

from configs import settings
from src import model_caller, dataset_loader, trial_runner, session_manager, utils, trial_runner
from src.logger import MainLogger, TrialLogger
import secrets

# 1. Container Global Logger
sbx_id = os.getenv('SBX_ID', 'unknown')
model_id = os.getenv('MODEL_ID', 'unknown').replace("/", "_")
container_name = f"{sbx_id}_{model_id}"
log_file_path = settings.LOG_DIR / f"app_{container_name}_{datetime.utcnow().strftime('%Y%m%d')}.log"

run_id = secrets.token_hex(3)  # ex: 'a4f2c1'
logging.info(f"✅ New execution started. This run's unique ID is: [ {run_id} ]")
    
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ],
    force=True
)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if '##########' in line or '%' in line:
                self.logger.log(logging.INFO, line.rstrip())
            else:
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        for h in self.logger.handlers:
            h.flush()

# Redirect stdout and stderr output to a logger
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
# sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

logging.info("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
logging.info("■■■■■■■■■■■ START: Reward Engineering example to CoT Reward Hacking Pilot. By Selly Yoon ■■■■■■■■■■■")
logging.info("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")

logging.info(f"--- Global logger initialized for container: {container_name} ---")


def main():
    """Main function to start and orchestrate the entire experiment."""

    # --- Step 1: Initialization ---
    # Get container-specific info from environment variables.
    sbx_id = int(os.environ.get("SBX_ID", 0))
    model_id = os.environ.get("MODEL_ID", "unknown_model_id") 
    print(f"{sbx_id}, {model_id}") 

    current_state = session_manager.load_state(sbx_id)
    trial = current_state.get("current_trial", 0) + 1 

    MainLogger.log_process_start(sbx_id=sbx_id)
    total_start_time = time.time()
    
    # Load the master dataset once for the entire run.
    split = "train"  # Master
    # split = "test"  # Tester 10 Question
    question_dataset = dataset_loader.load_pilot_dataset(split=split)      

    local_models = {}
    for model_config in settings.APPLICANT_MODELS:
        if model_config.get("model_id") == model_id:
            model_id_to_load = model_config["model_id"]
            if model_config.get("type") == "local":
                logging.info("Pre-loading local model for this container...")
                logging.info(f"Found matching model to load: {model_id_to_load}")
                model, tokenizer = model_caller.load_local_model(model_id_to_load)
                if model and tokenizer:
                    local_models[model_id_to_load] = {"model": model, "tokenizer": tokenizer}
                    logging.info(f"{model_id_to_load} loaded successfully.")
                else:
                    logging.error(f"{model_id_to_load} failed to load.")
                break

    if not local_models and any(m.get("model_id") == model_id and m.get("type") == "local" for m in settings.APPLICANT_MODELS):
        logging.warning(f"A local model config was found for '{model_id}' in settings.py, but it failed to load.")

    logging.info("Pre-loading finished.")

    # --- Step 2: Main Experiment Loop ---
    # The loop iterates through the determined range of trials.
    logging.debug(f"TOTAL_RUNS={settings.TOTAL_RUNS}, start_trial={trial}")
    logging.debug(f"Entering loop: range({trial}, {settings.TOTAL_RUNS})")
    for trial_num in range(trial, settings.TOTAL_RUNS + 1): 
        
        # Get all configuration for the current trial from the session manager.
        config = session_manager.next_session(sbx_id, model_id, split)
        config['run_id'] = run_id
        
        # Create a dedicated logger instance for this specific trial.
        trial_logger = TrialLogger(config)
        trial_logger.log_event("TRIAL_START")
        trial_start_time = time.time()

        # Branch the workflow based on the condition.
        if config['condition'] in ['A', 'B']:
            try:
                trial_runner.run_realtime_trial(config, question_dataset, trial_logger, local_models)
            except Exception as e:
                logging.error(f"trial {trial_num} failed: {e}")
        else: # Conditions C, D
            try:
                trial_runner.run_batch_trial(config, question_dataset, trial_logger, local_models)
            except Exception as e:
                logging.error(f"trial {trial_num} failed: {e}")
        
        # Log the completion and duration of the trial.
        trial_end_time = time.time()
        time_taken = time.strftime("%H:%M:%S", time.gmtime(trial_end_time - trial_start_time))
        trial_logger.log_event("TRIAL_FINISH", {"time_taken": time_taken})

        # Update the state file to mark this trial as successfully completed.
        session_manager.save_state(sbx_id, {"current_trial": trial_num})
        
        # if not container_name:
        #     MainLogger._log("The environment variable cannot be found. Please check the docker-compose.yml file.", {"CONTAINER_NAME": container_name})
        #     return
        
        # utils.backup_docker_log(
        #     session_id=config['session_id'],
        #     model_id=config['model_id'],
        #     container_name=container_name
        # )
        
    # --- Step 3: Finalization ---
    total_end_time = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(total_end_time - total_start_time))
    MainLogger.log_process_finish(sbx_id=sbx_id, total_time=total_time)
    logging.info("\n--- Experiment Finished Successfully ---")

logging.info("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")
logging.info("■■■■■■■■■■■ FINISH: Reward Engineering example to CoT Reward Hacking Pilot. By Selly Yoon ■■■■■■■■■■■")
logging.info("■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■")

if __name__ == "__main__":
    main()