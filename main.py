# main.py
# The main entrypoint for the Reward Engineering Pilot experiment.
# This script orchestrates the entire process from setup to completion.

import os
import time

print("--- INSPECTING src/model_caller.py FROM INSIDE THE CONTAINER ---")
try:
    with open("/app/src/model_caller.py", "r", encoding="utf-8") as f:
        print(f.read())
except Exception as e:
    print(f"Could not read the file: {e}")
print("------------------- INSPECTION END -------------------")

from configs import settings
from src import model_caller
from src import dataset_loader, trial_runner, session_manager
from src.logger import MainLogger, TrialLogger

def main():
    """Main function to start and orchestrate the entire experiment."""

    # --- Step 1: Initialization ---
    # Get container-specific info from environment variables.
    sbx_id = int(os.environ.get("SBX_ID", 0))
    model_id = os.environ.get("MODEL_ID", "unknown_model_id") 
    print(f"{sbx_id}, {model_id}") 

    current_state = session_manager.load_state(sbx_id)
    trial = current_state.get("current_trial", 0) + 1 

    MainLogger.log_process_start(sbx_id=sbx_id, start_trial=trial)
    total_start_time = time.time()
    
    # Load the master dataset once for the entire run.
    # split = "train"  # Master
    split = "test"  # Tester 10 Question
    question_dataset = dataset_loader.load_pilot_dataset(split=split)      

    print("Pre-loading local model for this container...")
    local_models = {}
    for model_config in settings.APPLICANT_MODELS:
        if model_config.get("model_id") == model_id:
            model_id_to_load = model_config["model_id"]
            
            print(f"INFO: Found matching model to load: {model_id_to_load}")
            model, tokenizer = model_caller.load_local_model(model_id_to_load)
            
            if model and tokenizer:
                local_models[model_id_to_load] = {"model": model, "tokenizer": tokenizer}
                print(f"INFO: {model_id_to_load} loaded successfully.")
            else:
                print(f"ERROR: {model_id_to_load} failed to load.")
            break

    if not local_models and any(m.get("model_id") == model_id and m.get("type") == "local" for m in settings.APPLICANT_MODELS):
        print(f"WARNING: A local model config was found for '{model_id}' in settings.py, but it failed to load.")

    print("Pre-loading finished.")

    # --- Step 2: Main Experiment Loop ---
    # The loop iterates through the determined range of trials.
    print(f"[DEBUG] TOTAL_RUNS={settings.TOTAL_RUNS}, start_trial={trial}")
    print(f"[DEBUG] Entering loop: range({trial}, {settings.TOTAL_RUNS + 1})")
    for trial_num in range(trial, settings.TOTAL_RUNS + 1): 
        
        # Get all configuration for the current trial from the session manager.
        config = session_manager.next_session(sbx_id, model_id, split)
        
        # Create a dedicated logger instance for this specific trial.
        trial_logger = TrialLogger(config)
        trial_logger.log_event("TRIAL_START")
        trial_start_time = time.time()

        # Branch the workflow based on the condition.
        if config['condition'] in ['A', 'B']:
            try:
                trial_runner.run_realtime_trial(config, question_dataset, trial_logger, local_models)
                trial_logger.log_event("TRIAL_FINISH")
            except Exception as e:
                print(f"[ERROR] trial {trial_num} failed:", e)
        else: # Conditions C, D
            try:
                trial_runner.run_batch_trial(config, question_dataset, trial_logger, local_models)
                trial_logger.log_event("TRIAL_FINISH")
            except Exception as e:
                print(f"[ERROR] trial {trial_num} failed:", e)
        
        # Log the completion and duration of the trial.
        trial_end_time = time.time()
        time_taken = time.strftime("%H:%M:%S", time.gmtime(trial_end_time - trial_start_time))
        trial_logger.log_event("TRIAL_FINISH", {"time_taken": time_taken})

        # Update the state file to mark this trial as successfully completed.
        session_manager.save_state(sbx_id, {"current_trial": trial_num})
        
    # --- Step 3: Finalization ---
    total_end_time = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(total_end_time - total_start_time))
    MainLogger.log_process_finish(sbx_id=sbx_id, total_time=total_time)
    print("\n--- Experiment Finished Successfully ---")

if __name__ == "__main__":
    main()