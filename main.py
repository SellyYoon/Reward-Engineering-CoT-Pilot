# main.py
# The main entrypoint for the Reward Engineering Pilot experiment.
# This script orchestrates the entire process from setup to completion.

import os
import time
from configs import settings
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
    master_dataset = dataset_loader.load_master_dataset()

    # --- Step 2: Main Experiment Loop ---
    # The loop iterates through the determined range of trials.
    for trial_num in range(trial, settings.TOTAL_RUNS + 1): 
        
        # Get all configuration for the current trial from the session manager.
        config = session_manager.next_session(sbx_id, model_id)
        
        # Create a dedicated logger instance for this specific trial.
        trial_logger = TrialLogger(config)
        trial_logger.log_event("TRIAL_START")
        trial_start_time = time.time()

        # Branch the workflow based on the condition.
        if config['condition'] in ['A', 'B']:
            trial_runner.run_realtime_trial(config, master_dataset, trial_logger)
        else: # Conditions C, D
            trial_runner.run_batch_trial(config, master_dataset, trial_logger)
        
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