# main.py
# The main entrypoint for the Reward Engineering Pilot experiment.
# This script orchestrates the entire process from setup to completion.

import os
import time
import argparse
from configs import settings
from src import dataset_loader, trial_runner, session_manager
from src.logger import MainLogger, TrialLogger

def main():
    """Main function to start and orchestrate the entire experiment."""
    
    # --- Step 1: Initialization ---
    # Get container-specific info from environment variables.
    sbx_id = int(os.environ.get("SBX_ID", 0))
    model_name = os.environ.get("MODEL_NAME", "unknown_model")
    
    # Handle command-line arguments to allow overriding the start trial.
    parser = argparse.ArgumentParser(description="Run the Reward Engineering Pilot experiment.")
    parser.add_argument("--start-trial", type=int, default=None, help="Force start from a specific trial.")
    args = parser.parse_args()

    # Determine the trial to start from (either from saved state or command line).
    start_from_state = session_manager.get_start_trial(sbx_id)
    start_trial = args.start_trial if args.start_trial is not None else start_from_state
    
    # Log the main process start event.
    MainLogger.log_process_start(sbx_id=sbx_id, start_trial=start_trial)
    total_start_time = time.time()
    
    # Load the master dataset once for the entire run.
    master_dataset = dataset_loader.load_master_dataset()

    # --- Step 2: Main Experiment Loop ---
    # The loop iterates through all trials defined in the settings.
    for trial_num in range(start_trial, settings.TOTAL_RUNS + 1):
        
        # Get all configuration for the current trial from the session manager.
        config = session_manager.setup_trial_session(sbx_id, model_name, trial_num)
        
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
        session_manager.save_trial_completion(sbx_id, trial_num)
        
    # --- Step 3: Finalization ---
    total_end_time = time.time()
    total_time = time.strftime("%H:%M:%S", time.gmtime(total_end_time - total_start_time))
    MainLogger.log_process_finish(sbx_id=sbx_id, total_time=total_time)
    print("\n--- Experiment Finished Successfully ---")

if __name__ == "__main__":
    main()