# src/session_manager.py
import json
import os
from configs import settings

BACKUP_DIR = settings.BACKUP_DIR
STATE_FILE = os.path.join(BACKUP_DIR, f"{os.environ.get('MODEL_ID', 'unknown_model_id')}_session.json")

def load_state(sbx_id: int) -> dict:
    """Load the state for a specific SBX_ID from a file."""
    state_file = os.path.join(settings.BACKUP_DIR, f"session_state_sbx{sbx_id}.json")
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    return {"current_trial": 0} # Return default if no state file exists

def save_state(sbx_id: int, state: dict):
    """Save the state for a specific SBX_ID to a file."""
    state_file = os.path.join(settings.BACKUP_DIR, f"session_state_sbx{sbx_id}.json")
    with open(state_file, "w") as f:
        json.dump(state, f)

def next_session(sbx_id: int, model_id: str) -> dict:
    """
    Sets up the configuration for the next trial.
    - Increments the trial number.
    - Creates a unique session ID.
    - Determines the condition.
    - Returns all config as a dictionary.
    """
    state = load_state(sbx_id)
    
    # Increment trial number and save the new state
    trial = state["current_trial"] + 1
    save_state(sbx_id, {"current_trial": trial})

    # Create session_id (e.g., SBX 1, Trial 5 -> "105")
    session_id = sbx_id * 100 + trial
    
    # Determine condition (A,B,C,D)
    condition = settings.determine_condition(trial)
    
    return {
        "session_id": session_id,
        "sbx_id": sbx_id,
        "trial_num": trial,
        "model_id": model_id,
        "condition": condition,
        "seed": settings.DEFAULT_SEED
    }
