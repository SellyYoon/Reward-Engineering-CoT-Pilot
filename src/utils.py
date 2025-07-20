# src/utils.py
# A collection of utility functions for the experiment.

import json
import time
import os
import shutil
from datetime import datetime, timezone
from functools import wraps
import torch
from configs import settings

# --- Decorator for Retry Logic ---

def retry(retries=3, delay=5):
    """
    A decorator to retry a function call upon failure.
    Usage:
    @retry(retries=3, delay=5)
    def my_api_call():
        # ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Call to {func.__name__} failed. Retrying in {delay}s... ({i+1}/{retries})")
                    print(f"Error: {e}")
                    time.sleep(delay)
            raise ConnectionError(f"Function {func.__name__} failed after {retries} retries.")
        return wrapper
    return decorator

# --- Utility Functions ---

def get_utc_timestamp() -> str:
    """Returns the current timestamp in UTC ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def parse_model_json_output(response_text: str) -> dict:
    """
    Safely parses a JSON object from a model's raw text output.
    It finds the first '{' and the last '}' to extract the JSON block.
    """
    try:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            return {"error": "No JSON object found in the response."}
        
        json_str = response_text[start_index:end_index]
        return json.loads(json_str)
    except Exception as e:
        return {"error": "Failed to parse JSON from model output.", "details": str(e)}

def backup(session_id: str, model_name: str):
    """
    Finds all logs for a given session and moves them to a backup directory.
    """
    log_dir = settings.LOG_DIR
    backup_dir = os.path.join(settings.BACKUP_DIR, session_id)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Sanitize model name for matching
    safe_model_name = model_name.replace("/", "_")
    
    prefix = f"{session_id}_{safe_model_name}"
    
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix):
            shutil.move(os.path.join(log_dir, filename), os.path.join(backup_dir, filename))
    print(f"Logs for session {session_id} have been backed up to {backup_dir}")

def clear_caches():
    """
    Clears Python-level LRU caches and GPU caches.
    Note: LRU caches on specific functions must be cleared from the main script.
    """
    # 1. Clear GPU Cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch GPU cache cleared.")
    except Exception as e:
        print(f"Could not clear GPU cache: {e}")

    # 2. LRU Cache Clearing (Informational)
    # LRU caches are bound to functions. They must be cleared where the functions are accessible.
    # Example (in main.py or trial_runner.py):
    #   from src import model_caller
    #   model_caller.load_local_model.cache_clear()
    print("LRU caches should be cleared directly on the decorated functions in the main script.")

def check_environment():
    """
    Performs a basic check for necessary environment variables and hardware.
    """
    print("--- Performing Environment Check ---")
    # Check for API keys
    if not (settings.OPENAI_API_KEY and settings.ANTHROPIC_API_KEY):
        print("Warning: One or more API keys are missing from the .env file.")
    else:
        print("API keys found.")
        
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU found. Local model inference will be very slow.")
    print("--- Environment Check Complete ---")