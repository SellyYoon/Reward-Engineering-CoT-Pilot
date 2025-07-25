# src/utils.py
# A collection of utility functions for the experiment.

import json
import time
import os
import shutil
from datetime import datetime, timezone
from functools import wraps
import torch
from configs import settings, prompts

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
    It handles Markdown code blocks and extracts the pure JSON.
    Also processes keys to remove leading/trailing spaces.
    """
    # 1. Attempt to remove Markdown code blocks
    json_start_tag = "```json"
    json_end_tag = "```"
    
    if json_start_tag in response_text and json_end_tag in response_text:
        start_index = response_text.find(json_start_tag) + len(json_start_tag)
        end_index = response_text.rfind(json_end_tag)
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response_text[start_index:end_index].strip()
        else:
            json_str = response_text
    else:
        json_str = response_text

    # 2. Find and extract the first ‘{’ and last ‘}’ from a pure JSON string.
    try:
        final_start_index = json_str.find('{')
        final_end_index = json_str.rfind('}') + 1
        
        if final_start_index == -1 or final_end_index == 0:
            error_message = f"Model output did not contain a valid JSON object after markdown parsing. Raw response start: {response_text[:200]}..."
            print(f"Warning: {error_message}")
            return {"pred_answer": error_message, "error": error_message}
        
        pure_json_str = json_str[final_start_index:final_end_index]
        raw_data = json.loads(pure_json_str)
        
        # Process keys to remove leading/trailing spaces
        cleaned_data = {}
        for k, v in raw_data.items():
            cleaned_data[k.strip()] = v
        return cleaned_data
    except json.JSONDecodeError as e:
        error_message = f"JSON parsing failed after extraction. Raw extracted JSON: {pure_json_str[:200]}... Error: {e}"
        print(f"Error: {error_message}")
        return {"pred_answer": error_message, "error": error_message}
    except Exception as e:
        error_message = f"Unexpected error parsing model output: {e}. Raw response start: {response_text[:200]}..."
        print(f"Error: {error_message}")
        return {"pred_answer": error_message, "error": error_message}

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
    
def applicant_system_prompt(condition: str):
	if condition in ("A", "C"):
		return (prompts.SESSION_START_PROMPT + "\n\n" + prompts.CORE_TASK_PROMPT + "\n\n")
	else:
		return (prompts.SESSION_START_PROMPT + "\n\n" + prompts.CORE_TASK_WHW_PROMPT + "\n\n")