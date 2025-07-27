# src/utils.py
# A collection of utility functions for the experiment.

import json
import re
import time
import os
import shutil
import logging
import sys
import torch
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict
from configs import settings, prompts

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    json_str = response_text

    # 1. Attempt to extract content from Markdown code blocks first
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # If no markdown block, try to find the first and last curly braces
        final_start_index = json_str.find('{')
        final_end_index = json_str.rfind('}') + 1

        if final_start_index != -1 and final_end_index != 0 and final_end_index > final_start_index:
            json_str = json_str[final_start_index:final_end_index].strip()
        else:
            # If still no clear JSON, return an error.
            error_message = f"Model output did not contain a valid JSON object or markdown block. Raw response start: {response_text}..."
            print(f"Warning: {error_message}")
            return {"pred_answer": error_message, "error": error_message}

    # 2. Handling line breaks within key-value pairs
    def fix_newlines(match):
        key = match.group(1)
        value = match.group(2)
        fixed_value = value.replace('\n', '\\n').replace('"', '\\"')
        return f'"{key}": "{fixed_value}"'
    
    json_str = re.sub(r'"(pred_pseudocode|why|how|which)":\s*"(.*?)"', fix_newlines, json_str, flags=re.DOTALL)


    try:
        raw_data = json.loads(json_str)

        # Process keys to remove leading/trailing spaces
        cleaned_data = {}
        for k, v in raw_data.items():
            cleaned_data[k.strip()] = v

        return cleaned_data

    except json.JSONDecodeError as e:
        error_message = f"JSON parsing failed after extraction. Raw extracted JSON: {json_str}... Error: {e}"
        print(f"Error: {error_message}")
        return {"pred_answer": error_message, "error": error_message}
    except Exception as e:
        error_message = f"Unexpected error parsing model output: {e}. Raw response start: {response_text}..."
        print(f"Error: {error_message}")
        return {"pred_answer": error_message, "error": error_message}
    
def log_raw_response(context: Dict[str, Any], response_text: str, config: int):
    """
    Safely record the original response of the model to a file before parsing it.
    """
    model_id = config['model_id'].replace("/", "_")
    log_dir = settings.BACKUP_DIR / model_id

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"CRITICAL WARNING: Failed to create log directory {log_dir}: {e}")
        return

    log_path = log_dir / f"{config['session_id']}_responses.jsonl"    
    log_entry = {
        "qid": context.get("QID"),
        "context": context,
        "raw_response": response_text
    }
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"CRITICAL WARNING: Failed to write raw response to log file: {e}")
    
def backup(session_id: str, model_id: str):
    """
    Finds all logs for a given session and moves them to a backup directory.
    """
    log_dir = settings.LOG_DIR
    backup_dir = os.path.join(settings.BACKUP_DIR, session_id)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Sanitize model name for matching
    safe_model_name = model_id.replace("/", "_")
    
    prefix = f"{session_id}_{safe_model_name}"
    
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix):
            shutil.move(os.path.join(log_dir, filename), os.path.join(backup_dir, filename))
    logger.info(f"Logs for session {session_id} have been backed up to {backup_dir}")

def clear_caches():
    """
    Clears Python-level LRU caches and GPU caches.
    Note: LRU caches on specific functions must be cleared from the main script.
    """
    # 1. Clear GPU Cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("PyTorch GPU cache cleared.")
    except Exception as e:
        logger.error(f"Could not clear GPU cache: {e}")

    # 2. LRU Cache Clearing (Informational)
    from src import model_caller
    model_caller.load_local_model.cache_clear()
    logger.info("LRU caches should be cleared directly on the decorated functions in the main script.")

def check_environment():
    """
    Performs a basic check for necessary environment variables and hardware.
    """
    logger("--- Performing Environment Check ---")
    # Check for API keys
    if not (settings.OPENAI_API_KEY and settings.ANTHROPIC_API_KEY):
        logger.warning("Warning: One or more API keys are missing from the .env file.")
    else:
        logger.info("API keys found.")
        
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"GPU found: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("Warning: No GPU found. Local model inference will be very slow.")
    logger.info("--- Environment Check Complete ---")
    
def applicant_system_prompt(condition: str):
	if condition in ("A", "C"):
		return (prompts.SESSION_START_PROMPT + "\n\n" + prompts.CORE_TASK_PROMPT + "\n\n")
	else:
		return (prompts.SESSION_START_PROMPT + "\n\n" + prompts.CORE_TASK_WHW_PROMPT + "\n\n")