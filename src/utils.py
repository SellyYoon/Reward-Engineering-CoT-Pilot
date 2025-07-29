# src/utils.py
# A collection of utility functions for the experiment.

import json
import re
import time
import os
import shutil
import subprocess
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
                    logger.warning(f"Call to {func.__name__} failed. Retrying in {delay}s... ({i+1}/{retries})")
                    logger.error(f"Error: {e}")
                    time.sleep(delay)
            raise ConnectionError(f"Function {func.__name__} failed after {retries} retries.")
        return wrapper
    return decorator

# --- Utility Functions ---

def get_utc_timestamp() -> str:
    """Returns the current timestamp in UTC ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def extract_fields_manually(response_text: str) -> dict:
    """
    A fallback function that extracts key fields using regex when full JSON parsing fails.
    This is the last resort to salvage data from a badly malformed response.
    """
    output = {}
    
    # Try to extract pred_answer from various possible patterns
    patterns = [
        r'["\']pred_answer["\']\s*:\s*["\'](.*?)["\']',
        r'Pred_answer:\s*(.*)',
        r'Answer:\s*(.*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match:
            output["pred_answer"] = match.group(1).strip()
            break

    # Extract pseudocode
    pseudo_match = re.search(r'["\']pred_pseudocode["\']\s*:\s*["\'](.*?)["\']', response_text, re.DOTALL)
    if pseudo_match:
        output["pred_pseudocode"] = pseudo_match.group(1).strip().replace('\n', '\\n')

    # Extract numerical counts
    for key in ["loop_count", "branch_count", "variable_count"]:
        match = re.search(r'["\']?' + key + r'["\']?\s*:\s*(\d+)', response_text)
        if match:
            output[key] = int(match.group(1))

    # If pred_answer is still missing, mark it as an extraction failure.
    if "pred_answer" not in output:
        output["error"] = "Failed to parse JSON and could not extract fields manually."
        output["pred_answer"] = f"Extraction failed. Raw response: {response_text[:150]}..."
        
    return output

def parse_model_json_output(response_text: str) -> dict:
    """
    Safely parses a JSON object from a model's raw text output by attempting a series of cleaning steps.
    """
    
    # Step 1: Isolate the most likely JSON string
    json_str = response_text
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
        else:
            # If no JSON structure is found at all, go straight to manual extraction
            return extract_fields_manually(response_text)

    # Step 2: Clean up common structural errors within the JSON string
    
    # Remove trailing commas from the end of lists or objects
    json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
    
    # Replace malformed values like dictionaries in 'pred_variable_count' with a placeholder number
    json_str = re.sub(r'(["\']pred_variable_count["\']\s*:\s*)\{.*?\}', r'\1 1', json_str, flags=re.DOTALL)

    # Fix unescaped newlines inside specific string values
    def fix_newlines(m):
        return f'"{m.group(1)}": "{m.group(2).replace(chr(10), chr(92)+"n").replace(chr(34), chr(92)+chr(34))}"'
    json_str = re.sub(r'"(pred_pseudocode|pred_reasoning_steps|why|how|which)"\s*:\s*"(.*?)"', fix_newlines, json_str, flags=re.DOTALL)
    
    # Step 3: Try to parse the cleaned JSON
    try:
        data = json.loads(json_str)
        # Clean up keys by stripping whitespace
        return {k.strip(): v for k, v in data.items()}
    except json.JSONDecodeError as e:
        # Step 4: If parsing still fails, fall back to manual regex extraction
        logger.warning(f"JSON parsing failed after cleaning: {e}. Falling back to manual extraction.")
        return extract_fields_manually(response_text)
        
def log_raw_response(context: Dict[str, Any], response_text: str, config: int):
    """
    Safely record the original response of the model to a file before parsing it.
    """
    model_id = config['model_id'].replace("/", "_")
    log_dir = settings.BACKUP_DIR / model_id

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        logger.critical(f"CRITICAL WARNING: Failed to create log directory {log_dir}: {e}")
        return
    timestamp = datetime.now().strftime('%Y%m%d%H')
    log_path = log_dir / f"{config['session_id']}_{timestamp}_responses.jsonl"    
    log_entry = {
        "qid": context.get("QID"),
        "context": context,
        "raw_response": response_text
    }
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.critical(f"CRITICAL WARNING: Failed to write raw response to log file: {e}")
    
def backup(session_id: str, model_id: str):
    """
    Finds all logs for a given session and moves them to a backup directory.
    """
    log_dir = Path(settings.LOG_DIR)
    model_id = model_id.replace("/", "_")
    backup_dir = Path(settings.BACKUP_DIR) / str(model_id)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Sanitize model name for matching
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    prefix = f"{session_id}_{timestamp}"
    logger.info(f"Starting application log backup for prefix: {prefix}")
    for filename in os.listdir(log_dir):
        if filename.startswith(prefix):
            try:
                shutil.move(str(log_dir / filename), str(backup_dir / filename))
            except FileNotFoundError:
                logger.warning(f"Could not find file during move operation: {filename}")
    logger.info(f"Application logs backed up to: {backup_dir}")

# --- Docker Container Log Backup ---
def backup_docker_log(container_name: str):
    
    backup_dir = Path(settings.BACKUP_DIR) / str(container_name)
    os.makedirs(backup_dir, exist_ok=True)
    logger.info(f"Attempting to back up Docker container logs for: {container_name}")
    script_path = "./src/backup_docker_logs.sh"

    # Check if the backup script exists before trying to run it.
    if not os.path.exists(script_path):
        logger.error(f"Docker log backup script not found at: {script_path}")
        return

    try:
        # Execute the shell script using subprocess.run
        result = subprocess.run(
            [script_path, container_name, str(backup_dir)],
            capture_output=True, # Capture the script's stdout and stderr
            text=True,           # Decode output as text
            check=True           # Raise an exception if the script returns a non-zero exit code
        )
        # Log the output from the script for better debugging
        if result.stdout:
            logger.info(f"[Docker Backup Script]:\n{result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"[Docker Backup Script ERROR]:\n{result.stderr.strip()}")

    except FileNotFoundError:
        logger.error(f"Script '{script_path}' not found. Ensure it has execute permissions (chmod +x).")
    except subprocess.CalledProcessError as e:
        # This block runs if the script fails (returns a non-zero exit code)
        logger.error(f"Docker log backup script failed with exit code {e.returncode}:")
        logger.error(f"STDOUT: {e.stdout.strip()}")
        logger.error(f"STDERR: {e.stderr.strip()}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Docker log backup: {e}")
            
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
    logger.info("--- Performing Environment Check ---")
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