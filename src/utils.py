# src/utils.py
# A collection of utility functions for the experiment.

import json
import random
import re
import time
import os
import shutil
import subprocess
import logging
import sys
import httpx
import torch
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, Union, Callable
from xai_sdk.chat import Response as XaiResponse
from configs import settings, prompts
from pydantic import BaseModel

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Decorator for Retry Logic ---
def retry_with_backoff(
    api_call_func: Callable[..., Any],
    api_kwargs: Dict[str, Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
) -> str:
    """
    Receives an API call function and retries it using an exponential backoff mechanism until it succeeds.
    Args:
    api_call_func: The function that actually calls the API (e.g., _call_openai_api).
    api_kwargs: A dictionary of arguments to be passed to api_call_func.
            
    max_retries: Maximum number of retries.
            initial_delay: Wait time (in seconds) after the first failure.
    backoff_factor: Value multiplied by the wait time for each retry.
    Returns:
    API response string on success, JSON string containing the error on final failure.
    """
    model_type = api_kwargs.get('config', {}).get('type', 'unknown_type')
    model_id = api_kwargs.get('config', {}).get('model_id', 'unknown_model')

    for attempt in range(max_retries):
        try:
            result = api_call_func(**api_kwargs)
            if result and isinstance(result, str):
                return result
            raise ValueError("API returned an empty but non-error response.")
        except httpx.HTTPStatusError as e:
            # Retry only on 5xx server errors or 408, 429 rate limit errors
            if e.response.status_code >= 500 or e.response.status_code in [408, 429]:
                if attempt < max_retries - 1:
                    delay = min(max_delay, initial_delay * (backoff_factor ** attempt))
                    jitter = random.uniform(0, delay * 0.1) # Add 10% jitter
                    sleep_time = delay + jitter
                    logging.warning(f"TYPE: {model_type}, ID: {model_id} API: Status {e.response.status_code}. Retrying in {sleep_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    logging.critical(f"Error calling {model_type} for model {model_id}: Max retries exceeded. {e}")
                    return json.dumps({"error": f"TYPE: {model_type}, ID: {model_id} API call failed after {max_retries} retries: {str(e)}"})
            else:
                # Other HTTP errors (e.g., 4xx client errors) are not retried
                logging.error(f"TYPE: {model_type}, ID: {model_id} API: Client error that cannot be retried (HTTP Status: {e.response.status_code}). Please check your request. Error: {e}")
                return json.dumps({"error": f"TYPE: {model_type}, ID: {model_id} API call failed: {str(e)}"})
            
        except Exception as e:
            logger.error(f"'TYPE: {model_type}, ID: {model_id}' API call failed (attempted {attempt + 1}/{max_retries}): {e}")
            
            if attempt + 1 == max_retries:
                logger.critical(f"The 'TYPE: {model_type}, ID: {model_id}' API failed all retries.")
                return json.dumps({"error": f"API call failed for TYPE: {model_type}, ID: {model_id} after {max_retries} retries."})
            
            delay = min(max_delay, initial_delay * (backoff_factor ** attempt))
            logger.info(f"{delay:.1f} seconds later, we will try again...")
            time.sleep(delay)
            
    return json.dumps({f"error": "TYPE: {model_type}, ID: {model_id} API Exhausted all retries."})

# --- Utility Functions ---

def get_utc_timestamp() -> str:
    """Returns the current timestamp in UTC ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def extract_fields_manually(response_content: str) -> dict:
    """
    A fallback function that extracts key fields using regex when full JSON parsing fails.
    This is the last resort to salvage data from a badly malformed response.
    """
    if not isinstance(response_content, str):
        return {}
        
    data = {}

    # Regular expression pattern helper functions
    def extract_value(key, pattern, default=None, is_bool=False, is_numeric=False, is_list=False):
        match = re.search(f'["\']{key}["\']\\s*:\\s*({pattern})', response_content, re.DOTALL)
        if not match:
            return default
        
        val = match.group(1).strip()

        if is_bool:
            return 'true' in val.lower()
        if is_numeric:
            numeric_val = re.search(r'\d+', val)
            return int(numeric_val.group(0)) if numeric_val else default
        if is_list:
            items_str = val.strip('[]').strip()
            items = [item.strip().strip('"\'') for item in items_str.split(',') if item.strip()]
            return items
            
        # 일반 텍스트 값은 앞뒤 따옴표와 공백 제거
        return val.strip().strip('"\'')

    # --- 1. Solver model field extraction ---
    data['pred_answer'] = extract_value('pred_answer', '".*?"')
    data['pred_reasoning_steps'] = extract_value('pred_reasoning_steps', r'\[.*?\]', default=[], is_list=True)
    data['pred_pseudocode'] = extract_value('pred_pseudocode', '".*?"')
    data['pred_loop_count'] = extract_value('pred_loop_count', r'\d+', default=0, is_numeric=True)
    data['pred_branch_count'] = extract_value('pred_branch_count', r'\d+', default=0, is_numeric=True)
    data['pred_variable_count'] = extract_value('pred_variable_count', r'\d+', default=0, is_numeric=True)

    # --- 2. Eval model field extraction ---
    data['coherence'] = extract_value('coherence', 'true|false', default=False, is_bool=True)
    data['rpg'] = extract_value('rpg', 'true|false', default=False, is_bool=True)
    data['eval_comment'] = extract_value('eval_comment', '".*?"')
    
    # --- 3. Handling nested ‘whw’ and ‘question’ objects ---
    whw_block = extract_value('whw_description', r'\{.*?\}', default="{}")
    whw_data = {
        'why': extract_value('why', '".*?"', default=""),
        'how': extract_value('how', '".*?"', default=""),
        'which': extract_value('which', '".*?"', default="")
    }

    if whw_block:
         whw_data['why'] = extract_value('why', '".*?"', default="") or whw_data['why']
         whw_data['how'] = extract_value('how', '".*?"', default="") or whw_data['how']
         whw_data['which'] = extract_value('which', '".*?"', default="") or whw_data['which']

    if any(whw_data.values()):
        data['whw_description'] = whw_data
    
    return {k: v for k, v in data.items() if v is not None}

def parse_model_json_output(response_content: str) -> dict:
    """
    Safely parses a JSON object from a model's raw text output by attempting a series of cleaning steps.
    """
    
    if isinstance(response_content, dict):
        return {str(k).strip(): v for k, v in response_content.items()}
    if not isinstance(response_content, str):
        logger.warning(f"Since the input value is not a string (type: {type(response_content)}), an empty dictionary is returned.")
        return {}
    
    # Step 1: Isolate the most likely JSON string
    json_str = response_content
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = response_content.find('{')
        end = response_content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_content[start:end+1]
        else:
            # If no JSON structure is found at all, go straight to manual extraction
            return extract_fields_manually(response_content)

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
        return extract_fields_manually(response_content)
        
def log_raw_response(context: Dict[str, Any], response_content: Union[str, Dict[str, Any], BaseModel, Any], config: Dict[str, Any]):
    """
    Safely record the original response of the model to a file before parsing it.
    """
    if not isinstance(config, dict):
        logger.critical(f"CRITICAL WARNING: 'config' parameter is not a dictionary: {type(config)}. Skipping log.")
        return
    
    model_id = config['model_id'].replace("/", "_")
    log_dir = Path(settings.BACKUP_DIR) / model_id

    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        logger.critical(f"CRITICAL WARNING: Failed to create log directory {log_dir}: {e}")
        return
    timestamp = datetime.now().strftime('%Y%m%d%H')
    log_path = log_dir / f"{config['session_id']}_{timestamp}_responses.jsonl"    
    
        # Check type of response_content and convert to string if it's a dictionary
    serialized_response_content: str
    try:
        if isinstance(response_content, str):
            serialized_response_content = response_content
        elif isinstance(response_content, BaseModel):
            # Pydantic BaseModel objects have .model_dump_json()
            serialized_response_content = response_content.model_dump_json()
        elif hasattr(response_content, 'model_dump_json') and callable(getattr(response_content, 'model_dump_json')):
            # For OpenAI objects
            serialized_response_content = response_content.model_dump_json()
        elif hasattr(response_content, 'to_json') and callable(getattr(response_content, 'to_json')):
            # For Anthropic objects (which have .to_json())
            serialized_response_content = response_content.to_json()
        elif hasattr(response_content, 'to_dict') and callable(getattr(response_content, 'to_dict')):
            # For Google Gemini objects (which have .to_dict())
            serialized_response_content = json.dumps(response_content.to_dict(), ensure_ascii=False)
        elif isinstance(response_content, XaiResponse): # Use the imported XaiResponse type
            serialized_response_content = response_content.content # Grok's raw JSON is in .content
        elif isinstance(response_content, dict):
            serialized_response_content = json.dumps(response_content, ensure_ascii=False)
        else:
            # Fallback for any other unhandled object types
            logger.warning(f"Unhandled response_content type for logging: {type(response_content)}. Attempting str conversion.")
            serialized_response_content = str(response_content) # Fallback to string representation

    except Exception as e:
        logger.critical(f"CRITICAL WARNING: Failed to serialize response_content for logging: {e}. Type: {type(response_content)}")
        serialized_response_content = f"ERROR_SERIALIZING_RESPONSE: {str(e)} - Original Type: {type(response_content)}"

    log_entry = {
        "qid": context.get("QID"),
        "context": context,
        "raw_response": serialized_response_content
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