# configs/settings.py
# All hyperparameters and configurations for the Reward Engineering Pilot experiment.
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Call API key via env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT.parent / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Model configuration
APPLICANT_MODELS = [
    {"sbx_id": 1, "name": "gpt-4.1", "model_id": "gpt-4.1-2025-04-14", "type": "api_openai"},
    {"sbx_id": 2, "name": "claude-sonnet-4", "model_id": "claude-sonnet-4-20250514", "type": "api_anthropic"},
    {"sbx_id": 3, "name": "gemini-2.5-pro", "model_id": "gemini-2.5-pro", "type": "api_google"},
    {"sbx_id": 4, "name": "llama3-8b", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "type": "local"},
]

EVAL_MODELS = {"name": "claude-opus-4", "model_id": "claude-opus-4-20250514", "type": "api_anthropic"}

# LLM generation parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 1024))
TOP_P = float(os.getenv("TOP_P", 0.9))
TOP_K = int(os.getenv("TOP_K", 50))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.2))

# Reward function hyperparameters
# θc: complexity threshold, θa: answer correctness threshold
THETA_C = float(os.getenv("THETA_C", 0.02))  # θc: complexity alignment tolerance
THETA_A = float(os.getenv("THETA_A", 0.02))  # θa: correctness alignment tolerance
BERTSCORE_THRESHOLD = float(os.getenv("BERTSCORE_THRESHOLD", 0.92))
REWARD_WINDOW_SIZE = float(os.getenv("REWARD_WINDOW_SIZE", 2))
# Reward weights (How vs Which)
REWARD_COMPLEXITY = float(os.getenv("REWARD_COMPLEXITY", 0.36))  # weight for process complexity (How)
REWARD_CORRECTNESS = float(os.getenv("REWARD_CORRECTNESS", 0.39))  # weight for answer correctness (Which)
REWARD_COHERENCE = float(os.getenv("REWARD_COHERENCE", 0.25))   # weight for answer coherence

# The count of sentences for one item cannot be more than 3x the count for another.
WHW_RULES = {
    "min_total_sentences": int(os.getenv("WHW_RULES_MIN_TOTAL_SENTENCE", 6)),
    "max_balance_ratio": float(os.getenv("WHW_RULES_MAX_BALANCE_RATIO", 2.99))
}

# Paths & repository settings
LOG_DIR = Path("/logs")
BACKUP_DIR = Path("/backups")
DATASET_PATH = Path(os.getenv("DATASET_PATH", PROJECT_ROOT / "datasets"))
RESULTS_DIR =  Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
# Create directories if they don't exist
# os.makedirs(LOG_DIR, exist_ok=True)
# os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Settings
TOTAL_RUNS = int(os.getenv("TOTAL_RUNS", 16))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", 42))
HF_DATASET_REPO = "SellyA/reward-enginerring-CoT-pilot"
HF_TESTDATASET_REPO = "SellyA/reward-enginerring-CoT-pilot-test"

# Generate a unique filename for this experimental run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def determine_condition(trial_num: int) -> str:
    """Determines the experimental condition (A, B, C, D) based on the trial number."""
    if 1 <= trial_num <= 4:
        return 'A'
    elif 5 <= trial_num <= 8:
        return 'B'
    elif 9 <= trial_num <= 12:
        return 'C'
    elif 13 <= trial_num <= 16:
        return 'D'
    else:
        raise ValueError(f"Trial number {trial_num} is out of the valid range (1-{TOTAL_RUNS}).")