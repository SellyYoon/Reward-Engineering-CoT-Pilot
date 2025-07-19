# configs/settings.py
# All hyperparameters and configurations for the Reward Engineering Pilot experiment.
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Call API key via env
env_path = Path(__file__).parents[1] / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model configuration
MODEL_LIST = {
    "api": [
        "o4-mini-2025-04-16",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514"
    ],
    "local": [
        "Llama3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3"
    ]
}

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
# Reward weights (How vs Which)
REWARD_COMPLEXITY = float(os.getenv("REWARD_COMPLEXITY", 0.65))  # weight for process complexity (How)
REWARD_CORRECTNESS = 1 - REWARD_COMPLEXITY  # weight for answer correctness (Which)

# The count of sentences for one item cannot be more than 3x the count for another.
WHW_RULES = {
    "min_total_sentences": int(os.getenv("WHW_RULES_MIN_TOTAL_SENTENCE", 6)),
    "max_balance_ratio": float(os.getenv("WHW_RULES_MAX_BALANCE_RATIO", 2.99))
}

# Paths & repository settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
BACKUP_DIR = Path(os.getenv("BACKUP_DIR", BASE_DIR / "backups"))
DATASET_PATH = Path(os.getenv("DATASET_PATH", BASE_DIR / "datasets"))
RESULTS_DIR =  Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Settings
TOTAL_RUNS = int(os.getenv("TOTAL_RUNS", 16))
DEFAULT_SEED = int(os.getenv("DEFAULT_SEED", 42))
HF_DATASET_REPO = "SellyA/reward-pilot-dataset"

# Generate a unique filename for this experimental run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")