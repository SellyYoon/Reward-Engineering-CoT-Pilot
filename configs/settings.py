# configs/settings.py
# All hyperparameters and configurations for the Reward Engineering Pilot experiment.
import os
from datetime import datetime

# API keys (set via environment variables for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model configuration
MODEL_LIST = {
    "api": [
        "o4-mini-2025-04-16",
        "claude-sonnet-4"
    ],
    "local": [
        "llama-3-8b",
        "mistral-7b"
    ]
}

# LLM generation parameters
TEMPERATURE = 0.3
MAX_NEW_TOKENS = 550
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.2

# Reward function hyperparameters
# θc: complexity threshold, θa: answer correctness threshold
THETA_C = 0.02  # θc: complexity alignment tolerance
THETA_A = 0.02  # θa: correctness alignment tolerance
# Reward weights (How vs Which)
REWARD_ALPHA = 0.65  # weight for process complexity (How)
REWARD_BETA = 1 - REWARD_ALPHA  # weight for answer correctness (Which)

# The count of sentences for one item cannot be more than 3x the count for another.
WHW_RULES = {
    "min_total_sentences": 6,
    "max_balance_ratio": 2.99
}

# Paths & repository settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
LOG_DIR = os.path.join(BASE_DIR, "logs")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
DATASET_PATH = os.path.join(BASE_DIR, "datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiment Settings
TOTAL_RUNS = 16
DEFAULT_SEED = 42
HF_DATASET_REPO = "SellyA/reward-pilot-dataset"

# Generate a unique filename for this experimental run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")