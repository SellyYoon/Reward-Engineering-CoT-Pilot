# main.py


from pathlib import Path
from dataset.dataset import build_master_set, standardize_and_flatten, save_and_push
from src.dataset_loader import load_master_dataset
from src.session_manager import init_session, determine_condition
from src.llm_utils import call_judge_llm_batch, solve_and_parse
from src.reward_system import calculate_rewards
from src.logger import log_results
from configs import settings

# Configuration
HF_DATASET_REPO = settings.HF_DATASET_REPO
LOCAL_DIR = Path(__file__).parent.parent / "datasets" / "final_problem_set"
DB_PATH = Path(__file__).parent.parent / "master.db"


def run_experiment():
    # 1. Initialize session
    session_id, trial = init_session()

    # 2. Determine condition A/B/C/D
    condition = determine_condition(trial)

    # 3. Build or load master dataset
    # One-time build & upload
    if trial == 1:
        master = build_master_set()
        final_ds = standardize_and_flatten(master)
        save_and_push(final_ds, LOCAL_DIR, HF_DATASET_REPO)

    # Always load dataset for problem set
    raw_examples = load_master_dataset()

    # 4. If B/D, call Judge LLM once for all items
    goal_path = None
    if condition in ("B", "D"):
        goal_path = call_judge_llm_batch(raw_examples)

    # 5. Solve each problem and parse outputs
    items = solve_and_parse(raw_examples, condition)

    # 6. Compute rewards (per-item or batch sum)
    rewards = calculate_rewards(condition, items, goal_path)

    # 7. Log results
    log_results(session_id, trial, condition, items, rewards, DB_PATH)

    print(f"Session {session_id} trial {trial} ({condition}) complete. Rewards: {rewards}")


if __name__ == "__main__":
    run_experiment()
