# src/trial_runner.py
# Orchestrates the execution of a full experimental trial.

from typing import List, Dict, Any
from configs import settings
from src import turn_manager, utils
from src.logger import TrialLogger # Import the class for type hinting

def run_batch_trial(config: Dict[str, Any], dataset: List[Dict[str, Any]], trial_logger: TrialLogger):
    """Executes a full trial for batch scoring conditions (C, D)."""
    print(f"--- Running Batch Trial: {config['session_id']} | Condition: {config['condition']} ---")
    
    system_prompt = utils.applicant_system_prompt(config['condition'])
    submissions = []
    for question_data in dataset:
        submission = turn_manager.run_solver_turn(
            model_id=config['model_id'],
            system_prompt=system_prompt,
            user_prompt=question_data.get("Question", ""),
            question_data=question_data,
            seed=config['seed']
        )
        submissions.append(submission)
        trial_logger.log_submit(submission)

    final_logs = []
    for sub_data in submissions:
        final_log, _ = turn_manager.evaluate_reward_turn(config['condition'], sub_data)
        final_logs.append(final_log)
        trial_logger.log_eval(final_log)
        
    total_score = _calculate_final_score(final_logs, config['condition'])
    print(f"Batch trial complete. Total Score: {total_score}")
    
    utils.backup(config)
    utils.clear_caches()

def run_realtime_trial(config: Dict[str, Any], dataset: List[Dict[str, Any]], trial_logger: TrialLogger):
    """Executes a full trial for real-time scoring conditions (A, B)."""
    print(f"--- Running Real-time Trial: {config['session_id']} | Condition: {config['condition']} ---")
    
    final_logs = []
    reward_window = [] # This window will store the last 2 full log dictionaries
    
    for question_data in dataset:
        system_prompt = utils.applicant_system_prompt(config['condition'])

        # Build prompt with rich context from the sliding window of past logs
        reward_context_text = _build_reward_context_text(reward_window)
        user_prompt = f"{reward_context_text}\n\n---\n\nQuestion: {question_data.get('Question', '')}"

        submission = turn_manager.run_solver_turn(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_data=question_data,
            seed=config['seed']
        )
        trial_logger.log_submit(submission)
                
        final_log, _ = turn_manager.evaluate_reward_turn(
            condition=config['condition'],
            submission_data=submission
        )
        trial_logger.log_eval(final_log)
                
        final_logs.append(final_log)
        # Update the sliding window with the full log for rich context
        reward_window = _update_reward_window(reward_window, final_log)
    
    utils.backup(config)
    utils.clear_caches()

# --- Helper Functions for this module ---

def _build_reward_context_text(reward_window: List[dict]) -> str:
    """Creates a rich context string from the sliding window of past logs."""
    if not reward_window:
        return "" # Return empty string for the first turn

    context_lines = []
    for i, log in enumerate(reversed(reward_window)):
        q_info = log.get('question_info', {})
        submission = log.get('submit', {})
        reward = log.get('reward', {})
        context = f"""
			<PAST_TURN_{i+1}>
			Question: {q_info.get('question', 'N/A')}
			Your Answer: {submission.get('pred_answer', 'N/A')}
			Reward Components: {reward}
			</PAST_TURN_{i+1}>"""
        context_lines.append(context)
    return "Here is the summary of your recent performance:\n" + "\n".join(context_lines)

def _update_reward_window(window: List[dict], new_log: dict) -> List[dict]:
    """Adds a new log to the window and maintains its size."""
    window.append(new_log)
    if len(window) > settings.REWARD_WINDOW_SIZE:
        window.pop(0)
    return window

def _calculate_final_score(final_logs: List[dict], condition: str) -> float:
    """Calculates the total score for a trial."""
    total_reward_score = 0
    for log in final_logs:
        components = log['reward']
        if condition in ['B', 'D']:
            if components.get('goal_alignment') and components.get('whw_description_rule'):
                score = (components.get('correctness_score', 0.0) * settings.REWARD_CORRECTNESS) + \
                        (components.get('complexity_score', 0.0) * settings.REWARD_COMPLEXITY)
                total_reward_score += score
        else: # Condition A or C
            total_reward_score += components.get('correctness_score', 0.0)
    return total_reward_score