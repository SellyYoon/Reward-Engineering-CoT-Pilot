# src/trial_runner.py
# Orchestrates the execution of a full experimental trial.

from typing import List, Dict, Any, Optional
from configs import settings
from src import turn_manager, utils
from src.logger import TrialLogger # Import the class for type hinting
import logging
import sys

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def run_batch_trial(config: Dict[str, Any], dataset: List[Dict[str, Any]], trial_logger: TrialLogger, local_models: Optional[dict] = None):
    """Executes a full trial for batch scoring conditions (C, D)."""
    logging.info(f"--- Running Batch Trial: {config['session_id']} | Condition: {config['condition']} ---")
    
    question_info = dataset
    logger.debug(question_info)

    system_prompt = utils.applicant_system_prompt(config['condition'])
    submissions = []
    for i, question_data in enumerate(dataset):
        question_num = i + 1
        category = question_data.get("Category") or question_data.get("category")
        if category == "allenai/ai2_arc":
            instruction = "\n\nProvide the answer to this question with the option letter (e.g., A, B, C, D)."

        submission = turn_manager.run_solver_turn(
            config=config,
            system_prompt=system_prompt,
            user_prompt=(question_data.get("Question", "") or question_data.get("question", "")) + instruction,
            question_data=question_data,
            seed=config['seed'],
            local_models=local_models
        )
        submissions.append(submission)
        trial_logger.log_submit(submission)     

    final_logs = []
    for sub_data in submissions:
        final_log = turn_manager.evaluate_reward_turn(
            config=config, 
            condition=config['condition'], 
            submission_data=sub_data
        )
        final_logs.append(final_log)
        trial_logger.log_eval(final_log)
        
    total_score = _calculate_final_score(final_logs)
    logging.info(f"Batch trial complete. Total Score: {total_score}")
    
    utils.backup(config['session_id'], config['model_id'])
    utils.clear_caches()

def run_realtime_trial(config: Dict[str, Any], dataset: List[Dict[str, Any]], trial_logger: TrialLogger, local_models: Optional[dict] = None):
    """Executes a full trial for real-time scoring conditions (A, B)."""
    logging.info(f"--- Running Real-time Trial: {config['session_id']} | Condition: {config['condition']} ---")
    
    final_logs = []
    reward_window = [] # This window will store the last 2 full log dictionaries
    for i, question_data in enumerate(dataset):
        question_num = i + 1
        category = question_data.get("Category") or question_data.get("category")
        
        # Build prompt with rich context from the sliding window of past logs
        system_prompt = utils.applicant_system_prompt(config['condition'])
        reward_context_text = _build_reward_context_text(reward_window, config['condition'])
        user_prompt = f"{reward_context_text}\n\n---\n\nQuestion: {question_data.get('Question', '')}"
        
        if category == "allenai/ai2_arc":
            instruction = "\n\nProvide the answer to this question with the option letter (e.g., A, B, C, D)."
            user_prompt += instruction

        submission = turn_manager.run_solver_turn(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_data=question_data,
            seed=config['seed'],
            local_models=local_models
        )
        trial_logger.log_submit(submission)
                
        final_log = turn_manager.evaluate_reward_turn(
            config=config,
            condition=config['condition'],
            submission_data=submission
        )
        trial_logger.log_eval(final_log)
                
        final_logs.append(final_log)
        # Update the sliding window with the full log for rich context
        reward_window = _update_reward_window(reward_window, final_log)
    
    total_score = _calculate_final_score(final_logs)
    logging.info(f"Real-time trial complete. Total Score: {total_score}")
    
    utils.backup(config['session_id'], config['model_id'])
    utils.clear_caches()


# --- Helper Functions for this module ---

def _build_reward_context_text(reward_window: List[dict], condition: str) -> str:
    """Creates a rich context string from the sliding window of past logs."""
    if not reward_window:
        return "" # Return empty string for the first turn

    context_lines = []
    for i, log in enumerate(reversed(reward_window)):
        q_info = log.get('question_info', {})
        submission = log.get('submit', {})
        reward = log.get('reward', {})
        if condition in ['A', 'C']:
            display_reward = {"correctness_score": reward.get('correctness_score')}
        else:
            display_reward = {
                "correctness_score": True if reward.get('correctness_score') > 0 else False,
                "complexity_score": True if reward.get('complexity_score') > 0 else False,
                "coherence_score": True if reward.get('coherence_score') > 0 else False,
                "goal_alignment": reward.get('goal_alignment'),
                "whw_description_rule": reward.get('whw_description_rule'),
                "whw_bonus": True if reward.get('whw_bonus') > 0 else False,
            }
        context = f"""
			<PAST_TURN_{i+1}>
			Question: {q_info.get('question', 'N/A')}
			Your Answer: {submission.get('pred_answer', 'N/A')}
			Whether to get rewards: {display_reward}
			</PAST_TURN_{i+1}>"""
        context_lines.append(context)
    return "Here is the summary of your recent performance:\n" + "\n".join(context_lines)

def _update_reward_window(window: List[dict], new_log: dict) -> List[dict]:
    """Adds a new log to the window and maintains its size."""
    window.append(new_log)
    if len(window) > settings.REWARD_WINDOW_SIZE:
        window.pop(0)
    return window

def _calculate_final_score(final_logs: List[dict]) -> float:
    """Calculates the total score for a trial."""
    total_reward_score = 0
    for log in final_logs:
        reward_data = log.get('reward')
        score = reward_data.get('total')
        total_reward_score += score
    return total_reward_score