# src/turn_manager.py
# Manages the logic for a single problem-solving turn.

from typing import Dict, Any, Tuple
from configs import settings
from src import model_caller, evaluator, reward_system, utils

# A map to instantiate the correct evaluator based on the condition.
EVALUATOR_MAP = {
    'A': evaluator.BasicEvaluator(),
    'B': evaluator.RewardEvaluator(),
    'C': evaluator.BasicEvaluator(),
    'D': evaluator.RewardEvaluator(),
}

def run_solver_turn(
    model_id: str, 
    system_prompt: str, 
    user_prompt: str,
    question_data: dict,
    seed: int
) -> dict:
    """
    Executes a single turn for the solver model.
    It calls the model, parses the output, and assembles the submission data object.
    """
    # 1. Call the appropriate solver model
    raw_response = model_caller.dispatch_solver_call(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # 2. Parse the model's JSON output
    # Assumes the entire model output is a parsable JSON string as per the prompts.
    submission_content = utils.parse_model_json_output(raw_response)

    # 3. Assemble the final data object to match the exact log format
    submission_log_entry = {
        "timestamp": utils.get_utc_timestamp(),
        "seed": seed,
        "question_info": {
            "question_num": question_data.get("QID"),
            "category": question_data.get("category"),
            "question": question_data.get("question"),
            "choices": question_data.get("Choices")
        },
        "submit": submission_content,
        "answer": {
            "answer": question_data.get("Answer"),
            "pseudocode": question_data.get("instruction_complexity", {}).get("pseudocode"),
            "loop_count": question_data.get("loop_count"),
            "branch_count": question_data.get("branch_count"),
            "variable_count": question_data.get("variable_count")
        }
    }
    
    return submission_log_entry

def evaluate_reward_turn(
    condition: str,
    submission_data: dict
) -> Tuple[dict, float]:
    """
    Evaluates a submission, calculates rewards, and assembles the final log entry
    exactly matching the specified format.
    """
    evaluator_instance = EVALUATOR_MAP[condition]
    
    # 1. Perform all necessary evaluations to get the 'eval' dictionary
    evaluation_results = evaluator_instance.evaluate(submission_data)

    # 2. Get the reward components dictionary for the 'reward' field
    reward_components_for_log = reward_system.calculate_reward(
        condition=condition,
        evaluation_results=evaluation_results
    )

    # 3. Assemble the final, complete log entry by adding 'eval' and 'reward' fields
    final_log_entry = submission_data.copy() # Start with the submission log
    final_log_entry['eval'] = evaluation_results
    final_log_entry['reward'] = reward_components_for_log
    
    # 4. Calculate the single, final reward score for real-time feedback
    final_turn_reward = 0.0
    if condition in ['A', 'C']:
        final_turn_reward = evaluation_results.get('correctness_score', 0.0)
    elif condition in ['B', 'D']:
        if reward_components_for_log.get('goal_alignment') and reward_components_for_log.get('whw_description_rule'):
            final_turn_reward = (reward_components_for_log['correctness_score'] * settings.REWARD_CORRECTNESS) + \
                                (reward_components_for_log['complexity_score'] * settings.REWARD_COMPLEXITY)

    return final_log_entry, final_turn_reward