# src/turn_manager.py
# Manages the logic for a single problem-solving turn.

import re
from typing import Dict, Any, Optional, Tuple
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
    config: dict, 
    system_prompt: str, 
    user_prompt: str,
    question_data: dict,
    seed: int,
    local_models: Optional[dict] = None
) -> dict:
    """
    Executes a single turn for the solver model.
    It calls the model, parses the output, and assembles the submission data object.
    """
    
    # 1. Call the appropriate solver model
    raw_response = model_caller.dispatch_solver_call(
        config=config,
        temperature=settings.TEMPERATURE,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        local_models=local_models,
        log_context=question_data
    )

    print(f"DEBUG: Raw model response for QID {question_data.get('QID')}:")
    print("---START RAW RESPONSE---")
    print(raw_response)
    print("---END RAW RESPONSE---")
    
    # 2. Parse the model's JSON output
    # Assumes the entire model output is a parsable JSON string as per the prompts.
    submission_content = utils.parse_model_json_output(raw_response)
    
        # 3. Add specific logic to reassign pred_answer for Llama/Mistral models
    #    if it's malformed (e.g., contains 'str(i)' or previous JSON parsing error message).
    #    This check is based on model_id and the content of pred_answer.
    if config['model_id'] in ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
        pred_answer_val = submission_content.get("pred_answer")
        if isinstance(pred_answer_val, str) and ("JSON parsing failed" in pred_answer_val):
            print(f"DEBUG: Malformed pred_answer '{pred_answer_val}' detected for model {config['model_id']}. Attempting extraction from pseudocode.")
            pseudocode_val = submission_content.get("pred_pseudocode", "")
            
            # Look for all patterns like 'return "..."', 'return ["..."]', 'print("...")'
            # The regex now captures the content within quotes or brackets after return/print
            # It also handles the "pred_answer":"..." pattern from the raw error string
            answer_patterns = re.findall(
                r"return\s*\"(.*?)\"|"  # return "answer"
                r"return\s*\[\"(.*?)\"\]|" # return ["answer"]
                r"print\s*\(\"(.*?)\"\)|" # print("answer")
                r"print\s*\((\d+)\)|" # print(number) - for numerical answers like 608
                r"\"pred_answer\":\s*\"(.*?)\"", # "pred_answer":"answer" from previous error strings
                pseudocode_val, 
                re.DOTALL
            )
            
            extracted_answer = None
            # Iterate through all found patterns and get the last non-empty one
            for match_tuple in answer_patterns:
                for group_val in match_tuple:
                    if group_val is not None and group_val.strip() != "":
                        extracted_answer = group_val.strip() # Update with the latest found value

            if extracted_answer:
                print(f"DEBUG: Successfully extracted '{extracted_answer}' from pseudocode (last occurrence).")
                submission_content["pred_answer"] = extracted_answer
            else:
                # Fallback if no specific answer found in pseudocode or from malformed string
                submission_content["pred_answer"] = "Extraction failed from malformed pseudocode"

    # 3. Assemble the final data object to match the exact log format
    submission_log_entry = {
        "timestamp": utils.get_utc_timestamp(),
        "seed": seed,
        "question_info": {
            "question_num": question_data.get("QID"),
            "category": question_data.get("Category"),
            "question": question_data.get("Question"),
        },
        "submit": submission_content,
        "answer": {
            "answer": question_data.get("Answer"),
            "reasoning_steps": question_data.get("reasoning_steps"),
            "pseudocode": question_data.get("pseudocode"),
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
                                (reward_components_for_log['complexity_score'] * settings.REWARD_COMPLEXITY) + \
                                (reward_components_for_log['coherence_score'] * settings.REWARD_COHERENCE)
                                
    return final_log_entry, final_turn_reward