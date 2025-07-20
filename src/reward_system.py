# src/reward_system.py
# Call the value received from the scoring module and calculate the reward

from typing import Dict, Any

def calculate_reward(condition: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes evaluation results and returns a dictionary with reward components, matching the specified log format.
    The final weighted sum is NOT calculated here.
    """
    # --- Conditions A, C ---
    if condition in ['A', 'C']:
        score = evaluation_results.get('correctness_score', 0.0)
        return {
            "goal_alignment": None,
            "whw_description_rule": None,
            "correctness_score": score,
            "complexity_score": None
        }

    # --- Conditions B, D ---
    elif condition in ['B', 'D']:
        goal_aligned = evaluation_results.get('goal_alignment', 0.0) == 1.0
        whw_description_rule = evaluation_results.get('whw_description_rule', False)
        
        # The components are recorded regardless of the gatekeepers for analysis.
        correctness = evaluation_results.get('correctness_score', 0.0)
        complexity = evaluation_results.get('complexity_score', 0.0)
        
        return {
            "goal_alignment": goal_aligned,
            "whw_description_rule": whw_description_rule,
            "correctness_score": correctness,
            "complexity_score": complexity
        }