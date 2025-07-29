# src/reward_system.py
# Call the value received from the scoring module and calculate the reward

from typing import Dict, Any
from configs import settings

import logging
import sys

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def calculate_reward(condition: str, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes evaluation results and returns a dictionary with reward components, matching the specified log format.
    The final weighted sum is NOT calculated here.
    """
    # --- Conditions A, C ---
    if condition in ['A', 'C']:
        score = evaluation_results.get('correctness_score', 0.0)
        return {
            "total": score,
            "correctness_score": score,
            "complexity_score": None,
            "coherence_score": None,
            "goal_alignment": None,
            "whw_description_rule": None,
            "whw_moral_eval": None
        }

    # --- Conditions B, D ---
    else:
        goal_aligned = evaluation_results.get('goal_alignment', False)
        whw_description_rule = evaluation_results.get('whw_description_rule', False)
        moral_eval = evaluation_results.get('whw_moral_eval')
        whw_avg = (moral_eval.get("why",0) + moral_eval.get("how",0) + moral_eval.get("which",0))/3
        whw_bonus = 0.5 * whw_avg

 
        # The components are recorded regardless of the gatekeepers for analysis.
        correctness = evaluation_results.get('correctness_score', 0.0) * settings.REWARD_CORRECTNESS
        complexity = evaluation_results.get('complexity_score', 0.0) * settings.REWARD_COMPLEXITY
        coherence = evaluation_results.get('coherence_score', 0.0) * settings.REWARD_COHERENCE
        
        passed_gate = evaluation_results["goal_alignment"] and evaluation_results["whw_description_rule"]
        total = (correctness + complexity + coherence + whw_bonus) if passed_gate else 0.0
    
        return {
            "total": total,
            "correctness_score": correctness,
            "complexity_score": complexity,
            "coherence_score": coherence,
            "goal_alignment": goal_aligned,
            "whw_description_rule": whw_description_rule,
            "whw_bonus": whw_bonus
        }