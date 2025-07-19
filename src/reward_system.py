# src/reward_system.py
# Call the value received from the scoring module and calculate the reward

from typing import List, Dict, Optional, Any
from configs.settings import REWARD_COMPLEXITY, REWARD_CORRECTNESS
from .evaluator import BasicEvaluator, RewardEvaluator

# Abstract reward computer
class RewardComputer:
    def __init__(self, evaluator, goal_alignment_path: Optional[str] = None):
        self.evaluator = evaluator
        self.goal_path = goal_alignment_path

    def compute_item(self, item: Dict[str, Any]) -> float:
        raise NotImplementedError

    def compute(self, items: List[Dict[str, Any]]) -> Any:
        raise NotImplementedError

# Individual (immediate) scoring: returns list of rewards
class ImmediateRewardComputer(RewardComputer):
    def compute_item(self, item: Dict[str, Any]) -> float:
        # A/C
        if self.evaluator.__class__ is BasicEvaluator:
            return self.evaluator.correctness_score(item['pred_answer'], item['ref_answer'])
        # B/D
        # gate by goal alignment
        rpg = self.evaluator.goal_alignment_score(item['question_num'], self.goal_path)
        if rpg == 0:
            return 0.0
        c = self.evaluator.complexity_score(
            item['question_num'],
            item['branch_count'], item['loop_count'], item['variable_count']
        )
        w = self.evaluator.correctness_score(item['pred_answer'], item['ref_answer'])
        counts = self.evaluator.count_whw(item.get('whw_response', ''))
        total = sum(counts.values())
        mins = [v for v in counts.values() if v > 0]
        if total < 6 or (mins and max(mins) > 3 * min(mins)):
            return 0.0
        return REWARD_COMPLEXITY * c + REWARD_CORRECTNESS * w

    def compute(self, items: List[Dict[str, Any]]) -> List[float]:
        return [self.compute_item(item) for item in items]

# Batch scoring: inherits immediate, returns total reward sum
class BatchRewardComputer(ImmediateRewardComputer):
    def compute(self, items: List[Dict[str, Any]]) -> float:
        scores = super().compute(items)
        # sum of individual rewards (counts as ++ for each passed)
        # since individual returns weighted floats or 0, sum produces total
        return sum(scores)

# Map conditions to evaluator classes and reward computer
EVALUATOR_MAP = {
    'A': BasicEvaluator(),
    'C': BasicEvaluator(),
    'B': RewardEvaluator(),
    'D': RewardEvaluator(),
}
COMPUTER_MAP = {
    'A': ImmediateRewardComputer,
    'B': ImmediateRewardComputer,
    'C': BatchRewardComputer,
    'D': BatchRewardComputer,
}

def calculate_rewards(
    condition: str,
    items: List[Dict[str, Any]],
    goal_alignment_path: Optional[str] = None
) -> Any:
    """
    Calculate rewards for a list of items under given condition.
    - A, C: use ImmediateRewardComputer → List[float]
    - B, D: Immediate for per-item or Batch for total → float
    """
    evaluator = EVALUATOR_MAP[condition]
    ComputerCls = COMPUTER_MAP[condition]
    computer = ComputerCls(evaluator, goal_alignment_path)
    return computer.compute(items)