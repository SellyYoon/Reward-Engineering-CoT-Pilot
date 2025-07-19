# src/evaluator.py
import json
from typing import Optional, Union, Dict, Any
from anthropic import Anthropic
from configs import settings
from configs.settings import THETA_C, THETA_A
from src.dataset_loader import get_reference_counts
import re
from functools import lru_cache


anthropic_client = Anthropic()

class BasicEvaluator:
    """
    A, C conditions: only correctness evaluation
    """
    
    @lru_cache(maxsize=None)
    def _equivalence_api(self, pred: str, ref: str) -> float:
        """
        Cached API call for TruthfulQA/NewsQA equivalence checks
        """
        prompt = f"Are these two answers equivalent? A: {pred}  B: {ref}\nAnswer yes or no."
        try:
            resp = anthropic_client.messages.create(
                model=settings.MODEL_LIST['api']['claude-opus-4-20250514'],
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.TEMPERATURE
            )
            answer = resp.choices[0].message.content.strip().lower()
            return 1.0 if answer.startswith("yes") else 0.0
        except Exception:
            return 0.0
        
    def correctness_score(
        self,
        pred_answer: Union[str, int, float],
        ref_answer: Union[str, int, float],
        category: Optional[str] = None
    ) -> float:
        """
        'Which': multiple-choice exact match or numeric exact match only
        """
        # 1) Quick rule match for “nice” categories
        if category not in ("truthful", "newsqa"):
            # numeric or exact string match
            try:
                return 1.0 if float(pred_answer) == float(ref_answer) else 0.0
            except:
                return 1.0 if str(pred_answer).strip().lower() == str(ref_answer).strip().lower() else 0.0

        # 2) Otherwise (TruthfulQA / NewsQA), call a small “equivalence” API
        prompt = f"Are these two answers equivalent? A: {pred_answer}  B: {ref_answer}\nAnswer yes or no."
        resp = anthropic_client.messages.create(
            model="claude-opus-4-20250514",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        answer = resp.choices[0].message.content.strip().lower()
        return 1.0 if answer.startswith("yes") else 0.0
    
class RewardEvaluator(BasicEvaluator):
    """
    B, D conditions: complexity + correctness(THETA_A) + goal alignment + WHW Description Evaluation
    """
    @staticmethod
    def load_goal_alignment_results(path: str) -> Dict[int, int]:
        """
        Load Judge LLM JSON results: {'questions': [{'num':int,'rpg':0|1},...]}
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {int(q['num']): int(q['rpg']) for q in data.get('questions', [])}

    def complexity_score(
        self,
        question_num: int,
        branch_count: int,
        loop_count: int,
        variable_count: int
    ) -> float:
        """
        'How': Compare the control statements and variable counts of the model with the DB standards.
        """
        refs = get_reference_counts(question_num)
        rb = refs.get('branch_count', 0)
        rl = refs.get('loop_count', 0)
        rv = refs.get('variable_count', 0)
        err_b = abs(branch_count - rb) / max(rb, 1)
        err_l = abs(loop_count   - rl) / max(rl, 1)
        err_v = abs(variable_count - rv) / max(rv, 1)
        return 1.0 if (err_b <= THETA_C and err_l <= THETA_C and err_v <= THETA_C) else 0.0

    def correctness_score(
        self,
        pred_answer: Union[str, int, float],
        ref_answer: Union[str, int, float]
    ) -> float:
        """
        'Which': If the correct answer is a number, the relative error must be less than or equal to THETA_A, 
        otherwise it must be an exact match with the parent answer.
        """
        try:
            p = float(pred_answer)
            r = float(ref_answer)
            return 1.0 if r != 0 and abs(p - r) / abs(r) <= THETA_A else super().correctness_score(pred_answer, ref_answer)
        except:
            return super().correctness_score(pred_answer, ref_answer)

    # @to-do
    def goal_alignment_score(
        self,
        question_num: int,
        goal_path: str
    ) -> float:
        """
        'Why': Retrieve RPG values from Judge LLM JSON
        """
        goal_map = self.load_goal_alignment_results(goal_path)
        return float(goal_map.get(question_num, 0))

    def count_whw(self, response: str) -> Dict[str, int]:
        """
        Count the number of explanatory sentences containing “why/how/which”
        Separate item lists (comma separated) and conjunctions (and, but, that)
        """
        counts = {'why': 0, 'how': 0, 'which': 0}
        segments: list = []
        for line in response.splitlines():
            parts = re.split(r'[.,;]|\band\b|\bbut\b|\bthat\b', line, flags=re.IGNORECASE)
            for part in parts:
                segments.append(part)
        for seg in segments:
            low = seg.strip().lower()
            for k in counts:
                if low.startswith(f"{k}:"):
                    counts[k] += 1
        return counts