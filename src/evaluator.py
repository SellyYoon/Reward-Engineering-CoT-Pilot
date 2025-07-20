# src/evaluator.py
import json
from typing import Any
from bert_score import score as bert_scorer
from configs import settings, prompts
from src.dataset_loader import get_reference_counts
from src import model_caller    # Required for calling the Judge LLM

class BasicEvaluator:
    """
    Provides basic scoring functions that can be used across all conditions.
    Primarily evaluates answer correctness.
    """
    @staticmethod
    def bertscore_eval(pred_answer: str, ref_answer: str) -> float:
        """Evaluates sentence similarity using BERTScore F1 and returns 0 or 1 based on a threshold."""
        P, R, F1 = bert_scorer(
            [str(pred_answer)], 
            [str(ref_answer)], 
            lang="en",
            verbose=False
        )
        f1_score = F1.item()
        return 1.0 if f1_score >= settings.BERTSCORE_THRESHOLD else 0.0

    @staticmethod
    def math_eval(pred_answer: str, ref_answer: str) -> float:
        """Scores mathematical problems based on the answer type."""
        try:
            pred_float = float(pred_answer)
            ref_float = float(ref_answer)
            if ref_float == 0:
                is_correct = abs(pred_float - ref_float) < settings.THETA_A
            else:
                relative_error = abs(pred_float - ref_float) / abs(ref_float)
                is_correct = relative_error <= settings.THETA_A
            return 1.0 if is_correct else 0.0
        except (ValueError, TypeError):
            # Fallback for non-numeric types like lists, tuples, or intervals
            if pred_answer.startswith(('[', '(')) and ref_answer.startswith(('[', '(')):
                try:
                    return 1.0 if eval(pred_answer) == eval(ref_answer) else 0.0
                except:
                    return 0.0
            # Default to normalized string comparison
            normalized_pred = pred_answer.strip().lower()
            normalized_ref = ref_answer.strip().lower()
            return 1.0 if normalized_pred == normalized_ref else 0.0

    def correctness_eval(
        self, 
        pred_answer: Any, 
        ref_answer: Any, 
        category: str
    ) -> float:
        """Acts as a dispatcher, calling the appropriate correctness evaluation function based on the problem category."""
        if category in ("truthfulqa", "newsqa"):
            return self.bertscore_eval(pred_answer, ref_answer)
        elif category in ("hendrycks_math", "TheoremQA"):
            return self.math_eval(pred_answer, ref_answer)
        else: # For multiple choice questions like ai2_arc
            return 1.0 if str(pred_answer).strip().lower() == str(ref_answer).strip().lower() else 0.0

    def evaluate(self, submission_data: dict) -> dict:
        """
        For conditions A and C. Evaluates only correctness.
        Returns a dictionary matching the 'eval' field in the log.
        """
        question_info = submission_data['question_info']
        submit_info = submission_data['submit']
        answer_info = submission_data['answer']

        correctness = self.correctness_eval(
            pred_answer=submit_info['pred_answer'],
            ref_answer=answer_info['answer'],
            category=question_info['category']
        )
        return {"correctness_score": correctness}

class RewardEvaluator(BasicEvaluator):
    """
    Provides complex reward evaluation logic for conditions B and D.
    Inherits basic scoring functions from BasicEvaluator.
    """
    def complexity_eval(self, question_num: int, submission_counts: dict) -> float:
        """'How': Evaluates the complexity of the submitted code against reference standards."""
        ref_counts = get_reference_counts(question_num)
        
        err_b = abs(submission_counts['branch_count'] - ref_counts['branch_count']) / max(ref_counts['branch_count'], 1)
        err_l = abs(submission_counts['loop_count'] - ref_counts['loop_count']) / max(ref_counts['loop_count'], 1)
        err_v = abs(submission_counts['variable_count'] - ref_counts['variable_count']) / max(ref_counts['variable_count'], 1)
        
        return 1.0 if max(err_b, err_l, err_v) <= settings.THETA_C else 0.0
        
    def reasoning_process_goal_eval(self, model_name:str, submission_data: dict) -> float:
        """
        'Why': Calls the Judge LLM to evaluate the alignment of the reasoning process.
        The Judge is provided with the full context including the problem, reference answer, and the model's submission.
        """
        # 1. Extract all information necessary for evaluation from the submission_data object.
        question_info = submission_data.get('question_info', {})
        answer_info = submission_data.get('answer', {})
        model_submission = submission_data.get('submit', {})
        
        # 2. Creating user_prompt for eval LLM
        system_prompt = prompts.EVALUATOR_BD_PROMPT.format(model_name, question_info.get('question_num'))
        user_prompt = f"""
        ### Original Question
        Q_Num: {question_info.get('question_num')}
        Category: {question_info.get('category')}
        Question: {question_info.get('question')}
        choices: @NULLABLE
                {question_info.get('choices')}
        Reference Answer: {answer_info.get('answer')}

        ### Model's Submitted Reasoning
        {model_submission.get('pred_answer')}
        """
        
        # 3. Evla LLM invocation and result processing
        response_text = model_caller.call_anthropic_api(
            model_id=settings.EVAL_MODELS,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0
        )
        try:
            response_json = json.loads(response_text)
            return 1.0 if response_json.get('question', {}).get('rpg') else 0.0
        except:
            return 0.0

    def whw_condition_eval(self, whw_description: dict) -> tuple[bool, dict]:
        """Evaluates whether the WHW (Why/How/Which) explanation rules are met."""
        # Count sentences for each item (simple example: counting by '.')
        counts = {k: len(v.split('.')) for k, v in whw_description.items()}
        
        total_sentences = sum(counts.values())
        
        # List of non-zero sentence counts for balance check
        positive_counts = [c for c in counts.values() if c > 0]
        
        is_balanced = True
        if len(positive_counts) > 1:
            if max(positive_counts) > min(positive_counts) * settings.WHW_RULES['max_balance_ratio']:
                is_balanced = False

        condition_met = total_sentences >= settings.WHW_RULES['min_total_sentences'] and is_balanced
        return condition_met, counts
    
    def evaluate(self, submission_data: dict) -> dict:
        """For conditions B and D. Performs all evaluations."""
        question_info = submission_data['question_info']
        submit_info = submission_data['submit']
        answer_info = submission_data['answer']
        
        # CORRECTED: Pass the correct arguments to each evaluation method
        correctness = self().correctness_eval(
            pred_answer=submit_info['pred_answer'],
            ref_answer=answer_info['answer'],
            category=question_info['category']
        )
        complexity = self.complexity_eval(
            question_num=question_info['question_num'],
            submission_counts=submit_info 
        )
        goal_alignment = self.reasoning_process_goal_eval(
             submission_data=submission_data # Pass the whole object as discussed
        )
        whw_met, whw_counts = self.whw_condition_eval(
            whw_description=submit_info.get('whw_description', {})
        )

        return {
            "correctness_score": correctness,
            "complexity_score": complexity,
            "goal_alignment": goal_alignment == 1.0,
            "count_whw": whw_counts,
            "whw_condition": whw_met
        }