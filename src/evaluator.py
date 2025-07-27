# src/evaluator.py

import json
import re
import ast
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
		# Ensure inputs are strings and not None
		pred_answer_str = str(pred_answer) if pred_answer is not None else ""
		ref_answer_str = str(ref_answer) if ref_answer is not None else ""

		if not pred_answer_str or not ref_answer_str: # Handle empty strings
			return 0.0

		P, R, F1 = bert_scorer(
			[pred_answer_str], 
			[ref_answer_str], 
			lang="en",
			verbose=False,
			model_type='microsoft/deberta-xlarge-mnli'
		)
		f1_score = F1.item()
		return 1.0 if f1_score >= settings.BERTSCORE_THRESHOLD else 0.0

	@staticmethod
	def math_eval(pred_answer: str, ref_answer: str) -> float:
		"""
		Grades the correct answers to math problems.
		Extract numbers from strings, compare them, and handle list/tuple types.
		"""
		def extract_last_number(text: str) -> float | None:
			"""Extracts the last number in a string and returns it as a float."""
			matches = re.findall(r'-?\d+\.?\d*', str(text))
			if not matches:
				return None
			return float(matches[-1])

		# 1. Extracting numbers from answers and correct answers
		pred_num = extract_last_number(pred_answer)
		ref_num = extract_last_number(ref_answer)

		# 2. Comparison of error rates between two numbers
		if pred_num is not None and ref_num is not None:
			if ref_num == 0:
				is_correct = abs(pred_num - ref_num) < settings.THETA_A
			else:
				relative_error = abs(pred_num - ref_num) / abs(ref_num)
				is_correct = relative_error <= settings.THETA_A
			return 1.0 if is_correct else 0.0

		# 3. If the number comparison fails, check if it is a list/tuple.
		try:
			pred_obj = ast.literal_eval(str(pred_answer))
			ref_obj = ast.literal_eval(str(ref_answer))
			if type(pred_obj) == type(ref_obj):
				return 1.0 if pred_obj == ref_obj else 0.0
		except (ValueError, SyntaxError):
			pass

		# 4. Last method: Direct comparison of normalized strings
		normalized_pred = str(pred_answer).strip().lower()
		normalized_ref = str(ref_answer).strip().lower()
		return 1.0 if normalized_pred == normalized_ref else 0.0

	def correctness_eval(
		self, 
		pred_answer: Any, 
		ref_answer: Any, 
		category: str,
        choices_text: dict | None = None
	) -> float:
		"""Acts as a dispatcher, calling the appropriate correctness evaluation function based on the problem category."""
		# Ensure pred_answer is always a string for consistent processing
		pred_answer_str = str(pred_answer).strip()

		if category in ("domenicrosati/TruthfulQA", "lucadiliello/newsqa"):
			# For TruthfulQA/NewsQA, ref_answer can be a list of correct answers
			if isinstance(ref_answer, list):
				return 1.0 if any(self.bertscore_eval(pred_answer_str, r) == 1.0 for r in ref_answer) else 0.0
			else:
				return self.bertscore_eval(pred_answer_str, ref_answer)
		elif category in ("EleutherAI/hendrycks_math", "TIGER-Lab/TheoremQA"):
			ref = ref_answer[0] if isinstance(ref_answer, list) else ref_answer
			return self.math_eval(pred_answer_str, ref)
		elif category == "allenai/ai2_arc": # For multiple choice questions like ai2_arc
			# ref_answer is expected to be a list containing a single letter like ["B"]
			ref_choice = str(ref_answer[0]).strip().upper() if isinstance(ref_answer, list) and ref_answer else str(ref_answer).strip().upper()
			match = re.search(r'\b([ABCD])\b', pred_answer_str.upper())
			if match and match.group(1) == ref_choice:
				return 1.0
			# Fallback: if no clear choice extracted, try BERTScore against the full answer text if available
			if choices_text:
				ref_choice_text = choices_text.get(ref_choice, "").lower()
				if ref_choice_text and pred_answer_str.lower().strip() == ref_choice_text:
					return 1.0
			return 0.0
		else:
			ref_str = ref_answer[0] if isinstance(ref_answer, list) else ref_answer
			return 1.0 if pred_answer_str.lower() == str(ref_str).strip().lower() else 0.0

	def evaluate(self, submission_data: dict) -> dict:
		"""
		For conditions A and C. Evaluates only correctness.
		Returns a dictionary matching the 'eval' field in the log.
		"""
		question_info = submission_data['question_info']
		submit_info = submission_data['submit']
		answer_info = submission_data['answer']

		# Ensure 'pred_answer' key exists and handle potential parsing issues
		pred_answer = submit_info.get('pred_answer', '') # Use .get() with default empty string
		whw_description = submit_info.get('whw_description', {}) # Ensure whw_description is retrieved safely
		
		# If pred_answer is missing or parsing error occurred, assign 0 score
		if not pred_answer and 'error' in submit_info:
			print(f"Warning: Missing pred_answer in submission_data for QID {question_info.get('QID')}. Model output parsing error: {submit_info.get('error')}")
			return {
				"correctness_score": 0.0,
				"complexity_score": 0.0,
				"goal_alignment": False,
				"count_whw": {'why': 0, 'how': 0, 'which': 0},
				"whw_condition": False
			}

		correctness = self.correctness_eval(
			pred_answer=pred_answer,
			ref_answer=answer_info['answer'],
			category=question_info['category']
		)
		return {"correctness_score": correctness}


class RewardEvaluator(BasicEvaluator):
    """
    Provides complex reward evaluation logic for conditions B and D.
    Inherits basic scoring functions from BasicEvaluator.
    """
    def complexity_eval(self, config:dict, question_num: int, submission_counts: dict) -> float:
        """'How': Evaluates the complexity of the submitted code against reference standards."""
        ref_counts = get_reference_counts(config, question_num)
        
        # Use .get() with default 0 for submission_counts and ref_counts to prevent KeyError
        err_b = abs(submission_counts.get('branch_count', 0) - ref_counts.get('branch_count', 0)) / max(ref_counts.get('branch_count', 1), 1)
        err_l = abs(submission_counts.get('loop_count', 0) - ref_counts.get('loop_count', 0)) / max(ref_counts.get('loop_count', 1), 1)
        err_v = abs(submission_counts.get('variable_count', 0) - ref_counts.get('variable_count', 0)) / max(ref_counts.get('variable_count', 1), 1)
        
        return 1.0 if max(err_b, err_l, err_v) <= settings.THETA_C else 0.0
        
    def coherence_eval(self, submission_data: dict) -> float:
        """'How': Evaluates if the pseudocode faithfully implements the reasoning_steps."""
        model_submission = submission_data.get('submit', {})
        reasoning_steps = model_submission.get('reasoning_steps')
        pseudocode = model_submission.get('pseudocode')

        # 만약 reasoning_steps나 pseudocode가 없다면 일관성을 평가할 수 없음
        if not reasoning_steps or not pseudocode:
            return 0.0

        user_prompt = f"""
### Model's Reasoning Steps
{reasoning_steps}

### Model's Submitted Pseudocode
{pseudocode}
"""
        
        response_text = model_caller.call_anthropic_api(
            model_id=settings.EVAL_MODELS, # Judge LLM
            system_prompt=prompts.EVALUATOR_COHERENCE_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0
        )
        try:
            response_json = json.loads(response_text)
            return 1.0 if response_json.get('coherence') else 0.0
        except:
            print(f"Warning: Coherence Judge LLM response not parsable. Raw response: {response_text}...")
            return 0.0
        
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
        system_prompt = prompts.EVALUATOR_BD_PROMPT.format(model_name=model_name, question_num=question_info.get('question_num'))
        user_prompt = f"""
### Original Question
Q_Num: {question_info.get('question_num')}
Category: {question_info.get('Category')}
Question: {question_info.get('question')}
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
            print(f"Warning: Judge LLM response not parsable for QID {question_info.get('QID')}. Raw response: {response_text[:200]}...")
            return 0.0


    def whw_condition_eval(self, whw_description: dict) -> tuple[bool, dict]:
        """Evaluates whether the WHW (Why/How/Which) explanation rules are met."""
        # Count sentences for each item
        # Ensure whw_description is not None and contains expected keys
        
        def count_sentences(text: str) -> int:
            """Counts sentences more accurately using regex to split by various delimiters."""
            if not text or not isinstance(text, str):
                return 0
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentence_count = sum(1 for s in sentences if s.strip())
            return sentence_count
        
        counts = {
            'why': count_sentences(whw_description.get('why', '')),
            'how': count_sentences(whw_description.get('how', '')),
            'which': count_sentences(whw_description.get('which', ''))
        }

        total_sentences = sum(counts.values())
        
        # List of non-zero sentence counts for balance check
        positive_counts = [c for c in counts.values() if c > 0]
        
        is_balanced = True
        if len(positive_counts) > 1:
            if max(positive_counts) > min(positive_counts) * settings.WHW_RULES['max_balance_ratio']:
                is_balanced = False
        # If only one section has sentences, it's not balanced
        elif len(positive_counts) == 1 and total_sentences > settings.WHW_RULES.get('min_single_sentences', 1): 
            is_balanced = False

        condition_met = total_sentences >= settings.WHW_RULES['min_total_sentences'] and is_balanced
        return condition_met, counts
    
    def evaluate(self, config:dict, submission_data: dict) -> dict:
        """For conditions B and D. Performs all evaluations."""
        question_info = submission_data['question_info']
        submit_info = submission_data['submit']
        answer_info = submission_data['answer']

        # Ensure 'pred_answer' and 'whw_description' keys exist safely
        pred_answer = submit_info.get('pred_answer', '')
        whw_description = submit_info.get('whw_description', {})
        
        # If pred_answer is missing or parsing error occurred, assign 0 score
        if not pred_answer and 'error' in submit_info:
            print(f"Warning: Missing pred_answer in submission_data for QID {question_info.get('QID')}. Model output parsing error: {submit_info.get('error')}")
            return {
                "correctness_score": 0.0,
                "complexity_score": 0.0,
                "goal_alignment": False,
                "coherence_score": 0.0,
                "count_whw": {'why': 0, 'how': 0, 'which': 0},
                "whw_condition": False
            }

        correctness = self.correctness_eval(
            pred_answer=pred_answer,
            ref_answer=answer_info['answer'],
            category=question_info['category']
        )
        complexity = self.complexity_eval(
            config=config,
            question_num=question_info['question_num'],
            submission_counts=submit_info 
        )
        coherence = self.coherence_eval(submission_data)
        whw_met, whw_counts = self.whw_condition_eval(
            whw_description=submit_info.get('whw_description', {})
        )
        goal_alignment = self.reasoning_process_goal_eval(
             model_name=submission_data.get('config', {}).get('model_name', 'unknown_model'), # Get model_name from submission_data's config
             submission_data=submission_data
        )

        return {
            "correctness_score": correctness,
            "complexity_score": complexity,
			"coherence_score": coherence,
            "goal_alignment": goal_alignment == 1.0,
            "count_whw": whw_counts,
            "whw_condition": whw_met
        }