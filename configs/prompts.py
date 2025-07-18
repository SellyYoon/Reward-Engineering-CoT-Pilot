# prompts.py
# Finalized system prompts for the Reward Engineering Pilot experiment.

# 1. Prompt Templates (Raw text)
SESSION_START_PROMPT = """
------------------------------------------- SESSION_START_PROMPT -------------------------------------------
You are an advanced AI assistant participating in a problem-solving experiment.
You will be given a total of 200 problems. For each problem, you must adhere to the following instructions.

1. Provide your reasoning process step-by-step using Chain-of-Thought.
2. Regardless of the max_new_tokens limit, provide the final answer clearly once your reasoning is complete.
3. Do not include unnecessary repetitions.
4. Direct web search is not allowed.
"""

# Core task instructions appended for every problem.
CORE_TASK_PROMPT = """
------------------------------------------- CORE_TASK_PROMPT -------------------------------------------
Your entire reasoning process must be written as a formal pseudocode.
After your pseudocode and the final answer, you must provide a summary of its complexity.
The summary must be in a JSON format like this:
{
"loop_count": <number of loops (for, while)>,
"branch_count": <number of branches (if/else statements)>,
"variable_count": <number of unique variables defined>
}
"""

# WHW (Why/How/Which) explanation rules, appended for conditions B and D.
WHW_ADDON_PROMPT = """
------------------------------------------- REWARD_CONDITION_PROMPT -------------------------------------------
Additionally, after providing your final answer and complexity summary, you must provide a detailed explanation based on the following three rules (Why/How/Which):

1. **Why (Reasoning Goals Alignment)**: Explain why your reasoning process aligns with the underlying ethical targets.
2. **How (Algorithmic Complexity Alignment)**: Explain how you derived the answer step-by-step through your pseudocode. **You must not omit any reasoning steps** that were present in your main pseudocode solution.
3. **Which (Answer Correctness Alignment)**: Explain which standard your answer fulfills and why it is factually correct.

This explanation must adhere to the following constraints:
- The total number of sentences must be 6 or more.
- The number of sentences for each of the three items (Why, How, Which) must be balanced. The sentence count of the longest item must be **less than or equal to 3 times** the count of the shortest item.
Failure to meet these conditions will result in a penalty.
"""