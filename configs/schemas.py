# config/schemas.py
# Defines the Pydantic models for structured outputs from LLMs.

from typing import List
from pydantic import BaseModel, Field

class WHWDescription(BaseModel):
    """Pydantic model for the Why/How/Which explanation structure."""
    why: str = Field(description="Why your reasoning process aligns with the underlying ethical targets.")
    how: str = Field(description="How you derived the answer step-by-step through your pseudocode.")
    which: str = Field(description="Which standard your answer fulfills and why it is factually correct.")

class TaskOutput(BaseModel):
    """
    Pydantic model for the final output structure of the experiment,
    including complexity analysis and the WHW description.
    """
    pred_answer: str = Field(description="The factual, direct final answer to the question.")
    pred_reasoning_steps: List[str] = Field(description="The step-by-step reasoning process based on concrete facts.")
    pred_pseudocode: str = Field(description="The pseudocode implementation of the reasoning process.")
    pred_loop_count: int = Field(description="The total number of for/while statements.")
    pred_branch_count: int = Field(description="The total number of if/elif/else/except/finally statements.")
    pred_variable_count: int = Field(description="The total count of unique variable names first assigned a value.")
    whw_description: WHWDescription = Field(description="A detailed explanation based on the Why/How/Which rules.")

class SimpleTaskOutput(BaseModel):
    """
    A simpler Pydantic model for tasks that do not require
    pseudocode, complexity, or WHW explanations.
    """
    pred_answer: str = Field(description="The factual, direct final answer to the question.")
    pred_reasoning_steps: List[str] = Field(description="The step-by-step reasoning process based on concrete facts.")