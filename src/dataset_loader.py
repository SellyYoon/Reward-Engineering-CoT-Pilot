# src/dataset_loader.py
# Loads and standardizes the master problem set from the Hugging Face Hub.

from functools import lru_cache
from datasets import load_dataset
from configs import settings

@lru_cache(maxsize=1)
def load_pilot_dataset(split: str):
    return load_dataset(settings.HF_DATASET_REPO, split=split)

def get_reference_counts(config: dict, question_num: int) -> dict:
    ds = load_pilot_dataset(config['split'])
    return {
        "branch_count": ds[question_num].get("branch_count", 0),
        "loop_count":   ds[question_num].get("loop_count", 0),
        "variable_count":ds[question_num].get("variable_count", 0)
    }

# --- Independent Test Block ---
# This allows you to test this module by running `python src/dataset_loader.py`
if __name__ == "__main__":
    print("--- Testing dataset_loader.py ---")
    dataset = load_pilot_dataset()
    
    if dataset:
        print("\n--- First example from the dataset ---")
        first_example = dataset[0]
        print(first_example)
        
        # Verify that standardized column names exist
        assert 'Question' in first_example, "Standardized 'Question' column not found!"
        print("\nTest passed: Dataset loaded and columns seem correct.")
