# dataset.py
# Run only once

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import create_repo

# 1. Define Dataset Creation Logic
def get_base_datasets(seed=42):
    """Loads and samples the base datasets from Hugging Face."""
    ds = {}
    ds["truthful"] = load_dataset("domenicrosati/TruthfulQA", split="validation").shuffle(seed).select(range(60))
    ds["newsqa"]   = load_dataset("lucadiliello/newsqa", split="train").shuffle(seed).select(range(40))
    ds["theorem"]  = load_dataset("TIGER-Lab/TheoremQA", split="test").shuffle(seed).select(range(30))
    ds["arc"]      = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed).select(range(30))
    
    math_algebra = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    math_intermediate_algebra = load_dataset("EleutherAI/hendrycks_math", "intermediate_algebra", split="train")
    math_precalculus = load_dataset("EleutherAI/hendrycks_math", "precalculus", split="train")
    combined_math = concatenate_datasets([math_algebra, math_intermediate_algebra, math_precalculus])
    ds["math"] = combined_math.shuffle(seed).select(range(40))

    return ds

# 2. Generate the base 200-question set
all_datasets_dict = get_base_datasets(seed=42)
problem_set = concatenate_datasets(all_datasets_dict.values())
print("Base problem set created with 200 questions.")

# 3. Create the final master set with multiple shuffles
final_master_set = problem_set.shuffle(seed=101).shuffle(seed=202).shuffle(seed=303)
print("Final master set created through multiple shuffles.")

# 4. Save the final master set to a local directory (Safer)
local_save_path = "./datasets"
final_master_set.save_to_disk(local_save_path)
print(f"Master set saved locally to: {local_save_path}")

# 5. Upload the final set to Hugging Face Hub
repo_id = "SellyA/reward-pilot-dataset" 
create_repo(repo_id, repo_type="dataset", private=True)
final_master_set.push_to_hub(repo_id)

print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")