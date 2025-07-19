# dataset/dataset.py
# Run only once: generate and tag master problem set with instruction complexity metrics

import os
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import create_repo
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from configs import settings

# Initialize OpenAI client
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")
client = OpenAI()

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

# 2. Shuffles and tag → Integrate
def build_master_set():
    base = get_base_datasets()
    combined = concatenate_datasets(base.values())
    # Create the final master set with multiple shuffles
    master = combined.shuffle(101).shuffle(202).shuffle(303)

    # Tag instruction complexity via LLM API
    def tag_complexity(example, idx):
        q = example.get("Question") or example.get("question") or example.get("problem")
        """Calls an LLM to generate pseudocode and calculate complexity metrics."""
        prompt = f"""
        Convert the following question into Python-style pseudocode and count its control structures.

        Question:
        {q}

        Return a JSON with:
        - pseudocode: the pseudocode solution
        - loop_count: number of loops (for/while)
        - branch_count: number of branches (if/elif/else)
        - variable_count: number of unique variables defined
        Example:
        {{"pseudocode":"...","loop_count":2,"branch_count":3,"variable_count":5}}
        """

        try:
            response = client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
                
        except:
            data = {"pseudocode":"", "loop_count":-1, "branch_count":-1, "variable_count":-1}
        return {
          **example,
          "instruction_complexity": data,
          "category": example["_data_page_url"].split("/")[-1] if "_data_page_url" in example else None
        }
    master = master.map(tag_complexity, with_indices=True, batched=False, remove_columns=[])
    return master

# 3. Mapping candidate names by column
def standardize_and_flatten(ds: Dataset) -> Dataset:
    # 컬럼명 통일
    rename = {}
    for std, alts in {
        "Question": ["Question", "question", "problem", "context"],
        "Choices": "choices",
        "Answer":   ["Answer", "answerKey", "answers", "label", "solution", ["Best Answer", "Correct Answers"]],
        "Incorrect Answer": "Incorrect answer"
    }.items():
        for cand in alts:
            if cand in ds.column_names:
                rename[cand] = std
                break
    ds = ds.rename_columns(rename)
    # Move the complexity attribute to the top level.
    def _flatten(ex):
        ic = ex.get("instruction_complexity", {})
        ex["loop_count"] = ic.get("loop_count", 0)
        ex["branch_count"] = ic.get("branch_count", 0)
        ex["variable_count"] = ic.get("variable_count", 0)
        return ex
    ds = ds.map(_flatten)
    return ds

# 4) Local Storage & HF Hub Upload
def save_and_push(ds: Dataset, local_path: Path, repo_id: str):
    ds.save_to_disk(local_path)
    create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    ds.push_to_hub(repo_id)

if __name__ == "__main__":
    # Build → Standardize → Save
    master = build_master_set()
    final = standardize_and_flatten(master)
    local_dir = project_root / "datasets" / "final_problem_set"
    save_and_push(final, local_dir, settings.HF_DATASET_REPO)
    
    # Stored in local DB
    from sqlite3 import connect
    import json
    conn = connect(project_root / "master.db")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS problems (
        QID INTEGER PRIMARY KEY,
        category     TEXT,
        Question     TEXT,
        Choices      TEXT,
        Answer       TEXT,
        branch_count INTEGER,
        loop_count   INTEGER,
        variable_count INTEGER
      )
    """)
    for i, ex in enumerate(final):
        conn.execute(
          "INSERT OR REPLACE INTO problems VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
          (
            i,
            ex.get("category"),
            ex["Question"],
            json.dumps(ex.get("Choices", [])),
            ex["Answer"],
            ex["branch_count"],
            ex["loop_count"],
            ex["variable_count"],
          )
        )
    conn.commit()
    conn.close()
print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{settings.HF_DATASET_REPO}")