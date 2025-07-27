# dataset/test_dataset.py
# Run only once: generate and tag master problem set with instruction complexity metrics

import json
from pathlib import Path
import sqlite3
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo
from dotenv import load_dotenv
from configs import settings

# Initialize OpenAI client
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")
hfrepo = settings.HF_DATASET_REPO
    
# 1. Define Dataset Creation Logic
def get_base_datasets(seed: int = settings.DEFAULT_SEED) -> dict:
    """Loads and samples the base datasets from Hugging Face."""
    ds = load_dataset(hfrepo, split="train").select(range(10))
    return ds

# 2) Local Storage & HF Hub Upload
def save_sqlite(ds: Dataset, db_path: Path):
    """Saves the dataset into a local SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
	  CREATE TABLE IF NOT EXISTS problems (
        QID INTEGER PRIMARY KEY,
			Category TEXT,
			Question TEXT,
			Answer TEXT,
            reasoning_steps TEXT,
			pseudocode TEXT,
			loop_count INTEGER,
			branch_count INTEGER,
			variable_count INTEGER
      )
    """)
    for i, ex in enumerate(ds):
        q = json.dumps(ex.get("Question", []), ensure_ascii=False)
        a = json.dumps(ex.get("Answer", []), ensure_ascii=False)
        r_steps = json.dumps(ex.get("reasoning_steps", []), ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO problems
            (QID, Category, Question, Answer, reasoning_steps, pseudocode, loop_count, branch_count, variable_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
				ex.get("QID"),
				ex.get("Category"),
				q,
				a,
                r_steps,
				ex.get("pseudocode"),
				ex.get("loop_count"),
				ex.get("branch_count"),
				ex.get("variable_count"),
            ),
        )
    conn.commit()
    conn.close()

def push_to_hub(ds: Dataset, repo_id: str):
    """Pushes the Arrow/Parquet dataset to Hugging Face Hub."""
    create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    
    # Define the desired column order for the Hugging Face Dataset
    desired_columns = [
        "QID",
        "Category",
        "Question",
        "Answer",
        "reasoning_steps",
        "pseudocode",
        "loop_count",
        "branch_count",
        "variable_count"
    ]
    
    df = pd.DataFrame(ds.to_dict())
    
    # Ensure all desired columns exist in the DataFrame before reindexing
    # If a column is missing, it will be filled with NaN
    for col in desired_columns:
        if col not in df.columns:
            df[col] = None # Or appropriate default value

    df = df[desired_columns] # Reindex to enforce order

    hf_ds_ordered = HFDataset.from_pandas(df, preserve_index=False)
    hf_ds_ordered.push_to_hub(repo_id, split='test')

if __name__ == "__main__":

    test = get_base_datasets()
    
    # 1) Save locally to disk
    local_dir = project_root / "datasets" / "test_problem_set"
    test.save_to_disk(local_dir)

    # 2) Save to SQLite
    save_sqlite(test, project_root / "tester.db")

    # 3) Push to Hugging Face Hub
    hf_ds = HFDataset.from_pandas(
        pd.DataFrame(test.to_dict()), preserve_index=False
    )
    push_to_hub(hf_ds, hfrepo)

    print(f"Dataset successfully saved locally and uploaded to https://huggingface.co/datasets/{hfrepo}")