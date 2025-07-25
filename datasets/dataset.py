# dataset/dataset.py
# Run only once: generate and tag master problem set with instruction complexity metrics

import json
from pathlib import Path
import sqlite3
import pandas as pd
from datasets import Dataset as HFDataset
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import create_repo
from openai import OpenAI
from dotenv import load_dotenv
from configs import settings

# Initialize OpenAI client
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / ".env")
client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
# 1. Define Dataset Creation Logic
def get_base_datasets(seed: int = 42) -> dict:
    """Loads and samples the base datasets from Hugging Face."""
    ds = {}
    ds["truthful"] = load_dataset("domenicrosati/TruthfulQA", split="train").shuffle(seed).select(range(60)).map(lambda ex: {"category": "domenicrosati/TruthfulQA"})
    ds["newsqa"]   = load_dataset("lucadiliello/newsqa", split="train").shuffle(seed).select(range(40)).map(lambda ex: {"category": "lucadiliello/newsqa"})
    
    ds["arc"]      = load_dataset("ai2_arc", "ARC-Challenge", split="train").shuffle(seed).select(range(30)).map(lambda ex: {"category": "allenai/ai2_arc"})
    
    theorem = load_dataset("TIGER-Lab/TheoremQA", split="test").filter(lambda ex: ex.get("Picture") is None or ex.get("Picture") == "")
    ds["theorem"]  = theorem.shuffle(seed).select(range(30)).map(lambda ex: {"category": "TIGER-Lab/TheoremQA"})
    
    math_algebra = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    math_intermediate_algebra = load_dataset("EleutherAI/hendrycks_math", "intermediate_algebra", split="train")
    math_precalculus = load_dataset("EleutherAI/hendrycks_math", "precalculus", split="train")
    combined_math = concatenate_datasets([math_algebra, math_intermediate_algebra, math_precalculus])
    ds["math"] = combined_math.shuffle(seed).select(range(40)).map(lambda ex: {"category": "EleutherAI/hendrycks_math"})

    return ds

# 2. Shuffles and tag → Integrate
def build_master_set() -> Dataset:
    base = get_base_datasets()
    combined = concatenate_datasets(base.values())
    # Create the final master set with multiple shuffles
    master = combined.shuffle(101).shuffle(202).shuffle(303)

    # Tag instruction complexity via LLM API
    def tag_complexity(ex, idx):
        category = ex.get("category")
        if category == "lucadiliello/newsqa":
            q = ex.get("context") + "\n\n" + ex.get("question")
        elif category == "allenai/ai2_arc":
            choices_obj = ex.get('choices', {'text': [], 'label': []})
            choices_text_parts = [f"{label}. {text}" for label, text in zip(choices_obj.get('label', []), choices_obj.get('text', []))]
            choices_text = "\n".join(choices_text_parts)
            q = f"{ex.get('question', '')}\n\nChoices:\n{choices_text}"
        else:
            q = ex.get("Question") or ex.get("question") or ex.get("problem") or ""
        if category == "domenicrosati/TruthfulQA":
            ans = (
                f"Best Answer: {ex.get('Best Answer','')}\n\n"
                f"Correct Answers: {ex.get('Correct Answers','')}\n\n"
                f"Incorrect Answers: {ex.get('Incorrect Answers','')}"
            )
        else:
            ans = ex.get("Answer") or ex.get("answerKey") or ex.get("answers") or ex.get("solution") or ""

        prompt = f"""
Convert the following question into Python-style pseudocode, guided by the provided choices and the correct answer. Then count its control structures.

Category:
{category}

Question:
{q}

Answer:
{ans}

Return a JSON with these fields:
- pseudocode: the pseudocode solution
- loop_count: number of loops (for/while)
- branch_count: number of branches (if/elif/else)
- variable_count: number of unique variables defined
ex output:
{{"pseudocode":"...","loop_count":2,"branch_count":3,"variable_count":5}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
                
        except Exception:
            data = {"pseudocode": "", "loop_count": -1, "branch_count": -1, "variable_count": -1}
        return {
          **ex,
          "instruction_complexity": data,
        }
    master = master.map(tag_complexity, with_indices=True, batched=False)
    return master

# 3. Mapping candidate names by column
def standardize_and_flatten(ds: Dataset) -> Dataset:
    """Standardizes column names, then for each ex reassigns Question/Answer
    according to category-specific rules, and flattens complexity metrics."""
    rename_map = {}
    candidates = {
        "Question": ["Question", "question", "problem", "context"],
        "Choices": ["Choices", "choices"],
        "Answer": ["Answer", "answerKey", "answers", "solution"],
        "Best Answer": ["Best Answer"],
        "Correct Answers": ["Correct Answers"],
        "Incorrect Answers": ["Incorrect Answers"]
    }
    for std, alts in candidates.items():
        for alt in alts:
            if alt in ds.column_names:
                rename_map[alt] = std
                break
    ds = ds.rename_columns(rename_map)
    
    def _flatten(ex, idx):
        
        qid = idx + 1
        category = ex.get("category")

        if category == "lucadiliello/newsqa":
            # NewsQA: context + question
            raw_q = f"{ex.get('context','')}\n\n{ex.get('question','')}"
        elif category == "allenai/ai2_arc":
            # ARC: question + choices
            choices_obj = ex.get('Choices', {'text': [], 'label': []})
            choices_text_parts = [f"{label}. {text}" for label, text in zip(choices_obj.get('label', []), choices_obj.get('text', []))]
            choices_text = "\n".join(choices_text_parts)
            raw_q = f"{ex.get('Question', '')}\n\nChoices:\n{choices_text}"
        else:
            raw_q = ex.get("Question") or ex.get("question") or ex.get("problem") or ""
        question = str(raw_q)

        ans_list = []
        if category == "domenicrosati/TruthfulQA":
            # TruthfulQA: [Best Answer + Correct Answers + Incorrect Answers]
            parts = []
            best = ex.get("Best Answer")
            corr = ex.get("Correct Answers")
            inc  = ex.get("Incorrect Answers")
            if best:
                parts.append(f"Best Answer: {best}")
            if corr:
                parts.append(f"Correct Answers: {corr}")
            if inc:
                parts.append(f"Incorrect Answers: {inc}")
            ans_list = [str(p) for p in parts]
        elif category == "allenai/ai2_arc":
            ans_list = [str(ex.get("answerKey", ""))]     # ARC: answerKey
        else:
            raw_ans = ex.get("Answer") or ex.get("answerKey") or ex.get("answers") or ex.get("solution") or ""
            if isinstance(raw_ans, list):
                for item in raw_ans:
                    if isinstance(item, dict) and 'text' in item:
                        ans_list.append(str(item['text']))
                    elif isinstance(item, list):
                        for sub in item:
                            ans_list.append(str(sub))
                    else:
                        ans_list.append(str(item))
            else:
                ans_list = [str(raw_ans)]


        # complexity metrics
        ic = ex.get("instruction_complexity", {})
        
        return {
            "QID": qid,
            "category": category,
            "Question": question,
            "Answer": ans_list,
            "pseudocode": ic.get("pseudocode", ""),
            "loop_count": ic.get("loop_count", 0),
            "branch_count": ic.get("branch_count", 0),
            "variable_count": ic.get("variable_count", 0)
        }
    
    ds = ds.map(_flatten, remove_columns=ds.column_names, with_indices=True)
    return ds


# 4) Local Storage & HF Hub Upload
def save_sqlite(ds: Dataset, db_path: Path):
    """Saves the dataset into a local SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
	  CREATE TABLE IF NOT EXISTS problems (
        QID INTEGER PRIMARY KEY,
			Category TEXT,
			Question TEXT,
			Answer TEXT,
			pseudocode TEXT,
			loop_count INTEGER,
			branch_count INTEGER,
			variable_count INTEGER
      )
    """)
    for i, ex in enumerate(ds):
        q = json.dumps(ex.get("Question", []), ensure_ascii=False)
        a = json.dumps(ex.get("Answer", []), ensure_ascii=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO problems
            (QID, category, Question, Answer, pseudocode, loop_count, branch_count, variable_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
				i,
				ex.get("category"),
				q,
				a,
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
        "category",
        "Question",
        "Answer",
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
    hf_ds_ordered.push_to_hub(repo_id)

if __name__ == "__main__":
    # Build and prepare
    master = build_master_set()
    final = standardize_and_flatten(master)

    # 1) Save locally to disk
    local_dir = project_root / "datasets" / "final_problem_set"
    final.save_to_disk(local_dir)

    # 2) Save to SQLite
    save_sqlite(final, project_root / "master.db")

    # 3) Push to Hugging Face Hub
    hf_ds = HFDataset.from_pandas(
        pd.DataFrame(final.to_dict()), preserve_index=False
    )
    push_to_hub(hf_ds, settings.HF_DATASET_REPO)

    print(f"Dataset successfully saved locally and uploaded to https://huggingface.co/datasets/{settings.HF_DATASET_REPO}")