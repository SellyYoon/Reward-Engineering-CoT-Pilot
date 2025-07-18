# src/utils.py
import logging, shutil, subprocess
import json

def setup_logging():
    logging.basicConfig(filename="logs/pilot.log", level=logging.INFO)

def backup_and_reset(block_id):
    shutil.make_archive(f"backup/block_{block_id}", 'zip', "logs/")
    subprocess.run(["git", "reset", "--hard", "HEAD"])

def parse_complexity_json(llm_output: str) -> dict:
    try:
        start = llm_output.rfind('{')
        end   = llm_output.rfind('}') + 1
        return json.loads(llm_output[start:end])
    except Exception:
        return {"loop_count": -1, "branch_count": -1, "variable_count": -1}
