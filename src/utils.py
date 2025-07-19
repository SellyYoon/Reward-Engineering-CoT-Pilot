# src/utils.py
import logging, os, json, shutil
from datetime import datetime

def setup_logging():
    logging.basicConfig(filename="logs/pilot.log", level=logging.INFO)

def backup_and_reset(state_file: str, backup_dir: str) -> int:
    """
    1) state_file(JSON) 을 읽어 trial 값을 +1
    2) backup_dir 에 docker 스냅샷과 데이터를 통채로 백업하고 docker를 초기 셋팅 상태로 Rollback
    3) 수정된 trial 값을 state_file 에 덮어쓰기
    4) 증가된 trial 반환
    """
    # 1. 현재 상태 로드
    if os.path.exists(state_file):
        state = json.load(open(state_file))
    else:
        state = {"trial": 0}

    # 2. 백업
    trial_new = state["trial"] + 1
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_name = f"trial_{state['trial']:03d}_{timestamp}.json"
    shutil.copy(state_file, os.path.join(backup_dir, backup_name)) if os.path.exists(state_file) else None

    # 3. 상태 업데이트
    state["trial"] = trial_new
    with open(state_file, "w") as f:
        json.dump(state, f)

    return trial_new

def parse_complexity_json(llm_output: str) -> dict:
    try:
        start = llm_output.rfind('{')
        end   = llm_output.rfind('}') + 1
        return json.loads(llm_output[start:end])
    except Exception:
        return {"loop_count": -1, "branch_count": -1, "variable_count": -1}
