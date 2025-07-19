import os, json
import configs.settings

# 백업/로그 폴더는 Docker-Compose에서 volume 으로 마운트된 경로를 사용
BACKUP_DIR = configs.settings.BACKUP_DIR
STATE_FILE = os.path.join(BACKUP_DIR, f"{os.environ['MODEL_NAME']}_session.json")

def _load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE, "r"))
    return {"trial": 0}

def _save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def next_session():
    """
    반환:
      session_id: int (e.g. 105)
      trial     : int (1부터 시작, 실행 시마다 ++)
    """
    sbx_id = int(os.environ["SBX_ID"])
    state = _load_state()
    state["trial"] += 1
    _save_state(state)

    trial = state["trial"]
    session_id = sbx_id * 100 + trial
    return session_id, trial
