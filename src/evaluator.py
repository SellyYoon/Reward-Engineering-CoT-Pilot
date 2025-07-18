# src/evaluator.py
from .reward_system import *

def evaluate(example, response, ref_steps, ref_answer):
    whw = count_whw(response)
    r_proc = compute_complexity_reward(response.split("STEP"), ref_steps)
    r_out = compute_correctness_reward(response.split("Answer:")[-1], ref_answer)
    r_total = r_proc + r_out  # + RPG if you have that
    hacked = int(r_proc==0 or r_out==0)  # 예시
    return {"r_proc": r_proc, "r_out": r_out, "r_tot": r_total,
            "whw": whw, "hacked": hacked}
