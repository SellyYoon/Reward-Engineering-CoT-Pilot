def compute_complexity_reward(pred_steps, ref_steps, theta=0.02):
    rel_err = abs(len(pred_steps) - len(ref_steps)) / len(ref_steps)
    return 1 if rel_err < theta else 0

def compute_correctness_reward(pred_answer, ref_answer, theta=0.02):
    err = abs(float(pred_answer) - float(ref_answer)) / float(ref_answer)
    return 1 if err < theta else 0

def count_whw(response):
    # “Why:”, “How:”, “What:” 등 키워드로 문장 개수 세기
    counts = {"why":0, "how":0, "what":0}
    for line in response.lower().splitlines():
        for k in counts:
            if line.strip().startswith(f"{k}:"):
                counts[k] += 1
    return counts
