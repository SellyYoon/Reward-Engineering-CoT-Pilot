from models import load_local_llm, call_o4_mini, call_claude
from dataset import get_datasets
from evaluator import compute_rewards
from utils import setup_logging, backup_and_reset
from datasets import load_from_disk

def get_final_dataset():
    # Hugging Face Hub에 있는 내 비공개 데이터셋을 바로 불러오기
    # (실행 전 터미널에서 huggingface-cli login 필요)
    repo_id = "SellyA/reward-pilot-dataset"
    master_set = load_dataset(repo_id, split="train") 
    return master_set

def main():
    setup_logging()
    local_model, local_tok = load_local_llm()
    
    # 2. Load the master problem set from the Hub
    repo_id = "SellyA/reward-pilot-dataset"
    master_set = load_dataset(repo_id, split="train") 
    print(f"Loaded master problem set from {repo_id}")
    
    # 3. Run experiment loops
    datasets = get_datasets()
    for block in range(1, 32):
        print(f"--- Starting Block {block} ---")
        for i, example in enumerate(master_set):
            prompt = example["Question"]
            
            # 이 부분에서 block 또는 조건(A,B,C,D)에 따라
            # o4-mini, Claude, Llama, Mistral을 번갈아 호출하는 로직이 필요합니다.
            # response = call_o4_mini(prompt, ...)
            
            # 결과 계산 및 로그 기록
            # compute_rewards(...)
            # log(...)

        # 한 회차가 끝나면 백업 및 리셋
        backup_and_reset()
        
if __name__=="__main__":
    main()
    
    
# from src.dataset_loader import *
# from src.model_caller import *
# from src.evaluator import evaluate
# from src.utils import setup_logging, backup_and_reset

# def run_block(block_id, model_funcs, datasets):
#     setup_logging()
#     for ds_name, ds in datasets.items():
#         for ex in ds:
#             prompt = make_prompt(ex, cfg[block_id])  # configs.prompts.py
#             res = model_funcs[block_id % len(model_funcs)](prompt, **gen_args)
#             eval_res = evaluate(ex, res, ex["steps"], ex["answer"])
#             logging.info(f"{block_id},{ds_name},{eval_res}")

# if __name__=="__main__":
#     models = [call_o4mini, call_claude, *(load_local_llm())]
#     datasets = {
#         "truthful": load_truthfulqa(),
#         # …
#     }
#     for block in range(1, 33):
#         run_block(block, models, datasets)
#         backup_and_reset(block)
