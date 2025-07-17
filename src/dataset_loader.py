from datasets import load_dataset
from configs.settings import HF_DATASET_REPO

def load_master_set(split="train"):
    return load_dataset(HF_DATASET_REPO, split=split)