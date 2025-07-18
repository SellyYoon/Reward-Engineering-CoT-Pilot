# models.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_local_llm(model_name="llama-3-8b"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )
    return model, tokenizer