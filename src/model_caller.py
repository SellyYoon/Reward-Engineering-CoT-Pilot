import os, openai, anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")

def call_o4mini(prompt, **gen_args):
    return openai.ChatCompletion.create(
        model="o4-mini-2025-04-16",
        messages=[{"role":"user","content":prompt}],
        **gen_args
    )

def call_claude(prompt, **gen_args):
    return anthropic.Completion.create(
        model="claude-sonnet-4",
        prompt=prompt,
        **gen_args
    )

def load_local_llm(model_name="llama-3-8b"):
    quant = BitsAndBytesConfig(load_in_4bit=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant, device_map="auto"
    )
    return model, tok

def call_local(model, tokenizer, prompt, **gen_args):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **gen_args)
    return tokenizer.decode(out[0], skip_special_tokens=True)
