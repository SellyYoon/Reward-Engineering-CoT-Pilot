# src/model_caller.py
# Handles inference calls to both API-based and local LLMs.

import torch
import traceback
from typing import Optional
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache
from configs import settings
from src import utils

# --- Client Initialization ---

# Initialize API clients. The libraries will automatically find the API keys from the environment variables
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

# --- Local Model Loading ---
@lru_cache(maxsize=None)
def load_local_model(model_id: str):
    """
    Loads a local Hugging Face model and its tokenizer into VRAM.
    This is a heavy operation and should be done only once per model.
    """
    if not model_id:
        raise ValueError(f"Local model '{model_id}' not found in settings.")

    print(f"Loading local model: {model_id} into VRAM... (This may take a while)")

    # Configuration for loading the model in 4-bit for VRAM efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto", # Automatically use the GPU if available
        )
        print(f"-> Model {model_id} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading local model {model_id}: {e}")
        return None, None

# --- Inference Functions ---

def call_openai_api(model_id: str, temperature: Optional[float], system_prompt: str, user_prompt: str, context: Optional[dict] = None) -> str:
    """Calls the OpenAI API and returns the response text."""
    try:
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_NEW_TOKENS,
            top_p=settings.TOP_P,
            response_format={"type": "json_object"},
        )
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, response)
        return response
    except Exception as e:
        print(f"Error calling OpenAI API for model {model_id}: {e}")
        return "ERROR: OpenAI API call failed."

def call_anthropic_api(model_id: str, temperature: Optional[float], system_prompt: str, user_prompt: str, context: Optional[dict] = None) -> str:
    """Calls the Anthropic API and returns the response text."""
    try:
        response = anthropic_client.messages.create(
            model=model_id,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "{"}
            ],
            temperature=temperature,
            max_tokens=settings.MAX_NEW_TOKENS,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K
        )
        json_response = "{" + response.content[0].text
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, json_response)
        return json_response
        
    except Exception as e:
        print(f"Error calling Anthropic API for model {model_id}: {e}")
        return "ERROR: Anthropic API call failed."

def call_local_model(model, model_id: str, tokenizer, system_prompt: str, user_prompt: str, context: Optional[dict] = None) -> str:
    """Generates a response from a loaded local model."""
    try:
        # Determine prompt format based on model_id
        if "Mistral" in model_id or "mistralai" in model_id:
            # Mistral Instruct format
            messages = f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
        elif "Llama-3" in model_id or "meta-llama" in model_id:
            # Llama 3 Instruct format
            # This format includes special tokens for roles and message boundaries
            messages = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>"
            )
        else:
            # Default to a simple concatenation if model format is unknown
            print(f"Warning: Unknown local model format for {model_id}. Using simple concatenation.")
            messages = f"{system_prompt}\n{user_prompt}"

        inputs = tokenizer(
            messages,
            return_tensors="pt"
        ).to(model.device)

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate the output
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
            repetition_penalty=settings.REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and return the response
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        json_response = response.choices[0].message.content
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, json_response)
        return response
    
    except Exception as e:
        print(f"Error during local model inference for {model_id}: {type(e).__name__}: {e}")
        if 'input_ids' in locals() and isinstance(input_ids, torch.Tensor):
            print(f"DEBUG: input_ids shape: {input_ids.shape}, content: {input_ids}")
            print(f"DEBUG: formatted_prompt: {messages}...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!      DETAILED INFERENCE ERROR         !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return "ERROR: Local model inference failed."

def dispatch_solver_call(
    sbx_id: int, 
    model_id: str,
    temperature: Optional[float],   
    system_prompt: str, 
    user_prompt: str, 
    local_models: Optional[dict] = None,
    log_context: Optional[dict] = None
) -> str:
    """
    Finds the solver model config for the given sbx_id and calls it.
    For local models, it uses the pre-loaded model object.
    """
    # Find model settings corresponding to sbx_id
    model_config = next((m for m in settings.APPLICANT_MODELS if m["sbx_id"] == sbx_id), None)
    if not model_config:
        raise ValueError(f"Solver model with sbx_id '{sbx_id}' not found in settings.")

    model_type = model_config["type"]
    
    # Call the appropriate function depending on the model type
    if model_type == "api_openai":
        return call_openai_api(model_id, temperature, system_prompt, user_prompt, context=log_context)
    elif model_type == "api_anthropic":
        return call_anthropic_api(model_id, temperature, system_prompt, user_prompt, context=log_context)
    elif model_type == "local":
        if not local_models or model_id not in local_models:
            raise ValueError(f"Local model {model_id} was not pre-loaded.")
            
        loaded_model_obj = local_models[model_id]
        model = loaded_model_obj["model"]
        tokenizer = loaded_model_obj["tokenizer"]
        return call_local_model(model, model_id, tokenizer, system_prompt, user_prompt, context=log_context)