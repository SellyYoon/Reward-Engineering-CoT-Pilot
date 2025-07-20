# src/model_caller.py
# Handles inference calls to both API-based and local LLMs.

from typing import Optional
import torch
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs import settings
from functools import lru_cache

# --- Client Initialization ---

# Initialize API clients. The libraries will automatically find the API keys
# from the environment variables (which should be loaded by an entrypoint script
# like main.py or settings.py using `load_dotenv`).
openai_client = OpenAI()
anthropic_client = Anthropic()

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

def call_openai_api(model_id: str, temperature: Optional[float], system_prompt: str, user_prompt: str) -> str:
    """Calls the OpenAI API and returns the response text."""
    try:
        temperature if temperature else settings.TEMPERATURE
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_NEW_TOKENS,
            top_p=settings.TOP_P
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API for model {model_id}: {e}")
        return "ERROR: OpenAI API call failed."

def call_anthropic_api(model_id: str, temperature: Optional[float], system_prompt: str, user_prompt: str) -> str:
    """Calls the Anthropic API and returns the response text."""
    try:
        temperature if temperature else settings.TEMPERATURE
        response = anthropic_client.messages.create(
            model=model_id,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=settings.MAX_NEW_TOKENS,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API for model {model_id}: {e}")
        return "ERROR: Anthropic API call failed."

def call_local_model(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Generates a response from a loaded local model."""
    try:
        # Format the prompt for the local model
        messages = [{"role": "user", "content": system_prompt + user_prompt}]
        input_ids = tokenizer(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate the output
        outputs = model.generate(
            input_ids,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            temperature=settings.TEMPERATURE,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
            repetition_penalty=settings.REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and return the response
        response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response_text
    except Exception as e:
        print(f"Error during local model inference: {e}")
        return "ERROR: Local model inference failed."

def dispatch_solver_call(
    sbx_id: int, 
    temperature: Optional[float],
    system_prompt: str, 
    user_prompt: str, 
) -> str:
    """
    Finds the solver model config for the given sbx_id and calls it.
    For local models, it uses the pre-loaded model object.
    """
    # Find model settings corresponding to sbx_id
    model_config = next((m for m in settings.APPLICANT_MODELS if m["sbx_id"] == sbx_id), None)
    if not model_config:
        raise ValueError(f"Solver model with sbx_id '{sbx_id}' not found in settings.")

    model_id = model_config["id"]
    model_type = model_config["type"]
    
    # Call the appropriate function depending on the model type.
    temperature if temperature else settings.TEMPERATURE
    if model_type == "api_openai":
        return call_openai_api(model_id, temperature, system_prompt, user_prompt)
    elif model_type == "api_anthropic":
        return call_anthropic_api(model_id, temperature, system_prompt, user_prompt)
    elif model_type == "local":
        model, tokenizer = load_local_model(model_id)
        return call_local_model(model, tokenizer, system_prompt, user_prompt)