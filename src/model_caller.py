# src/model_caller.py
# Handles inference calls to both API-based and local LLMs.

import os
from typing import Optional
import torch
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from configs import settings

# --- Client Initialization ---

# Initialize API clients. The libraries will automatically find the API keys
# from the environment variables (which should be loaded by an entrypoint script
# like main.py or settings.py using `load_dotenv`).
openai_client = OpenAI()
anthropic_client = Anthropic()

# --- Local Model Loading ---

def load_local_model(model_name: str):
    """
    Loads a local Hugging Face model and its tokenizer into VRAM.
    This is a heavy operation and should be done only once per model.
    """
    model_id = settings.MODEL_LIST["local"].get(model_name)
    if not model_id:
        raise ValueError(f"Local model '{model_name}' not found in settings.")

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

def call_openai_api(model_id: str, system_prompt: str, user_prompt: str) -> str:
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

def call_local_model(model, tokenizer, user_prompt: str) -> str:
    """Generates a response from a loaded local model."""
    try:
        # Format the prompt for the local model
        messages = [{"role": "user", "content": user_prompt}]
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

# --- Independent Test ---
if __name__ == '__main__':
    print("--- Testing model_caller.py ---")
    
    # Test API models
    test_system_prompt = "You are a helpful assistant."
    test_user_prompt = "What is the capital of South Korea?"
    
    print("\nTesting OpenAI API...")
    o4_response = call_openai_api(settings.MODEL_CONFIG["api"]["o4-mini"], test_system_prompt, test_user_prompt)
    print(f"o4-mini response: {o4_response}")

    print("\nTesting Anthropic API...")
    claude_response = call_anthropic_api(settings.MODEL_CONFIG["api"]["claude-4-sonnet"], test_system_prompt, test_user_prompt)
    print(f"Claude Sonnet response: {claude_response}")

    # Test local model (requires GPU and VRAM)
    print("\nTesting Local Model (Llama-3)...")
    llama_model, llama_tokenizer = load_local_model("llama3-8b")
    if llama_model and llama_tokenizer:
        llama_response = call_local_model(llama_model, llama_tokenizer, test_user_prompt)
        print(f"Llama-3 response: {llama_response}")
