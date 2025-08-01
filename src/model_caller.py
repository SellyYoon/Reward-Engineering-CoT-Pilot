# src/model_caller.py
# Handles inference calls to both API-based and local LLMs.

import json
import torch
import traceback
from huggingface_hub import login
from typing import Optional, Type, Any, Dict, Union
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from xai_sdk import Client
from xai_sdk.chat import user, system
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache
from configs import settings, schemas
from src import utils
import time
import random
import httpx


import logging
import sys

# --- logger initalization ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Client Initialization ---

# Initialize API clients. The libraries will automatically find the API keys from the environment variables
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
grok_client = Client(api_key=settings.XAI_API_KEY)
genai.configure(api_key=settings.GOOGLE_GENAI_API_KEY)
hf_api_key = settings.HF_API_KEY
if hf_api_key:
    try:
        login(token=hf_api_key)
        logging.info("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        logging.exception(f"Failed to log into Hugging Face Hub: {e}")

# --- Local Model Loading ---
@lru_cache(maxsize=None)
def load_local_model(model_id: str):
    """
    Loads a local Hugging Face model and its tokenizer into VRAM.
    This is a heavy operation and should be done only once per model.
    """
    if not model_id:
        raise ValueError(f"Local model '{model_id}' not found in settings.")

    logging.info(f"Loading local model: {model_id} into VRAM... (This may take a while)")

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
        logging.info(f"-> Model {model_id} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading local model {model_id}: {e}")
        return None, None

# --- Inference Functions ---

def call_openai_api(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    eval_model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    context: Optional[dict] = None
) -> str:
    """Calls the OpenAI API and returns the response text."""
    try:
        temperature = temperature if temperature else settings.TEMPERATURE
        model_id = eval_model_id if eval_model_id else config.get('model_id')
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=settings.MAX_NEW_TOKENS,
            top_p=settings.TOP_P,
            response_format={"type": "json_object"},
        )
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, response, config)
        
        # Return the actual content for the application's use.
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            logging.error(f"OpenAI API response content is empty or malformed for model {model_id}.")
            return json.dumps({"error": "Empty or malformed OpenAI response content"})
    except Exception as e:
        model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
        logging.error(f"Error calling OpenAI API for model {model_id_for_error}: {e}")
        return logging.error("ERROR: OpenAI API call failed.")

def call_anthropic_api(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    eval_model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    context: Optional[dict] = None
) -> str:
    """Calls the Anthropic API and returns the response text."""
    
    max_retries = 5
    initial_delay = 1.0 # seconds
    backoff_factor = 2.0
    max_delay = 60.0 # seconds

    for i in range(max_retries):
        try:
            temperature = temperature if temperature else settings.TEMPERATURE
            model_id = eval_model_id if eval_model_id else config.get('model_id')
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

            log_context = context or {}
            log_context['model_id'] = model_id
            utils.log_raw_response(log_context, response, config)

            if response.content and response.content[0].text:
                return response.content[0].text
            else:
                logging.error(f"Anthropic API response content is empty or malformed for model {model_id}.")
                return json.dumps({"error": "Empty or malformed Anthropic response content"})
            
        except httpx.HTTPStatusError as e:
            # Retry only on 5xx server errors or 429 rate limit errors
            if e.response.status_code >= 500 or e.response.status_code == 429:
                if i < max_retries - 1:
                    delay = min(max_delay, initial_delay * (backoff_factor ** i))
                    jitter = random.uniform(0, delay * 0.1) # Add 10% jitter
                    sleep_time = delay + jitter
                    logging.warning(f"Anthropic API: Status {e.response.status_code}. Retrying in {sleep_time:.2f} seconds... (Attempt {i+1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Error calling Anthropic API for model {model_id}: Max retries exceeded. {e}")
                    return json.dumps({"error": f"Anthropic API call failed after {max_retries} retries: {str(e)}"})
            else:
                # Other HTTP errors (e.g., 4xx client errors) are not retried
                logging.error(f"Error calling Anthropic API for model {model_id}: Non-retryable HTTP error. {e}")
                return json.dumps({"error": f"Anthropic API call failed: {str(e)}"})
            
        except Exception as e:
            model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
            logging.error(f"Error calling Anthropic API for model {model_id_for_error}: {e}")
            return logging.error("ERROR: Anthropic API call failed.")

def call_grok_api(    
    config: dict[str, Any],     
    system_prompt: str,
    user_prompt: str,     
    output_schema: Type[BaseModel],
    eval_model_id: Optional[str] = None,
    temperature: Optional[float] = None, 
    context: Optional[dict] = None
) -> str:
    """
    Calls xAI Grok API and returns the response text.
    """
    
    
    try:
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        model_id = eval_model_id if eval_model_id else config.get('model_id')

        chat = grok_client.chat.create(model="grok-4-0709", temperature=0)
        chat.append(system(system_prompt))
        chat.append(user(user_prompt))

        response, parsed_object = chat.parse(output_schema)

        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, response, config)

        return parsed_object.model_dump_json()

    except Exception as e:
        model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
        logging.error(f"Error calling Grok API for model {model_id_for_error}: {e}")
        return logging.error("ERROR: Grok API call failed.")
    
def call_gemini_api(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    eval_model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    context: Optional[dict] = None
) -> str:
    """Calls Google Gemini API and returns the response text."""
    try:
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        model_id = eval_model_id if eval_model_id else config.get('model_id')

        model = genai.GenerativeModel(model_id, system_instruction=system_prompt)
        
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=temperature,
            top_p=settings.TOP_P
        )

        response = model.generate_content(
            user_prompt,
            generation_config=generation_config
        )
        
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, response, config)
        
        response_text = response.text
        return response_text
    except Exception as e:
        model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
        logging.error(f"Error calling Google Gemini API for model {model_id_for_error}: {e}")
        return json.dumps({"error": "Google Gemini API call failed", "message": str(e)})
    
def call_local_model(
    model,
    config: dict[str, Any],
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    context: Optional[dict] = None,
    temperature: Optional[float] = None
) -> str:
    """Generates
     a response from a loaded local model."""
    try:
        # Determine prompt format based on model_id
        temperature = temperature if temperature else settings.TEMPERATURE        
        model_id = config['model_id']
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
            logging.warning(f"Unknown local model format for {model_id}. Using simple concatenation.")
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
            temperature=temperature,
            top_p=settings.TOP_P,
            top_k=settings.TOP_K,
            repetition_penalty=settings.REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id
        )

        # Decode and return the response
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, response, config)
        return response
    
    except Exception as e:
        logging.error(f"Error during local model inference for {model_id}: {type(e).__name__}: {e}")
        if 'input_ids' in locals() and isinstance(input_ids, torch.Tensor):
            logging.debug(f"input_ids shape: {input_ids.shape}, content: {input_ids}")
            logging.debug(f"formatted_prompt: {messages}...")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.error("!!!      DETAILED INFERENCE ERROR         !!!")
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            traceback.print_exc()
            logging.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return "ERROR: Local model inference failed."

API_DISPATCHER = {
    "api_openai": call_openai_api,
    "api_anthropic": call_anthropic_api,
    "api_xai": call_grok_api,
    "api_google": call_gemini_api,
    "local": call_local_model
}

def dispatch_solver_call(
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    local_models: Optional[dict] = None,
    log_context: Optional[dict] = None,
    output_schema: Optional[Type[BaseModel]] = None
) -> str:
    """
    Finds the solver model config for the given sbx_id and calls it.
    For local models, it uses the pre-loaded model object.
    """
    model_id = config['model_id']
    sbx_id = config['sbx_id']
    temperature = temperature if temperature else settings.TEMPERATURE
    output_schema = output_schema if output_schema else schemas.TaskOutput
    
    # Find model settings corresponding to sbx_id
    model_config = next((m for m in settings.APPLICANT_MODELS if m["sbx_id"] == sbx_id), None)
    if not model_config:
        raise ValueError(f"Solver model with sbx_id '{sbx_id}' not found in settings.")

    model_type = model_config["type"]
    target_api_func = API_DISPATCHER.get(model_type)

    if not target_api_func:
        error_msg = f"'The API call function corresponding to {model_type} cannot be found."
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

    # 1. Prepare a dictionary of common arguments to be passed to API functions.
    api_kwargs = {
        "config": config,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": settings.TEMPERATURE,
        "context": log_context or {}
    }

    # 2. Add ‘special’ arguments according to model type
    if model_type == "api_xai":
        # Grok-specific: Logic that determines different schemas based on conditions
        if config.get('condition') in ('A', 'C'):
            api_kwargs['output_schema'] = schemas.SimpleTaskOutput
        else:
            api_kwargs['output_schema'] = schemas.TaskOutput
            
    elif model_type == "local":
        # Local model only: Pass a preloaded model and tokenizer object
        model_id = config['model_id']
        if not local_models or model_id not in local_models:
            raise ValueError(f"The preloaded local model ({model_id}) cannot be found.")
        
        loaded_model_obj = local_models[model_id]
        api_kwargs["model"] = loaded_model_obj["model"]
        api_kwargs["tokenizer"] = loaded_model_obj["tokenizer"]

    # 3. Call the retry function of utils to execute the final execution.
    return utils.retry_with_backoff(
        api_call_func=target_api_func,
        api_kwargs=api_kwargs
    )