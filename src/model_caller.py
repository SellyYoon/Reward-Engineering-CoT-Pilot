# src/model_caller.py
# Handles inference calls to both API-based and local LLMs.

import json
import torch
import traceback
from huggingface_hub import login
from typing import Optional, Type
from pydantic import BaseModel
from openai import OpenAI
from anthropic import Anthropic
from xai_sdk import Client
from xai_sdk.chat import user, system
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache
from configs import settings, schemas
from src import utils

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
grok_client = Client(api_key=settings.GROK_API_KEY)
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

def call_openai_api(config: dict, system_prompt: str, user_prompt: str, eval_model_id: Optional[str] = None, temperature: Optional[float] = None, context: Optional[dict] = None) -> str:
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
        return response
    except Exception as e:
        model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
        logging.error(f"Error calling OpenAI API for model {model_id_for_error}: {e}")
        return logging.error("ERROR: OpenAI API call failed.")

def call_anthropic_api(config: dict, system_prompt: str, user_prompt: str, eval_model_id: Optional[str] = None, temperature: Optional[float] = None, context: Optional[dict] = None) -> str:
    """Calls the Anthropic API and returns the response text."""
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
        json_response = "{" + response.content[0].text
        log_context = context or {}
        log_context['model_id'] = model_id
        utils.log_raw_response(log_context, json_response, config)
        return json_response
        
    except Exception as e:
        model_id_for_error = eval_model_id or config.get('model_id', 'unknown')
        logging.error(f"Error calling Anthropic API for model {model_id_for_error}: {e}")
        return logging.error("ERROR: Anthropic API call failed.")

def call_grok_api(
    config: dict, 
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
    
def call_local_model(model, config: dict, tokenizer, system_prompt: str, user_prompt: str, context: Optional[dict] = None, temperature: Optional[float] = None,) -> str:
    """Generates a response from a loaded local model."""
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

def dispatch_solver_call(
    config: dict,
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
    
    # Call the appropriate function depending on the model type
    if model_type == "api_openai":
        return call_openai_api(config, system_prompt, user_prompt, temperature, context=log_context)
    elif model_type == "api_anthropic":
        return call_anthropic_api(config, system_prompt, user_prompt, temperature, context=log_context)
    elif model_type == "api_xai":
        if output_schema:
            final_schema = output_schema
        elif config['condition'] in ('A', 'C'):
            final_schema = schemas.SimpleTaskOutput
        else: # Conditions 'B', 'D'
            final_schema = schemas.TaskOutput
        return call_grok_api(config, system_prompt, user_prompt, temperature=temperature, context=log_context, output_schema=final_schema)
    elif model_type == "local":
        if not local_models or model_id not in local_models:
            raise ValueError(f"Local model {model_id} was not pre-loaded.")
            
        loaded_model_obj = local_models[model_id]
        model = loaded_model_obj["model"]
        tokenizer = loaded_model_obj["tokenizer"]
        return call_local_model(model, config, tokenizer, system_prompt, user_prompt, context=log_context)