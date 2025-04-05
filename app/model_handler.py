# /text-to-sql-flask/app/model_handler.py

import gc
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import login

# Import config settings
from config import config

# --- Module-level variables to hold the loaded model and tokenizer ---
_tokenizer = None
_model = None
_model_load_error = None # Store potential loading error message

# --- Logging ---
log = logging.getLogger(__name__) # Use Flask's logger or specific module logger

# --- Helper Functions ---
def flush_memory():
    """Clears GPU cache and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _load_model_internal():
    """Internal function to load model and tokenizer, called once."""
    global _tokenizer, _model, _model_load_error
    log.info(f"Attempting to load model: {config.MODEL_NAME} on device: {config.DEVICE}")

    if config.HUGGING_FACE_TOKEN:
        log.info("Logging into Hugging Face Hub...")
        try:
            login(token=config.HUGGING_FACE_TOKEN)
            log.info("Hugging Face login successful.")
        except Exception as e:
            log.warning(f"Hugging Face login failed: {e}") # Non-critical usually

    device_map_setting = "auto" if config.USE_GPU else None
    target_device = config.DEVICE

    try:
        _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)
        log.info(f"Tokenizer loaded for {config.MODEL_NAME}.")
        if _tokenizer.pad_token is None:
            log.info("Adding padding token '[PAD]' to tokenizer.")
            _tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        log.error(f"Failed to load tokenizer for {config.MODEL_NAME}: {e}")
        _model_load_error = f"Tokenizer loading failed: {e}"
        return # Stop loading if tokenizer fails

    loaded_model = None
    # Attempt 1: AutoGPTQ
    try:
        log.info("Attempting to load model with AutoGPTQ...")
        loaded_model = AutoGPTQForCausalLM.from_quantized(
            config.MODEL_NAME, use_safetensors=True, trust_remote_code=True,
            device=target_device, use_triton=False, quantize_config=None,
            inject_fused_attention=False, inject_fused_mlp=False,
        )
        if _tokenizer.pad_token == '[PAD]':
            loaded_model.resize_token_embeddings(len(_tokenizer))
        log.info("AutoGPTQ model loaded successfully.")
    except Exception as e_gptq:
        log.warning(f"AutoGPTQ loading failed: {e_gptq}. Trying BitsAndBytes...")
        # Attempt 2: BitsAndBytes 4-bit
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            loaded_model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME, quantization_config=quantization_config,
                torch_dtype=torch.float16, device_map=device_map_setting,
                trust_remote_code=True,
            )
            if _tokenizer.pad_token == '[PAD]':
                loaded_model.resize_token_embeddings(len(_tokenizer))
            log.info("BitsAndBytes 4-bit model loaded successfully.")
        except Exception as e_bnb:
            log.warning(f"BitsAndBytes loading also failed: {e_bnb}. Trying without quantization...")
            # Attempt 3: No Quantization
            try:
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_NAME, torch_dtype=torch.float16,
                    device_map=device_map_setting, trust_remote_code=True,
                )
                if _tokenizer.pad_token == '[PAD]':
                    loaded_model.resize_token_embeddings(len(_tokenizer))
                log.info("Non-quantized model loaded.")
            except Exception as e_plain:
                log.error(f"CRITICAL: Failed to load model in any configuration: {e_plain}")
                _model_load_error = f"Model loading failed entirely: {e_plain}"
                return # Stop if all methods fail

    _model = loaded_model # Assign successfully loaded model to global
    log.info("Model and Tokenizer initialization complete.")


def create_prompt(question, schema):
    """Creates a structured prompt for the SQL generation task."""
    system_prompt = """### Instructions:
Given the database schema and a question, generate the SQL query that answers the question.
Respond ONLY with the SQL query. Start your response with the marker 'Sql Query:'. Do not include explanations or markdown formatting like ```sql.

### Schema:
{schema}

### Question:
{question}

### Response:
Sql Query:"""
    return system_prompt.format(schema=schema, question=question)


def generate_sql_query(prompt):
    """Generates the SQL query using the loaded model."""
    if not _tokenizer or not _model:
        raise RuntimeError("Model or Tokenizer not loaded.") # Should be caught by health check

    log.info("Starting SQL generation...")
    flush_memory()
    inputs = _tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(config.DEVICE)
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE if config.TEMPERATURE > 0 else None,
        "do_sample": config.TEMPERATURE > 0,
        "num_return_sequences": 1,
        "pad_token_id": _tokenizer.eos_token_id,
        "eos_token_id": _tokenizer.eos_token_id
    }
    generated_text = f"Error: Generation failed"
    try:
        with torch.no_grad():
            outputs = _model.generate(**generation_kwargs)
        full_response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_response
        log.info("SQL generation completed.")
    except Exception as e:
        log.error(f"Error during generation: {e}")
        generated_text = f"{prompt}Error generating SQL: {e}"
    finally:
        flush_memory()
    return generated_text


def extract_sql_query(full_text):
    """Extracts the SQL query from the model's full output text."""
    marker = "Sql Query:"
    response_marker = "### Response:"
    marker_pos = full_text.lower().find(marker.lower())
    if marker_pos != -1:
        actual_start = marker_pos + len(marker)
        sql_query = full_text[actual_start:].strip()
        if sql_query.startswith("```sql"): sql_query = sql_query[len("```sql"):].strip()
        if sql_query.endswith("```"): sql_query = sql_query[:-len("```")].strip()
        return sql_query
    elif "Error generating SQL:" in full_text:
        try: return full_text.split(response_marker)[-1].strip()
        except: return "Error generating SQL (extraction failed)."
    else:
        response_pos = full_text.lower().find(response_marker.lower())
        if response_pos != -1:
            potential_query = full_text[response_pos + len(response_marker):].strip()
            if potential_query.upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "Error:")):
                log.warning(f"Marker '{marker}' not found. Assuming text after '{response_marker}' is the query/response.")
                return potential_query
            else:
                log.warning(f"Marker '{marker}' not found and text after '{response_marker}' doesn't look like SQL.")
                return f"Query marker not found. Response: {potential_query[:200]}..."
        else:
            log.warning(f"Neither '{marker}' nor '{response_marker}' found in text: '{full_text[:200]}...'")
            return "Query marker not found."

def get_model_status():
    """Returns the loading status and any error."""
    return {
        "model_loaded": _model is not None and _tokenizer is not None,
        "error_message": _model_load_error
    }

# --- Trigger Model Loading when this module is imported ---
_load_model_internal()