"""Text generation with a loaded model."""

import logging

import torch

from .model import flush_memory

log = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1


def generate_sql_query(tokenizer, model, prompt, device,
                       max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=DEFAULT_TEMPERATURE):
    """
    Generates a response for `prompt` and returns the full decoded text
    (prompt included, which extraction relies on). Generation errors are
    returned embedded in the text rather than raised.
    """
    log.info("Starting SQL generation...")
    flush_memory()

    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True).to(device)
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature if temperature > 0 else None,  # 0 = deterministic greedy
        "do_sample": temperature > 0,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,  # Use EOS token for padding during generation
        "eos_token_id": tokenizer.eos_token_id,
    }

    generated_text = "Error: Generation failed"
    try:
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        log.info("SQL generation completed.")
    except Exception as e:
        log.error(f"Error during generation: {e}")
        generated_text = f"{prompt}Error generating SQL: {e}"
    finally:
        flush_memory()

    return generated_text
