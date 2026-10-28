"""Hardware detection, Hugging Face login and model/tokenizer loading."""

import gc
import logging

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "TheBloke/sqlcoder-7B-GPTQ"
PAD_TOKEN = "[PAD]"


class ModelLoadError(RuntimeError):
    """Raised when the tokenizer or model cannot be loaded."""


def default_device():
    """Returns the first GPU if CUDA is available, otherwise the CPU."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def flush_memory():
    """Clears GPU cache and runs Python's garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def login_to_hub(token):
    """Logs into the Hugging Face Hub. Failure is logged, not raised."""
    if not token:
        log.warning("Hugging Face token not provided. Downloads might fail for gated models.")
        return
    log.info("Logging into Hugging Face Hub...")
    try:
        login(token=token)
        log.info("Hugging Face login successful.")
    except Exception as e:
        log.warning(f"Hugging Face login failed: {e}")


def load_tokenizer(model_name):
    """Loads the tokenizer, adding a pad token if the model has none (common for Llama models)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    log.info(f"Tokenizer loaded for {model_name}.")
    if tokenizer.pad_token is None:
        log.info(f"Adding padding token '{PAD_TOKEN}' to tokenizer.")
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    return tokenizer


# --- Loading strategies, tried in order until one succeeds ---

def _load_gptq(model_name, device, device_map):
    # Imported lazily so the package still works where auto_gptq isn't installed
    from auto_gptq import AutoGPTQForCausalLM
    return AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        trust_remote_code=True,  # Caution: Only use if you trust the source
        device=device,           # Load directly to the target device
        use_triton=False,
        quantize_config=None,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )


def _load_bnb_4bit(model_name, device, device_map):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )


def _load_unquantized(model_name, device, device_map):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )


_LOAD_STRATEGIES = (
    ("AutoGPTQ", _load_gptq),
    ("BitsAndBytes 4-bit", _load_bnb_4bit),
    ("no quantization", _load_unquantized),
)


def load_model(model_name, device, device_map="auto"):
    """
    Loads the specified model and tokenizer.
    Attempts AutoGPTQ first, then falls back to BitsAndBytes 4-bit,
    then attempts loading without quantization.

    `device` is where AutoGPTQ places the model; `device_map` is passed to
    the transformers loaders (None keeps them on the CPU).
    Returns (tokenizer, model) or raises ModelLoadError.
    """
    log.info(f"Attempting to load model: {model_name} on device: {device}")
    try:
        tokenizer = load_tokenizer(model_name)
    except Exception as e:
        log.error(f"Failed to load tokenizer for {model_name}: {e}")
        raise ModelLoadError(f"Tokenizer loading failed: {e}") from e

    last_error = None
    for name, loader in _LOAD_STRATEGIES:
        log.info(f"Attempting to load model with {name}...")
        try:
            model = loader(model_name, device, device_map)
            # Resize embeddings if the pad token was added
            if tokenizer.pad_token == PAD_TOKEN:
                model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            log.warning(f"Loading with {name} failed: {e}")
            last_error = e
            continue
        log.info(f"Model loaded successfully with {name}.")
        return tokenizer, model

    log.error(f"CRITICAL: Failed to load model in any configuration: {last_error}")
    raise ModelLoadError(f"Model loading failed entirely: {last_error}") from last_error
