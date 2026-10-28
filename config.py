"""Flask API settings. Every value can be overridden with an environment variable."""

import os

import torch

from text2sql.generation import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
from text2sql.model import DEFAULT_MODEL_NAME
from text2sql.schema import API_SCHEMA


class Config:
    """Base configuration settings."""
    # --- Model Configuration ---
    MODEL_NAME = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")

    # --- Generation Parameters ---
    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
    TEMPERATURE = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))

    # --- Hardware Settings ---
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda:0" if USE_GPU else "cpu"
    # On CPU the transformers loaders keep the model where it is instead of using accelerate
    DEVICE_MAP = "auto" if USE_GPU else None

    # --- Server Configuration ---
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

    # --- Database Schema sent to the model with every question ---
    DATABASE_SCHEMA = API_SCHEMA


config = Config()
