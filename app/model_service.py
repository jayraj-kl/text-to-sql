"""Holds the API's loaded Text-to-SQL pipeline and its load status."""

import logging

from text2sql.model import ModelLoadError, login_to_hub
from text2sql.pipeline import TextToSQL

log = logging.getLogger(__name__)

# Key under which the service is stored in app.extensions
EXTENSION_KEY = "text2sql"


class ModelService:
    """Loads the model once and serves SQL generation requests."""

    def __init__(self, pipeline=None, error_message=None):
        self.pipeline = pipeline
        self.error_message = error_message

    @classmethod
    def from_config(cls, cfg):
        """Loads the model described by `cfg`. Failures are recorded, not raised."""
        login_to_hub(cfg.HUGGING_FACE_TOKEN)
        try:
            pipeline = TextToSQL.from_pretrained(
                cfg.MODEL_NAME,
                cfg.DEVICE,
                device_map=cfg.DEVICE_MAP,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                temperature=cfg.TEMPERATURE,
            )
        except ModelLoadError as e:
            return cls(error_message=str(e))
        log.info("Model and Tokenizer initialization complete.")
        return cls(pipeline=pipeline)

    @property
    def model_loaded(self):
        return self.pipeline is not None

    def get_status(self):
        """Returns the loading status and any error."""
        return {"model_loaded": self.model_loaded, "error_message": self.error_message}

    def generate(self, question, schema):
        if not self.model_loaded:
            raise RuntimeError("Model or Tokenizer not loaded.")
        return self.pipeline.generate(question, schema)
