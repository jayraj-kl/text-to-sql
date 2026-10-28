"""End-to-end question -> SQL pipeline around a loaded model."""

from dataclasses import dataclass
from typing import Any

from .extraction import extract_sql_query
from .generation import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, generate_sql_query
from .model import load_model
from .prompt import create_prompt


@dataclass
class TextToSQL:
    """Bundles a tokenizer/model pair with generation settings."""
    tokenizer: Any
    model: Any
    device: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_pretrained(cls, model_name, device, device_map="auto", **generation_settings):
        """Loads the model (see `load_model`) and wraps it. Raises ModelLoadError on failure."""
        tokenizer, model = load_model(model_name, device, device_map=device_map)
        return cls(tokenizer, model, device, **generation_settings)

    def generate(self, question, schema):
        """Returns the SQL query (or an error/fallback message) for `question` against `schema`."""
        prompt = create_prompt(question, schema)
        generated_text = generate_sql_query(
            self.tokenizer,
            self.model,
            prompt,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return extract_sql_query(generated_text)
