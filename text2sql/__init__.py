"""
Core Text-to-SQL library, shared by the Flask API (`app/`) and the CLI.

Modules:
    schema      example database schema
    prompt      prompt template and construction
    extraction  pulling the SQL query out of raw model output
    model       device detection, Hugging Face login, model/tokenizer loading
    generation  running the model on a prompt
    pipeline    TextToSQL: question + schema -> SQL in one call
    dataset     batch generation over a CSV
    cli         interactive / batch command-line entry point

Only the lightweight modules are imported here so that `import text2sql`
does not pull in torch/transformers.
"""

from .extraction import extract_sql_query
from .prompt import create_prompt
from .schema import API_SCHEMA, EXAMPLE_SCHEMA

__all__ = ["API_SCHEMA", "EXAMPLE_SCHEMA", "create_prompt", "extract_sql_query"]
