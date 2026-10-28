# Text-2-SQL

Turn natural-language questions into SQL with a locally hosted LLM
([`TheBloke/sqlcoder-7B-GPTQ`](https://huggingface.co/TheBloke/sqlcoder-7B-GPTQ) by default).

You can use it in two ways:

- **REST API** (Flask): `POST /generate-sql` with a question. The API answers against a fixed database schema.
- **Command line**: interactive question/answer mode, or batch generation over a CSV dataset.

Both use the same core library (`text2sql/`), so the prompt, generation settings and SQL extraction are identical.

## Project structure

```
Text-2-SQL/
├── text2sql/              # Core library (no Flask dependency)
│   ├── schema.py          # Example database schema given to the model
│   ├── prompt.py          # Prompt template + create_prompt()
│   ├── extraction.py      # extract_sql_query(): pulls the SQL out of raw model output
│   ├── model.py           # Device detection, HF login, model/tokenizer loading with fallbacks
│   ├── generation.py      # generate_sql_query(): runs the model on a prompt
│   ├── pipeline.py        # TextToSQL: question + schema -> SQL in one call
│   ├── dataset.py         # DatasetProcessor: batch generation over a CSV
│   └── cli.py             # `python -m text2sql` entry point
├── app/                   # Flask API
│   ├── __init__.py        # create_app() factory
│   ├── model_service.py   # Loads the model once, tracks load status/errors
│   └── routes.py          # /, /health, /generate-sql
├── config.py              # API settings (env-var overridable)
├── run.py                 # Starts the API server
├── tests/                 # pytest suite (uses fakes, no model download needed)
├── dataset/train.csv      # Sample dataset: question, query, schema
└── notebook.ipynb         # Original exploratory Colab notebook
```

### How a request flows

```
question ──► create_prompt(question, schema) ──► generate_sql_query(model) ──► extract_sql_query(text) ──► SQL
              text2sql/prompt.py                  text2sql/generation.py        text2sql/extraction.py
```

`TextToSQL.generate()` in `text2sql/pipeline.py` runs this chain. The API's `ModelService` and the CLI both call it.

### Model loading

`text2sql.model.load_model()` tries these in order and keeps the first that works:

1. **AutoGPTQ**: loads the pre-quantized GPTQ weights.
2. **BitsAndBytes 4-bit** (NF4, double quantization).
3. **fp16 without quantization**: needs much more memory.

If the tokenizer has no pad token, `[PAD]` is added and the model's embeddings are resized to match. If every attempt fails, `ModelLoadError` is raised. The API reports that error on `/health`, and the CLI exits.

## Installation

Requires Python 3.11. A CUDA GPU is strongly recommended: the model runs on CPU, but very slowly.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# auto_gptq ships CUDA-specific wheels; adjust cu118 to match your CUDA version
pip install -r requirements.txt --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

If `auto_gptq` can't be installed, the loader falls back to BitsAndBytes or fp16 automatically.

## Configuration

Set these with environment variables. The API reads them in `config.py`; the CLI reads `MODEL_NAME` and `HUGGING_FACE_TOKEN` and takes the rest as flags.

| Variable             | Default                     | Description                                  |
|----------------------|-----------------------------|----------------------------------------------|
| `MODEL_NAME`         | `TheBloke/sqlcoder-7B-GPTQ` | Hugging Face model id                        |
| `HUGGING_FACE_TOKEN` | *(unset)*                   | Token for gated/private models (optional)    |
| `MAX_NEW_TOKENS`     | `512`                       | Max tokens generated per query               |
| `TEMPERATURE`        | `0.1`                       | Sampling temperature (`0` = greedy decoding) |
| `FLASK_HOST`         | `0.0.0.0`                   | API bind address                             |
| `FLASK_PORT`         | `5000`                      | API port                                     |
| `FLASK_DEBUG`        | `False`                     | Flask debug mode (`true`/`1`/`t` to enable)  |

### Changing the database schema

The model only knows about the tables you give it. To use your own database, edit
`EXAMPLE_SCHEMA` in `text2sql/schema.py` and add your `CREATE TABLE` statements.
Include primary and foreign keys so the model can work out joins.

- **API**: uses `API_SCHEMA`, which is `EXAMPLE_SCHEMA` with its original indentation kept. To use a different schema, point `Config.DATABASE_SCHEMA` in `config.py` at your own string.
- **CLI**: use `--schema-file path/to/schema.sql`. Without it, the CLI uses `EXAMPLE_SCHEMA`.

## Usage

### REST API

```bash
python run.py
```

The model loads at startup, which can take a few minutes the first time while the weights download.

| Method | Path            | Description                                      |
|--------|-----------------|--------------------------------------------------|
| GET    | `/`             | Liveness text                                    |
| GET    | `/health`       | `200` if the model loaded, `503` with the error otherwise |
| POST   | `/generate-sql` | Body `{"question": "..."}` → `{"sql_query": "..."}` |

```bash
curl -X POST http://localhost:5000/generate-sql \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the average salary per department?"}'
```

Error responses are JSON with an `error` key:

- `400`: the body isn't JSON, or `question` is missing.
- `503`: the model failed to load.
- `500`: generation failed.

### Command line

```bash
# Interactive mode: type questions, `quit`/`exit` to stop
python -m text2sql
python -m text2sql --schema-file my_schema.sql --temperature 0

# Batch mode over a CSV with `question` and `schema` columns
python -m text2sql --dataset dataset/train.csv --sample-size 10 --output results.csv
```

Batch mode adds a `generated_sql_query` column next to the dataset's reference `query` column.

### As a library

```python
from text2sql import EXAMPLE_SCHEMA
from text2sql.model import default_device
from text2sql.pipeline import TextToSQL

pipeline = TextToSQL.from_pretrained("TheBloke/sqlcoder-7B-GPTQ", default_device())
print(pipeline.generate("Who are the five highest paid employees?", EXAMPLE_SCHEMA))
```

## Dataset

`dataset/train.csv` has three columns:

- `question`: natural-language question.
- `query`: the reference SQL.
- `schema`: the `CREATE TABLE` context for that question.

Use it with the CLI's batch mode.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests cover prompt building, SQL extraction and every API endpoint. They use a fake model, so nothing is downloaded.

## Known limitation

The prompt's instruction line contains the text `'Sql Query:'`. Model output includes the prompt, so
`extract_sql_query` finds that first occurrence. The returned "query" therefore starts with the rest of the prompt and only then the generated SQL.
This behavior is kept on purpose so results stay identical to earlier versions, and a test pins it
(`tests/test_extraction.py`). To fix it, search for the *last* marker or decode only the newly generated tokens. Both fixes change the output.
