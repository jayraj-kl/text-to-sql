"""
Command-line interface (replaces the old notebook.py script).

    python -m text2sql                                  # interactive mode
    python -m text2sql --dataset dataset/train.csv \\
                       --sample-size 10 --output out.csv  # batch mode
"""

import argparse
import logging
import os
import sys

from .generation import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
from .model import DEFAULT_MODEL_NAME, ModelLoadError, default_device, login_to_hub
from .pipeline import TextToSQL
from .schema import EXAMPLE_SCHEMA

EXIT_COMMANDS = ("quit", "exit")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m text2sql", description="Generate SQL from natural-language questions.")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--hf-token", default=os.environ.get("HUGGING_FACE_TOKEN"),
                        help="Hugging Face token (default: $HUGGING_FACE_TOKEN)")
    parser.add_argument("--schema-file", help="File with CREATE TABLE statements (default: built-in example schema)")
    parser.add_argument("--dataset", help="CSV with 'question' and 'schema' columns; enables batch mode")
    parser.add_argument("--sample-size", type=int, help="Only process the first N rows of the dataset")
    parser.add_argument("--output", help="Where to write batch results (CSV)")
    return parser.parse_args(argv)


def run_interactive(pipeline, schema):
    print("--- Database Schema ---")
    print(schema)
    print("-" * 50)
    print("\n--- Interactive SQL Query Mode ---")
    print("Type your natural language question and press Enter.")
    print(f"Type {' or '.join(repr(c) for c in EXIT_COMMANDS)} to stop.")
    print("-" * 50)

    while True:
        try:
            user_question = input("Your Question: ").strip()
            if not user_question:
                continue
            if user_question.lower() in EXIT_COMMANDS:
                print("Exiting interactive mode.")
                break

            print("Generating SQL...")
            sql_query = pipeline.generate(user_question, schema)
            print("-" * 30)
            print(f"Generated SQL:\n{sql_query}")
            print("-" * 30 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again or type 'quit'/'exit' to stop.")


def run_batch(pipeline, path, sample_size=None, output=None):
    from .dataset import DatasetProcessor  # pandas is only needed in batch mode

    df = DatasetProcessor(path, pipeline, sample_size=sample_size).generate_queries()
    if df is None:
        return 1
    if output:
        df.to_csv(output, index=False)
        print(f"Results written to {output}")
    else:
        print(df.to_string())
    return 0


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = default_device()
    print(f"Using device: {device}")
    login_to_hub(args.hf_token)

    try:
        pipeline = TextToSQL.from_pretrained(
            args.model_name, device,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
    except ModelLoadError as e:
        print(f"{e}. Cannot proceed.")
        return 1

    if args.dataset:
        return run_batch(pipeline, args.dataset, args.sample_size, args.output)

    schema = EXAMPLE_SCHEMA
    if args.schema_file:
        with open(args.schema_file, encoding="utf-8") as f:
            schema = f.read()
    run_interactive(pipeline, schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
