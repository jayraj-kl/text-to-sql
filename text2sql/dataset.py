"""Batch generation over a CSV dataset with `question` and `schema` columns."""

import logging

import pandas as pd
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

OUTPUT_COLUMN = "generated_sql_query"


class DatasetProcessor:
    """Handles loading data from CSV and running batch generation."""

    def __init__(self, path, pipeline, sample_size=None):
        self.path = path
        self.pipeline = pipeline
        self.sample_size = sample_size
        self.df = self._load_data()

    def _load_data(self):
        """Loads DataFrame from CSV path."""
        log.info(f"Loading data from: {self.path}")
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError:
            log.error(f"Data file not found at {self.path}")
            return None
        except Exception as e:
            log.error(f"Error loading data: {e}")
            return None

        if self.sample_size:
            log.info(f"Taking a sample of {self.sample_size} rows.")
            df = df.head(self.sample_size)
        df[OUTPUT_COLUMN] = None
        log.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def _process_row(self, natural_language, schema):
        """Generates and extracts SQL for a single DataFrame row."""
        if pd.isna(natural_language) or pd.isna(schema):
            return "Error: Missing question or schema"
        try:
            return self.pipeline.generate(natural_language, schema)
        except Exception as e:
            log.error(f"Error processing question '{str(natural_language)[:50]}...': {e}")
            return f"Error during processing: {e}"

    def generate_queries(self):
        """Applies the generation process row-by-row to the loaded DataFrame."""
        if self.df is None:
            log.error("DataFrame not loaded. Cannot generate queries.")
            return None

        log.info(f"Starting SQL query generation for {len(self.df)} rows...")
        tqdm.pandas(desc="Generating SQL")
        self.df[OUTPUT_COLUMN] = self.df.progress_apply(
            lambda row: self._process_row(row["question"], row["schema"]),
            axis=1,
        )
        log.info("SQL query generation complete.")
        return self.df
