"""Extraction of the SQL query from raw model output."""

import logging

from .prompt import RESPONSE_MARKER, SQL_MARKER

log = logging.getLogger(__name__)

# Prefixes that make text after the response marker look like a query/response.
_SQL_PREFIXES = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "Error:")


def extract_sql_query(full_text):
    """
    Extracts the SQL query from the model's full output text,
    searching for the 'Sql Query:' marker (case-insensitive).
    Includes fallbacks for common failure modes.
    """
    marker_pos = full_text.lower().find(SQL_MARKER.lower())

    if marker_pos != -1:
        sql_query = full_text[marker_pos + len(SQL_MARKER):].strip()

        # Clean up potential markdown backticks if the model included them
        if sql_query.startswith("```sql"):
            sql_query = sql_query[len("```sql"):].strip()
        if sql_query.endswith("```"):
            sql_query = sql_query[:-len("```")].strip()
        return sql_query

    if "Error generating SQL:" in full_text:
        # The generation step caught an error; surface that message
        try:
            return full_text.split(RESPONSE_MARKER)[-1].strip()
        except Exception:
            return "Error generating SQL (extraction failed)."

    # Fallback: marker not found. Try everything after "### Response:"
    response_pos = full_text.lower().find(RESPONSE_MARKER.lower())
    if response_pos == -1:
        log.warning(f"Neither '{SQL_MARKER}' nor '{RESPONSE_MARKER}' found in text: '{full_text[:200]}...'")
        return "Query marker not found."

    potential_query = full_text[response_pos + len(RESPONSE_MARKER):].strip()
    if potential_query.upper().startswith(_SQL_PREFIXES):
        log.warning(f"Marker '{SQL_MARKER}' not found. Assuming text after '{RESPONSE_MARKER}' is the query/response.")
        return potential_query

    # Could be explanation text if the model ignored instructions
    log.warning(f"Marker '{SQL_MARKER}' not found and text after '{RESPONSE_MARKER}' doesn't look like SQL.")
    return f"Query marker not found. Response: {potential_query[:200]}..."
