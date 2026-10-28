"""Prompt construction for the SQL generation task."""

# Marker the model is asked to start its answer with; extraction looks for it.
SQL_MARKER = "Sql Query:"
RESPONSE_MARKER = "### Response:"

PROMPT_TEMPLATE = """### Instructions:
Given the database schema and a question, generate the SQL query that answers the question.
Respond ONLY with the SQL query. Start your response with the marker 'Sql Query:'. Do not include explanations or markdown formatting like ```sql.

### Schema:
{schema}

### Question:
{question}

### Response:
Sql Query:"""


def create_prompt(question, schema):
    """Creates a structured prompt for the SQL generation task."""
    # Use .format() for safer substitution than f-strings in this context
    return PROMPT_TEMPLATE.format(schema=schema, question=question)
