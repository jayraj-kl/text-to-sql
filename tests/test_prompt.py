from text2sql import API_SCHEMA, EXAMPLE_SCHEMA, create_prompt


def test_prompt_layout():
    prompt = create_prompt("How many employees?", "CREATE TABLE t (id INT);")
    assert prompt.startswith("### Instructions:\n")
    assert "### Schema:\nCREATE TABLE t (id INT);\n\n### Question:\nHow many employees?\n\n### Response:\n" in prompt
    assert prompt.endswith("### Response:\nSql Query:")


def test_braces_in_question_are_not_interpreted():
    assert "{braces}" in create_prompt("with {braces}", "s")


def test_api_schema_keeps_its_historical_whitespace():
    assert API_SCHEMA.startswith("\n    CREATE TABLE Employees (\n    EmployeeID INT PRIMARY KEY,")
    assert API_SCHEMA.endswith("*/\n    ")
    assert API_SCHEMA.strip() == EXAMPLE_SCHEMA.strip()
