from text2sql import create_prompt, extract_sql_query


def test_extracts_text_after_marker():
    assert extract_sql_query("abc Sql Query: SELECT 1") == "SELECT 1"


def test_marker_is_case_insensitive_and_strips_markdown_fence():
    assert extract_sql_query("sql query:  ```sql\nSELECT 3;\n```  ") == "SELECT 3;"


def test_falls_back_to_text_after_response_marker_when_it_looks_like_sql():
    text = "### Response:\nWITH a AS (SELECT 1) SELECT * FROM a"
    assert extract_sql_query(text) == "WITH a AS (SELECT 1) SELECT * FROM a"


def test_non_sql_after_response_marker_is_reported_and_truncated():
    result = extract_sql_query("### Response: hello " + "x" * 300)
    assert result.startswith("Query marker not found. Response: hello ")
    assert result.endswith("...")
    assert len(result) == len("Query marker not found. Response: ") + 200 + len("...")


def test_no_markers_at_all():
    assert extract_sql_query("garbage") == "Query marker not found."


def test_generation_error_message_is_surfaced():
    assert extract_sql_query("Error generating SQL: x ### Response: tail") == "tail"


def test_full_model_output_matches_marker_inside_prompt_instructions():
    # The prompt's instructions mention 'Sql Query:' before the real marker at the end,
    # so extraction from (prompt + completion) starts there. Pinned to keep results stable.
    full_output = create_prompt("q", "schema") + " SELECT 1;"
    assert extract_sql_query(full_output).startswith("'. Do not include explanations")
    assert extract_sql_query(full_output).endswith("Sql Query: SELECT 1;")
