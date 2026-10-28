"""API tests using a fake model service, so no model is downloaded or loaded."""

import pytest

from app import create_app
from app.model_service import ModelService
from config import config


class FakePipeline:
    def __init__(self, result="SELECT 1;", error=None):
        self.result, self.error, self.calls = result, error, []

    def generate(self, question, schema):
        self.calls.append((question, schema))
        if self.error:
            raise self.error
        return self.result


def make_client(service):
    return create_app(config, model_service=service).test_client()


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def client(pipeline):
    return make_client(ModelService(pipeline=pipeline))


@pytest.fixture
def unloaded_client():
    return make_client(ModelService(error_message="Model loading failed entirely: boom"))


def test_index(client):
    assert client.get("/").get_data(as_text=True) == "Text-to-SQL API is running!"


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True}


def test_health_when_model_failed(unloaded_client):
    response = unloaded_client.get("/health")
    assert response.status_code == 503
    assert response.get_json() == {
        "status": "error",
        "model_loaded": False,
        "message": "Model failed to load: Model loading failed entirely: boom",
    }


def test_generate_sql_uses_configured_schema(client, pipeline):
    response = client.post("/generate-sql", json={"question": "How many employees?"})
    assert response.status_code == 200
    assert response.get_json() == {"sql_query": "SELECT 1;"}
    assert pipeline.calls == [("How many employees?", config.DATABASE_SCHEMA)]


@pytest.mark.parametrize("kwargs, error", [
    ({"json": {}}, "Request body must be JSON."),
    ({"json": {"question": ""}}, "Missing 'question' in request body."),
    ({"json": {"other": 1}}, "Missing 'question' in request body."),
])
def test_generate_sql_bad_requests(client, kwargs, error):
    response = client.post("/generate-sql", **kwargs)
    assert response.status_code == 400
    assert response.get_json() == {"error": error}


def test_generate_sql_non_json_body(client):
    response = client.post("/generate-sql", data="not json")
    assert response.status_code == 400
    assert response.get_json()["error"].startswith("Invalid JSON format: 415")


def test_generate_sql_when_model_failed(unloaded_client):
    response = unloaded_client.post("/generate-sql", json={"question": "q"})
    assert response.status_code == 503
    assert response.get_json() == {"error": "Model is not available. Model loading failed entirely: boom"}


def test_generate_sql_internal_error():
    client = make_client(ModelService(pipeline=FakePipeline(error=ValueError("bad"))))
    response = client.post("/generate-sql", json={"question": "q"})
    assert response.status_code == 500
    assert response.get_json() == {"error": "An internal error occurred: bad"}
