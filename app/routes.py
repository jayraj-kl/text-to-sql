"""HTTP endpoints of the Text-to-SQL API."""

import logging

from flask import Blueprint, current_app, jsonify, request

from .model_service import EXTENSION_KEY

log = logging.getLogger(__name__)

bp = Blueprint('main', __name__)


def _model_service():
    return current_app.extensions[EXTENSION_KEY]


@bp.route('/')
def index():
    """Basic index route."""
    return 'Text-to-SQL API is running!'


@bp.route('/generate-sql', methods=['POST'])
def generate_sql_endpoint():
    """API endpoint to generate SQL from a question, using the configured schema."""
    service = _model_service()
    status = service.get_status()
    if not status["model_loaded"]:
        log.error("Model not loaded, cannot process request.")
        return jsonify({"error": f"Model is not available. {status['error_message']}"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400
    except Exception as e:
        log.error(f"Failed to parse request JSON: {e}")
        return jsonify({"error": f"Invalid JSON format: {e}"}), 400

    question = data.get('question')
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    schema = current_app.config["DATABASE_SCHEMA"]
    log.info(f"Received request: Question='{question[:50]}...' (Using configured schema)")

    try:
        sql_query = service.generate(question, schema)
        log.info(f"Generated SQL: {sql_query[:100]}...")
        return jsonify({"sql_query": sql_query})
    except Exception as e:
        log.exception("An error occurred during SQL generation or extraction.")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    status = _model_service().get_status()
    if status["model_loaded"]:
        return jsonify({"status": "ok", "model_loaded": True}), 200
    return jsonify({
        "status": "error",
        "model_loaded": False,
        "message": f"Model failed to load: {status['error_message']}"
    }), 503
