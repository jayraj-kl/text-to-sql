# /text-to-sql-flask/app/routes.py

import logging
from flask import Blueprint, request, jsonify, current_app

# Import model handling functions and status check
from . import model_handler
# Import the config object to access the hardcoded schema
from config import config

log = logging.getLogger(__name__)

# Create a Blueprint
bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Basic index route."""
    return 'Text-to-SQL API is running!'

@bp.route('/generate-sql', methods=['POST'])
def generate_sql_endpoint():
    """
    API endpoint to generate SQL from a question, using a pre-defined schema.
    """
    status = model_handler.get_model_status()
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

    # --- Modified Section ---
    # Only get the 'question' from the request body
    question = data.get('question')
    if not question:
        # Changed error message: No longer expect 'schema'
        return jsonify({"error": "Missing 'question' in request body."}), 400

    # Use the hardcoded schema from config
    schema = config.DATABASE_SCHEMA
    # --- End Modified Section ---

    log.info(f"Received request: Question='{question[:50]}...' (Using hardcoded schema)")

    try:
        # Pass the hardcoded schema to create_prompt
        prompt = model_handler.create_prompt(question, schema)
        generated_text = model_handler.generate_sql_query(prompt)
        sql_query = model_handler.extract_sql_query(generated_text)

        log.info(f"Generated SQL: {sql_query[:100]}...")
        return jsonify({"sql_query": sql_query})

    except Exception as e:
        log.exception("An error occurred during SQL generation or extraction.")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500


@bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    status = model_handler.get_model_status()
    if status["model_loaded"]:
        return jsonify({"status": "ok", "model_loaded": True}), 200
    else:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "message": f"Model failed to load: {status['error_message']}"
        }), 503