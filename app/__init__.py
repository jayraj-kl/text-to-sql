# /text-to-sql-flask/app/__init__.py

import logging
from flask import Flask
from config import config # Import the instantiated config

def create_app():
    """Application factory function."""
    app = Flask(__name__)
    app.config.from_object(config) # Load config from config object

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)
    log.info(f"Flask app created. Debug mode: {config.DEBUG}")
    log.info(f"Target device: {config.DEVICE}")


    # --- Import and register blueprints ---
    # Import routes here to avoid circular dependencies
    # The import of model_handler happens *inside* routes or when routes calls it
    from . import routes
    app.register_blueprint(routes.bp)
    log.info("Registered main blueprint.")

    # --- Perform initial model status check ---
    # model_handler is imported implicitly when routes.py imports it,
    # or we can import it explicitly here to check status at startup.
    from . import model_handler
    status = model_handler.get_model_status()
    if not status["model_loaded"]:
        log.critical(f"CRITICAL FAILURE DURING STARTUP: Model failed to load - {status['error_message']}")
    else:
        log.info("Model status check passed during startup.")


    return app