"""Flask application factory for the Text-to-SQL API."""

import logging

from flask import Flask

from config import config as default_config

from . import routes
from .model_service import EXTENSION_KEY, ModelService


def create_app(cfg=default_config, model_service=None):
    """
    Application factory function.

    Loads the model described by `cfg` unless a ready `model_service` is
    passed in (useful for tests).
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    app = Flask(__name__)
    app.config.from_object(cfg)
    log.info(f"Flask app created. Debug mode: {cfg.DEBUG}")
    log.info(f"Target device: {cfg.DEVICE}")
    log.debug(f"Using database schema:\n{cfg.DATABASE_SCHEMA}")

    app.register_blueprint(routes.bp)
    log.info("Registered main blueprint.")

    if model_service is None:
        model_service = ModelService.from_config(cfg)
    app.extensions[EXTENSION_KEY] = model_service

    status = model_service.get_status()
    if not status["model_loaded"]:
        log.critical(f"CRITICAL FAILURE DURING STARTUP: Model failed to load - {status['error_message']}")
    else:
        log.info("Model status check passed during startup.")

    return app
