"""Starts the Text-to-SQL API on Flask's development server."""

import logging

from app import create_app
from config import config

# Create the Flask app instance using the factory (this loads the model)
app = create_app()

if __name__ == '__main__':
    # Flask's development server logger
    logging.getLogger('werkzeug').setLevel(logging.DEBUG if config.DEBUG else logging.INFO)

    print(f"Starting Flask server on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
