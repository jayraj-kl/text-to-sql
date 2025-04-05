# /text-to-sql-flask/run.py

import logging
from app import create_app
from config import config

# Create the Flask app instance using the factory
app = create_app()

if __name__ == '__main__':
    log = logging.getLogger('werkzeug') # Get Flask's development server logger
    if config.DEBUG:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    print(f"Starting Flask server on http://{config.HOST}:{config.PORT}")
    # Use host='0.0.0.0' to be accessible externally
    # debug=config.DEBUG enables/disables Flask's debug mode (reloader, debugger)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)