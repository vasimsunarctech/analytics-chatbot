from flask import Flask
from dotenv import load_dotenv
import os
import plotly.io as pio
pio.renderers.default = "json"   # or "svg" / "png" if cairosvg etc. set up hai


def create_app():
    load_dotenv()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    app.config["DATABASE"] = os.getenv("DATABASE", "./database.db")

    # register internals
    from .db import init_db_command, init_app_db
    init_app_db(app)
    app.cli.add_command(init_db_command)

    # pages (auth + views)
    from .app import bp as pages_bp
    app.register_blueprint(pages_bp)

    # API endpoints
    from .api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # filters/context
    from .utils import format_timestamp_value
    @app.template_filter("format_timestamp")
    def _fmt_ts(value):
        return format_timestamp_value(value)

    @app.context_processor
    def inject_user():
        from flask import session
        return {
            "logged_in": session.get("user_id") is not None,
            "username": session.get("username"),
            "full_name": session.get("full_name"),
        }

    return app
