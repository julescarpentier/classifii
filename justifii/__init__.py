import os

from flask import Flask

from justifii.database import db_session


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

    from justifii.blueprints import dashboard
    app.register_blueprint(dashboard.bp)
    app.add_url_rule('/', endpoint='index')

    from justifii.blueprints import auth
    app.register_blueprint(auth.bp)

    from justifii.blueprints import text
    app.register_blueprint(text.bp)

    from justifii.blueprints import rationale
    app.register_blueprint(rationale.bp)

    return app
