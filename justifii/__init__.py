import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI='sqlite:////' + os.path.join(app.instance_path, 'justifii.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
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

    from . import database
    database.init_app(app)

    from justifii.blueprints import default
    app.register_blueprint(default.bp)
    app.add_url_rule('/', endpoint='index')

    from justifii.blueprints import auth
    app.register_blueprint(auth.bp)

    from justifii.blueprints import text
    app.register_blueprint(text.bp)

    from justifii.blueprints import proof
    app.register_blueprint(proof.bp)

    return app