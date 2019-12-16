import click
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy
from os import listdir
from os.path import isdir, isfile, join

db = SQLAlchemy()


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    from justifii.models import Text

    db.drop_all()
    db.create_all()
    click.echo('Created tables.')

    dataset_path = 'data/20news-18828'

    for label in listdir(dataset_path):
        label_path = join(dataset_path, label)
        if isdir(label_path):
            for file in listdir(label_path):
                file_path = join(label_path, file)
                if isfile(file_path):
                    text = Text(file_path, label)
                    db.session.add(text)

    db.session.commit()
    click.echo('Populated tables.')


def init_app(app):
    db.init_app(app)
    app.cli.add_command(init_db_command)
