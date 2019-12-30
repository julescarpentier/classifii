import os

import click
from flask.cli import with_appcontext
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    from justifii.models import Text, Label

    db.drop_all()
    click.echo('Dropped all tables.')
    db.create_all()
    click.echo('Created all tables.')

    base_dir = 'data'
    text_data_dir = os.path.join(base_dir, '20news-18828')

    click.echo('Processing text dataset.')

    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label = Label(name)
            db.session.add(label)
            db.session.commit()
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    text = Text(fpath, label.id)
                    db.session.add(text)
    db.session.commit()

    click.echo('Found {} texts.'.format(Text.query.count()))


def init_app(app):
    db.init_app(app)
    app.cli.add_command(init_db_command)
