from flask import (
    Blueprint, g, render_template, abort, request, flash, redirect, url_for, jsonify
)

from justifii.database import db_session
from justifii.blueprints.auth import login_required
from justifii.models import Rationale, Text

bp = Blueprint('text', __name__, url_prefix='/text')


def get_text(text_id):
    text = Text.query.get(text_id)

    if text is None:
        abort(404, "Text #{} doesn't exist.".format(text_id))

    return text


@bp.route('/')
def index():
    return render_template('text/index.html', texts=Text.query.all())


@bp.route('/_get_texts', methods=('GET',))
def get_texts():
    data = [{
        'id': text.id,
        'fpath': text.fpath,
        'label': text.label.name,
        'show_url': url_for('text.show', text_id=text.id),
        'justify_url': url_for('text.justify', text_id=text.id),
    } for text in Text.query.all()]

    return jsonify(data=data)


@bp.route('/<int:text_id>')
def show(text_id):
    text = get_text(text_id)

    return render_template('text/show.html', text=text)


@bp.route('/<int:text_id>/justify', methods=('GET', 'POST'))
@login_required
def justify(text_id):
    text = get_text(text_id)

    existing_rationale = Rationale.query.filter_by(user_id=g.user.id, text_id=text_id).first()
    new = existing_rationale is None
    rationale = existing_rationale or Rationale()

    if new:
        rationale.user = g.user
        rationale.text = text

    if request.method == 'POST':
        tokens = request.form.getlist('tokens[]')
        error = None

        if not tokens:
            error = "No tokens selected"

        if error is not None:
            flash(error, 'danger')
        else:
            rationale.tokens = [int(token) for token in tokens]
            if new:
                db_session.add(rationale)
            db_session.commit()
            return redirect(url_for('text.show', text_id=text_id))

    return render_template('text/justify.html', text=text, rationale=rationale)
