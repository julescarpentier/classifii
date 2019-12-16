from flask import (
    Blueprint, g, render_template, abort, request, flash, redirect, url_for
)

from justifii.blueprints.auth import login_required
from justifii.database import db
from justifii.models import Proof, Text

bp = Blueprint('text', __name__, url_prefix='/text')


def get_text(text_id):
    text = Text.query.get(text_id)

    if text is None:
        abort(404, "Text #{} doesn't exist.".format(text_id))

    return text


@bp.route('/')
def index():
    return render_template('text/index.html', texts=Text.query.limit(100).all())


@bp.route('/<int:text_id>')
def show(text_id):
    text = get_text(text_id)

    return render_template('text/show.html', text=text, proofs=text.proofs, tokens=text.get_tokens())


@bp.route('/<int:text_id>/justify', methods=('GET', 'POST'))
@login_required
def justify(text_id):
    text = get_text(text_id)

    existing_proof = Proof.query.filter_by(user_id=g.user.id, text_id=text_id).first()
    new = existing_proof is None
    proof = existing_proof or Proof()

    if request.method == 'POST':
        tokens = request.form.getlist('tokens[]')
        error = None

        if not tokens:
            error = "No tokens selected"

        if error is not None:
            flash(error, 'danger')
        else:
            proof.tokens = tokens
            if new:
                proof.user = g.user
                proof.text = text
                db.session.add(proof)
            db.session.commit()
            return redirect(url_for('text.show', text_id=text_id))

    return render_template('text/justify.html', text=text, proof=proof, tokens=text.get_tokens())
