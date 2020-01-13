from flask import (
    Blueprint, g, render_template, url_for, abort, flash, redirect, jsonify
)

from justifii.blueprints.auth import login_required
from justifii.database import db_session
from justifii.models import Rationale

bp = Blueprint('rationale', __name__, url_prefix='/rationale')


def get_rationale(rationale_id, check_owner=True):
    rationale = Rationale.query.get(rationale_id)

    if rationale is None:
        abort(404, "Rationale #{} doesn't exist.".format(rationale_id))

    if check_owner and rationale.user_id != g.user.id:
        abort(403, "Rationale #{} is not yours.".format(rationale_id))

    return rationale


@bp.route('/', methods=('GET',))
def index():
    return render_template('rationale/index.html')


@bp.route('/_get_rationales', methods=('GET',))
def get_rationales():
    data = [{
        'id': rationale.id,
        'user': rationale.user.username,
        'text': rationale.text.fpath,
        'tokens': rationale.tokens,
        'show_url': url_for('rationale.show', rationale_id=rationale.id),
        'edit_url': url_for('text.justify', text_id=rationale.text.id),
        'delete_url': url_for('rationale.delete', rationale_id=rationale.id),
    } for rationale in Rationale.query.all()]

    return jsonify(data=data)


@bp.route('/<int:rationale_id>', methods=('GET',))
def show(rationale_id):
    rationale = get_rationale(rationale_id, check_owner=False)

    return render_template('rationale/show.html', rationale=rationale, tokens=rationale.text.get_word_sequence())


@bp.route('/<int:rationale_id>/delete', methods=('POST',))
@login_required
def delete(rationale_id):
    rationale = get_rationale(rationale_id)
    db_session.delete(rationale)
    db_session.commit()
    flash("Deleted Rationale #{}".format(rationale_id), 'success')

    return redirect(url_for('rationale.index'))
