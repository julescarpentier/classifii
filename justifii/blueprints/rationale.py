from flask import (
    Blueprint, g, render_template, url_for, abort, flash, redirect
)

from justifii.database import db_session
from justifii.blueprints.auth import login_required
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
    return render_template('rationale/index.html', rationales=Rationale.query.limit(1000).all())


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
