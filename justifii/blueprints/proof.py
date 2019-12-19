from flask import (
    Blueprint, g, render_template, url_for, abort, request, flash, redirect
)

from justifii.blueprints.auth import login_required
from justifii.database import db
from justifii.models import Proof

bp = Blueprint('proof', __name__, url_prefix='/proof')


def get_proof(proof_id, check_owner=True):
    proof = Proof.query.get(proof_id)

    if proof is None:
        abort(404, "Proof #{} doesn't exist.".format(proof_id))

    if check_owner and proof.user_id != g.user.id:
        abort(403, "Proof #{} is not yours.".format(proof_id))

    return proof


@bp.route('/', methods=('GET',))
def index():
    return render_template('proof/index.html', proofs=Proof.query.limit(100).all())


@bp.route('/<int:proof_id>', methods=('GET',))
def show(proof_id):
    proof = get_proof(proof_id, check_owner=False)

    return render_template('proof/show.html', proof=proof, tokens=proof.text.get_tokens())


@bp.route('/<int:proof_id>/edit', methods=('GET', 'POST'))
@login_required
def edit(proof_id):
    proof = get_proof(proof_id)

    if request.method == 'POST':
        tokens = request.form['tokens']
        error = None

        if not tokens:
            error = "No tokens selected"

        if error is not None:
            flash(error, 'danger')
        else:
            proof.tokens = tokens
            db.session.commit()
            return redirect(url_for('proof.show', id=proof_id))

    return render_template('proof/edit.html', proof=proof, tokens=proof.text.get_tokens())


@bp.route('/<int:proof_id>/delete', methods=('POST',))
@login_required
def delete(proof_id):
    proof = get_proof(proof_id)
    db.session.delete(proof)
    db.session.commit()
    flash("Deleted Proof #{}".format(proof_id), 'success')

    return redirect(url_for('proof.index'))
