from flask import (
    Blueprint, render_template, jsonify
)

from justifii.models import Rationale, Label, Text, User

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    return render_template('dashboard/index.html', text_count=Text.query.count(),
                           rationale_count=Rationale.query.count())


@bp.route('/_get_texts_labels')
def get_texts_labels():
    labels = [label.name for label in Label.query.all()]
    data = [Text.query.filter(Text.label_id == label.id).count() for label in Label.query.all()]
    return jsonify(labels=labels, data=data)


@bp.route('/_get_rationales_labels')
def get_rationales_labels():
    labels = [label.name for label in Label.query.all()]
    data = [Text.query.filter(Text.label_id == label.id, Text.rationales.any()).count() for label in Label.query.all()]
    return jsonify(labels=labels, data=data)


@bp.route('/_get_users_participation')
def get_users_participation():
    labels = [user.username for user in User.query.all()]
    data = [len(user.rationales) for user in User.query.all()]
    return jsonify(labels=labels, data=data)
