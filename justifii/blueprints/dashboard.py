from flask import (
    Blueprint, render_template
)

from justifii.models import Rationale

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    return render_template('dashboard/index.html', rationale_count=Rationale.query.count())
