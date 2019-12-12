from flask import (
    Blueprint, render_template
)

bp = Blueprint('annotate', __name__)


@bp.route('/')
def index():
    return render_template('annotate/index.html')
