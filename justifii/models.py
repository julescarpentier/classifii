import sys

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from justifii.database import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    rationales = db.relationship('Rationale', backref='user', lazy=True)

    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def __repr__(self):
        return '<User username={}>'.format(self.username)

    def __str__(self):
        return self.username


class Text(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fpath = db.Column(db.String(120), unique=True, nullable=False)

    label_id = db.Column(db.Integer, db.ForeignKey('label.id'), nullable=False)
    rationales = db.relationship('Rationale', backref='text', lazy=True)

    def __init__(self, fpath=None, label_id=None):
        self.fpath = fpath
        self.label_id = label_id

    def __repr__(self):
        return '<Text id={}, fpath={}, label={}, label_id={}>'.format(self.id, self.fpath, self.label, self.label_id)

    def __str__(self):
        return self.__repr__()

    def get_content(self):
        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
        with open(self.fpath, **args) as f:
            t = f.read()
            i = t.find('\n\n')  # skip header
            if 0 < i:
                t = t[i:]
            return t

    def get_word_sequence(self):
        return text_to_word_sequence(self.get_content(), lower=False)


class Label(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True, nullable=False)

    texts = db.relationship('Text', backref='label', lazy=True)

    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return '<Label id={}, name={}>'.format(self.id, self.name)

    def __str__(self):
        return self.name


class Rationale(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tokens = db.Column(db.PickleType, nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text_id = db.Column(db.Integer, db.ForeignKey('text.id'), nullable=False)

    def __init__(self, tokens=None):
        self.tokens = tokens

    def __repr__(self):
        return '<Rationale id={}, user={}>'.format(self.id, self.user)

    def __str__(self):
        return self.__repr__()

    def __contains__(self, item):
        if self.tokens is not None:
            return item in self.tokens

        return False

    def get_show(self):
        text = self.text.get_content()
        word_sequence = self.text.get_word_sequence()

        html = ""
        i = 0
        j = 0
        while i < len(text):
            if j < len(word_sequence) and text[i:i + len(word_sequence[j])] == word_sequence[j]:
                if j in self:
                    html += '<span style="background-color: rgb(253, 126, 20);">' + word_sequence[j] + '</span>'
                else:
                    html += word_sequence[j]
                i += len(word_sequence[j])
                j += 1
            else:
                html += text[i]
                i += 1

        return html

    def get_form(self):
        text = self.text.get_content()
        word_sequence = self.text.get_word_sequence()

        html = ""
        i = 0
        j = 0
        while i < len(text):
            if j < len(word_sequence) and text[i:i + len(word_sequence[j])] == word_sequence[j]:
                html += '<div class="form-check form-check-inline mr-0">'
                html += '<input class="form-check-input check-with-label d-none" type="checkbox" ' \
                        'id="token_{}" name="tokens[]" value="{}"{}>' \
                    .format(j, j, " checked" if j in self else "")
                html += '<label class="form-check-label label-for-check" for="token_{}">{}</label>' \
                    .format(j, word_sequence[j])
                html += '</div>'
                i += len(word_sequence[j])
                j += 1
            else:
                html += text[i]
                i += 1

        return html
