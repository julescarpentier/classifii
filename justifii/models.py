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
    file_path = db.Column(db.String(120), unique=True, nullable=False)
    label = db.Column(db.String(80), nullable=False)

    rationales = db.relationship('Rationale', backref='text', lazy=True)

    def __init__(self, file_path=None, label=None):
        self.file_path = file_path
        self.label = label

    def __repr__(self):
        return '<Text id={}, file_path={}, label={}>'.format(self.id, self.file_path, self.label)

    def __str__(self):
        return '<Text id={}, file_path={}, label={}>'.format(self.id, self.file_path, self.label)

    def get_keras_tokens(self):
        with open(self.file_path) as f:
            return text_to_word_sequence(f.read(), lower=False)


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
        return '<Rationale id={}, user={}>'.format(self.id, self.user)

    def tokens_contain(self, i):
        if self.tokens is None:
            return False

        return str(i) in self.tokens

    def get_show(self):
        with open(self.text.file_path) as f:
            content = f.read()
            tokens = self.text.get_keras_tokens()
            html = ""
            i = 0
            j = 0
            while i < len(content):
                if j < len(tokens):
                    if content[i:i + len(tokens[j])] == tokens[j]:
                        if self.tokens_contain(j):
                            html += '<span class="font-weight-bold">' + tokens[j] + '</span>'
                        else:
                            html += tokens[j]
                        i += len(tokens[j])
                        j += 1
                    else:
                        html += content[i]
                        i += 1
                else:
                    html += content[i]
                    i += 1

            return html

    def get_form(self):
        with open(self.text.file_path) as f:
            content = f.read()
            tokens = self.text.get_keras_tokens()
            html = ""
            i = 0
            j = 0
            while i < len(content):
                if j < len(tokens):
                    if content[i:i + len(tokens[j])] == tokens[j]:
                        html += '<div class="form-check form-check-inline mr-0">'
                        html += '<input class="form-check-input check-with-label d-none" type="checkbox" ' \
                                'id="token_{}" name="tokens[]" value="{}"{}>' \
                            .format(j, j, " checked" if self.tokens_contain(j) else "")
                        html += '<label class="form-check-label label-for-check" for="token_{}">{}</label>' \
                            .format(j, tokens[j])
                        html += '</div>'
                        i += len(tokens[j])
                        j += 1
                    else:
                        html += content[i]
                        i += 1
                else:
                    html += content[i]
                    i += 1

            return html
