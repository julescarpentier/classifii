# from keras.preprocessing.text import text_to_word_sequence
import nltk.tokenize as tkn

from justifii.database import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    proofs = db.relationship('Proof', backref='user', lazy=True)

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

    proofs = db.relationship('Proof', backref='text', lazy=True)

    def __init__(self, file_path=None, label=None):
        self.file_path = file_path
        self.label = label

    def __repr__(self):
        return '<Text id={}, file_path={}, label={}>'.format(self.id, self.file_path, self.label)

    def __str__(self):
        return '<Text id={}, file_path={}, label={}>'.format(self.id, self.file_path, self.label)

    def get_tokens(self):
        with open(self.file_path) as f:
            return tkn.WordPunctTokenizer().tokenize(f.read())


class Proof(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tokens = db.Column(db.PickleType, nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text_id = db.Column(db.Integer, db.ForeignKey('text.id'), nullable=False)

    def __init__(self, tokens=None):
        self.tokens = tokens

    def __repr__(self):
        return '<Proof id={}, user={}>'.format(self.id, self.user)

    def __str__(self):
        return '<Proof id={}, user={}>'.format(self.id, self.user)

    def tokens_contain(self, i):
        return str(i) in self.tokens
