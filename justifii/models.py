import sys

import numpy as np
from sqlalchemy import Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.orm import relationship
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from justifii.database import Base


class User(Base):
    username = Column(String(80), unique=True, nullable=False)
    password = Column(String(255), nullable=False)

    rationales = relationship(lambda: Rationale, backref='user', lazy=True)

    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def __repr__(self):
        return '<User username={}>'.format(self.username)

    def __str__(self):
        return self.username


class Text(Base):
    fpath = Column(String(120), unique=True, nullable=False)

    label_id = Column(Integer, ForeignKey('labels.id'), nullable=False)
    rationales = relationship(lambda: Rationale, backref='text', lazy=True)

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

    def get_r(self, nb_labels, max_sequence_length):
        sum_r = sum(rationale.get_r(nb_labels, max_sequence_length) for rationale in self.rationales)
        return 1/len(self.rationales) * sum_r


class Label(Base):
    name = Column(String(80), unique=True, nullable=False)
    target = Column(Integer, unique=True, nullable=False)

    texts = relationship(lambda: Text, backref='label', lazy=True)

    def __init__(self, name=None, target=None):
        self.name = name
        self.target = target

    def __repr__(self):
        return '<Label id={}, name={}, target={}>'.format(self.id, self.name, self.target)

    def __str__(self):
        return self.name


class Rationale(Base):
    tokens = Column(PickleType, nullable=True)

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    text_id = Column(Integer, ForeignKey('texts.id'), nullable=False)

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

    def get_r(self, nb_labels, max_sequence_length):
        word_sequence = self.text.get_word_sequence()
        r = np.zeros((nb_labels, len(word_sequence)))
        for token in self.tokens:
            r[self.text.label.target][token] = 1

        return pad_sequences(r, maxlen=max_sequence_length)

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
