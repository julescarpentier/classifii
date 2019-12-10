from webjustif.database import db


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def __init__(self, username=None, password=None):
        self.username = username
        self.password = password

    def __repr__(self):
        return '<User %r>' % self.username


class Text(db.Model):
    __tablename__ = 'texts'
    id = db.Column(db.Integer, primary_key=True)
    file_path = db.Column(db.String(127), unique=True, nullable=False)

    def __init__(self, file_path=None):
        self.file_path = file_path

    def __repr__(self):
        return '<Text %r>' % self.file_path
