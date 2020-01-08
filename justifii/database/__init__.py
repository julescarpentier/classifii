import os

from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.orm import scoped_session, sessionmaker

engine = create_engine('sqlite:///' + os.path.join('justifii', 'database', 'justifii.db'), convert_unicode=True)

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


@as_declarative()
class Base(object):
    @declared_attr
    def __tablename__(self):
        return self.__name__.lower() + 's'

    id = Column(Integer, primary_key=True)
    query = db_session.query_property()
