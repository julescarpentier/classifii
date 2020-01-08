import os

from justifii.database import db_session, Base, engine


def init_db():
    # import all modules here that might define models so that
    # they will be registered properly on the metadata.  Otherwise
    # you will have to import them first before calling init_db()
    import justifii.models as orm_models
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    base_dir = 'data'
    text_data_dir = os.path.join(base_dir, '20news-18828')

    print('Processing text dataset.')

    labels_index = {}  # dictionary mapping label name to numeric id
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            target = len(labels_index)
            labels_index[name] = target
            label = orm_models.Label(name, target)
            db_session.add(label)
            db_session.commit()
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    text = orm_models.Text(fpath, label.id)
                    db_session.add(text)
    db_session.commit()

    print('Found {} texts.'.format(orm_models.Text.query.count()))
    for name, target in labels_index.items():
        print('{}: {}'.format(target, name))
