import os

import numpy as np

BASE_DIR = '../data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')


def get_embeddings_index():
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    return embeddings_index
