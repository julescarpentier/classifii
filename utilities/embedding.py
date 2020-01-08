import os

import numpy as np
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Embedding

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
EMBEDDING_DIM = 100


def get_embeddings_index():
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'), encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found {} word vectors.'.format(len(embeddings_index)))
    return embeddings_index


def get_embedding_matrix(num_words, max_num_words, word_index):
    embeddings_index = get_embeddings_index()

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_trainable_embedding_layer(num_words, max_sequence_length):
    return Embedding(num_words, EMBEDDING_DIM, input_length=max_sequence_length, trainable=True,
                     name='embedded_sequence')


def get_pre_trained_embedding_layer(num_words, embedding_matrix, max_sequence_length):
    return Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
                     input_length=max_sequence_length, trainable=False, name='embedded_sequence')


def get_pre_trained_trainable_embedding_layer(num_words, embedding_matrix, max_sequence_length):
    return Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix),
                     input_length=max_sequence_length, trainable=True, name='embedded_sequence')
