from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from models import fully_conv_control, fully_conv_dropout, fully_conv_spacial_dropout, fully_conv_batchnorm
from utilities.dataset import get_texts_labels
from utilities.embedding import get_pre_trained_embedding_layer, get_embedding_matrix
from utilities.plotting import plot_compare_acc, plot_compare_loss

tf.keras.backend.clear_session()  # For easy reset of notebook state.

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2

# first, prepare text samples and their labels

print('Processing text dataset')

texts, labels, labels_index = get_texts_labels()

print('Found {} texts.'.format(len(texts)))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = get_embedding_matrix(num_words, MAX_NUM_WORDS, word_index)

# load pre-trained word embeddings into an Embedding layer
embedding_layer = get_pre_trained_embedding_layer(num_words, embedding_matrix, MAX_SEQUENCE_LENGTH)

print('Training models.')

control_model = fully_conv_control.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH, len(labels_index))
dropout_model = fully_conv_dropout.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH, len(labels_index))
spacial_dropout_model = fully_conv_spacial_dropout.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH,
                                                                      len(labels_index))
batchnorm_model = fully_conv_batchnorm.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH, len(labels_index))

control_model.summary()
dropout_model.summary()
spacial_dropout_model.summary()
batchnorm_model.summary()

control_history = control_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
dropout_history = dropout_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
spacial_dropout_history = spacial_dropout_model.fit(x_train, y_train, batch_size=128, epochs=10,
                                                    validation_data=(x_val, y_val))
batchnorm_history = batchnorm_model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))

# ensure the output folder exists
try:
    os.makedirs('output')
except OSError:
    pass

# Plot control, dropout and batchnorm accuracies
plot_compare_acc('./output/dropout_batchnorm_comparison_acc.png',
                 ['Control', 'Dropout', 'Spacial Dropout', 'BatchNorm'], control_history, dropout_history,
                 spacial_dropout_history, batchnorm_history)

# Plot control, dropout and batchnorm losses
plot_compare_loss('./output/dropout_batchnorm_comparison_loss.png',
                  ['Control', 'Dropout', 'Spacial Dropout', 'BatchNorm'], control_history, dropout_history,
                  spacial_dropout_history, batchnorm_history)
