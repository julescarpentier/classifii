from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import tensorflow as tf
from random import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from models import fully_conv_rationales
from utilities.dataset import get_texts_labels
from utilities.embedding import get_embedding_matrix, get_pre_trained_trainable_embedding_layer
from utilities.plotting import plot_acc, plot_loss

tf.keras.backend.clear_session()  # For easy reset of notebook state.

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2

# Prepare text samples and their labels

print('Processing text dataset')

texts, labels, labels_index = get_texts_labels()

print('Found {} texts.'.format(len(texts)))

# fake rationales
rationales = np.zeros((len(texts), len(labels_index), MAX_SEQUENCE_LENGTH))
for i in range(rationales.shape[0]):
    j = labels[i]
    for k in range(rationales.shape[2]):
        if random() < 0.03:
            rationales[i][j][k] = 1

# Vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found {} unique tokens.'.format(len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# rationales = rationales[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# split the data into a training set and a validation set
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
r_train = rationales[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
r_val = rationales[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = get_embedding_matrix(num_words, MAX_NUM_WORDS, word_index)

# Embedding layer
embedding_layer = get_pre_trained_trainable_embedding_layer(num_words, embedding_matrix, MAX_SEQUENCE_LENGTH)

print('Training model.')

model = fully_conv_rationales.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH, len(labels_index))

history = model.fit(x_train, (y_train, r_train), batch_size=32, epochs=10, validation_data=(x_val, (y_val, r_val)))

# ensure the output folder exists
try:
    os.makedirs('output')
except OSError:
    pass

# Plot accuracy
plot_acc('./output/fully_conv_rationales_acc.png', history)

# Plot trainable, pre-trained and both losses
plot_loss('./output/fully_conv_rationales_loss.png', history)
