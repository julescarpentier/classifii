from __future__ import absolute_import, division, print_function, unicode_literals

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

from justifii.database import db_session
from justifii.models import Text, Label
from models import fully_conv_with_rationales, fully_conv_without_rationales
from utilities.embedding import get_embedding_matrix, get_pre_trained_trainable_embedding_layer

tf.keras.backend.clear_session()  # For easy reset of notebook state.

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2

# ensure the output folder exists
try:
    os.makedirs('output')
except OSError:
    pass

# Prepare text samples and their labels

print('Retrieving texts from database')

texts = []
labels = []
nb_labels = Label.query.count()
rationales = []
for text in Text.query.filter(Text.rationales.any()):
    texts.append(text.get_content())
    labels.append(text.label.target)
    rationales.append(text.get_r(nb_labels, MAX_SEQUENCE_LENGTH))
rationales = np.asarray(rationales)
db_session.remove()

print('Retrieved {} texts.'.format(len(texts)))

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
rationales = rationales[indices]
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

print('Training models.')

model_with_rationales = fully_conv_with_rationales.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH, nb_labels)
model_without_rationales = fully_conv_without_rationales.get_compiled_model(embedding_layer, MAX_SEQUENCE_LENGTH,
                                                                            nb_labels)
history_with_rationales = model_with_rationales.fit(x_train, (y_train, r_train), batch_size=16, epochs=10,
                                                    validation_data=(x_val, (y_val, r_val)))
history_without_rationales = model_without_rationales.fit(x_train, y_train, batch_size=16, epochs=10,
                                                          validation_data=(x_val, y_val))

# model_with_rationales.save('output/fully_conv_with_rationales.h5')
# model_without_rationales.save('output/fully_conv_without_rationales.h5')
# print('Saved models')

# Plot accuracy
plt.figure()
plt.plot(history_with_rationales.history['val_topic_acc'])
plt.plot(history_without_rationales.history['val_acc'])
plt.plot(history_with_rationales.history['topic_acc'], '--')
plt.plot(history_without_rationales.history['acc'], '--')
plt.title('Compared accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['R', 'No R', 'R', 'No R'], loc='upper left')
plt.savefig('output/fully_conv_rationales_acc.png')
plt.close()

# Plot trainable, pre-trained and both losses
plt.figure()
plt.plot(history_with_rationales.history['val_topic_loss'])
plt.plot(history_without_rationales.history['val_loss'])
plt.plot(history_with_rationales.history['topic_loss'], '--')
plt.plot(history_without_rationales.history['loss'], '--')
plt.title('Compared losses')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['R', 'No R', 'R', 'No R'], loc='upper left')
plt.savefig('output/fully_conv_rationales_loss.png')
plt.close()

# Plot everything
plt.figure()
for metric, values in history_with_rationales.history.items():
    plt.plot(values, '-' if 'val' in metric else '--', label=metric)
plt.title('Everything to show with rationales')
plt.ylabel('Metrics')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.savefig('output/fully_conv_rationales_everything.png')
plt.close()
