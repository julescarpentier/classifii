#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

from models.data_parsing import *
from models.embedding_model import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

VALIDATION_SPLIT = 0.3
TRAINING_SPLIT = 1 - VALIDATION_SPLIT
EMBEDDING_DIM = 100
MAX_LENGTH_SEQUENCE = 2000
MAX_NB_WORDS = 80000

# DATAFRAME
df_train = get_dataframe()
print("SHUFFLING")
df_train = df_train.sample(frac=1)
print(df_train)
# print_stats(df_train)


# TEXTS TO TOKENIZE INTO INTEGERS (or to do in data_parsing)
texts = df_train['text'].to_numpy()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# fit_on_texts(texts) creates the vocabulary index based on word frequency (word_index dictionary)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print('Number of unique tokens: %s' % len(word_index))
# Transforms each text in texts to a sequence of integers, based on the word_index dictionary
sequences = tokenizer.texts_to_sequences(texts)
# padding (pre-padding i.e 0 before)
data = pad_sequences(sequences, maxlen=MAX_LENGTH_SEQUENCE)

# LABELS
targets = df_train['target'].to_numpy()
# Transform labels ([1 2 .. 19 20]) into binary vectors ([[1 0 .. 0 0] [0 1 0 .. 0] .. [0 0 .. 0 1]]). Necessary to do.
labels = to_categorical(np.asarray(targets), num_classes=20)

print("=======DATA AND LABELS========")
print(data[:10])
print(data.shape)
print(labels[:10])
print(labels.shape)
print("===============================")

# TRAIN AND TEST DATA (80%-20% ou 70%-30%)
nb_train_sample = int(TRAINING_SPLIT * data.shape[0])
x_train = data[:nb_train_sample]
y_train = labels[:nb_train_sample]
x_test = data[nb_train_sample:]
y_test = labels[nb_train_sample:]

# PRE-TRAINED WORD VECTORS (GloVe)
GLOVE_DIR = 'data/glove'
embeddings_index = {}
# Converting glove.txt into the appropriate dictionary { 'word_1' : word_1-vector, ... }
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
f.close()

print("==LENGTH EMBEDDING INDEX==")
print(len(embeddings_index))
print("=======================")

# MATRICE DE NOS PROPRES MOTS, OBTENUS A PARTIR DE NOTRE DICTIONNAIRE DE MOTS PRE-ENTRAINES
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be randomized.
        embedding_matrix[i] = embedding_vector

# MATRICE DE VECTEURS ALEATOIRES (ON VEUT VOIR SI L'EMBEDDED TEXT EST VRAIMENT UTILE)
# random_embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

# MODEL WITHOUT EMBEDDED TEXT
# print("MODEL WITH RANDOM VECTORIZED WORDS IN THE EMBEDDING LAYER")
# random_embedding_model = create_embedding_model(len(word_index), EMBEDDING_DIM, random_embedding_matrix,
#                                                 MAX_LENGTH_SEQUENCE)
# print(random_embedding_model.summary())
#
# # TRAINING THE RANDOM MODEL
# print("TRAINING MODEL WITHOUT EMBEDDED TEXT")
# history_1 = random_embedding_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)

# EMBEDDING MODEL
print("EMBEDDING MODEL")
embedding_model = create_embedding_model(len(word_index), EMBEDDING_DIM, embedding_matrix, MAX_LENGTH_SEQUENCE)
print(embedding_model.summary())

# TRAINING THE EMBEDDING MODEL
print("TRAINING EMBEDDING MODEL")
history_2 = embedding_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8, batch_size=32)


# summarize history for accuracy
def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def compare_history(history_1, history_2):
    plt.plot(history_1.history['accuracy'])
    plt.plot(history_1.history['val_accuracy'])
    plt.plot(history_2.history['accuracy'])
    plt.plot(history_2.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_no_embedding', 'test_no_embedding', 'train_with_embedding', 'test_with_embedding'],
               loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history_1.history['loss'])
    plt.plot(history_1.history['val_loss'])
    plt.plot(history_2.history['loss'])
    plt.plot(history_2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_no_embedding', 'test_no_embedding', 'train_with_embedding', 'test_with_embedding'],
               loc='upper left')
    plt.show()


# plot_history(history_1)
plot_history(history_2)
# compare_history(history_1, history_2)

# RESULTATS ACTUELS (modèle très simple: embeddingLayer-conv1D-Dense-Dense. Les batchs sont mélangés à chaque nouvelle époque)
# 2 labels (atheism vs comp.graphics) : précision 95% au bout de 2 époques
# 4 labels fortement distincts (atheism, graphics, autos, baseball) : 94% de précision au bout de 3 époques, pas d'améliorations ensuite
# 3 labels dont 2 ressemblants (atheism, comp.graphics, comp.os.ms-windows.misc) : 85% de précision au bout de 3 époques (tester pour plus)
# 3 labels ressemblants (ms-windows, pc.hardware, mac.hardware) : De grosses difficultés. Progression lente. 65% de précision en 8 époques. Overfiiting ?
# 8 labels : 82% de précision au bout de 4 époques. Il y a Overfitting ensuite (100% de précision sur les données d'entraînement et pas d'amélioration en test ainsi qu'un perte qui augmente))

# Suite à voir...
# Sur tous les labels.
# Avec un modèle plus complexe. (embeddingLayer-(conv1D-maxPooling1D)*3-Dense-Dense)
# Un embedding model plus gros (dimension 300 au lieu de 100) ?
# Une aide à la décision.
