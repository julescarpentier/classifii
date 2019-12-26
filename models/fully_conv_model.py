import tensorflow as tf
from tensorflow_core.python.keras.layers.embeddings import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, GlobalAveragePooling1D
from keras.models import Model

import numpy as np


def create_fully_cnn_model(len_word_index, embedding_dim, embedding_matrix, input_length):
    embedding_layer = Embedding(len_word_index + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=input_length, mask_zero=True, trainable=True)
    # model = tf.keras.Sequential()
    # model.add(embedding_layer)
    # model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(20, activation='softmax'))

    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # drop_out ici aussi ?
    l_conv1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
    l_drop1 = Dropout(0.5)(l_conv1)
    l_conv2 = Conv1D(128, 5, activation='relu')(l_drop1)
    l_drop2 = Dropout(0.5)(l_conv2)
    l_conv3 = Conv1D(128, 5, activation='relu')(l_drop2)
    # l_drop3 = Dropout(0.5)(l_conv3)  # global max pooling ?
    l_gap = GlobalAveragePooling1D()(l_conv3)
    last_layer = Dense(20, activation='softmax')
    last_layer_output = last_layer(l_gap)

    # CAM
    W = last_layer.kernel
    cam = tf.einsum('btf,fl->blt', l_conv3, W)  # batch x tokens x feats / feats x labels -> batch x labels x tokens

    # Matrice de justificatifs test (débuggage), on s'est arrêté là avant les vacs
    zeros_matrix = np.zeros.zeros.zeros((32, 20, input_length))

    model = Model(inputs=sequence_input, outputs=[last_layer_output, cam])

    def loss2(y_true, y_pred): # fonction de perte à définir
        return 0

    model.compile(optimizer='adam',
                  loss=['categorical_crossentropy', loss2(zeros_matrix, cam)],
                  metrics=['accuracy'])

    return model
