from __future__ import absolute_import, division, print_function, unicode_literals

from os import path

from tensorflow.keras.layers import Activation, Dense, Conv1D, Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

IMAGE_PATH = './images/fully_conv_batchnorm.png'


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)
    x = Conv1D(128, 5, padding='same')(embedded_sequence)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    topic_pred = Dense(nb_labels, activation='softmax', name='topic')(x)

    model = Model(inputs=sequence_input, outputs=topic_pred, name='fully_conv_batchnorm')

    if not path.exists(IMAGE_PATH):
        plot_model(model, to_file=IMAGE_PATH, show_shapes=True)

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model
