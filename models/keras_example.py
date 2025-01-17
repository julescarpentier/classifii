from __future__ import absolute_import, division, print_function, unicode_literals

from os import path

from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

IMAGE_PATH = 'output/keras_example.png'


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequence)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    topic_pred = Dense(nb_labels, activation='softmax', name='topic')(x)

    model = Model(inputs=sequence_input, outputs=topic_pred, name='keras_example_model')

    if not path.exists(IMAGE_PATH):
        plot_model(model, to_file=IMAGE_PATH, show_shapes=True)

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model
