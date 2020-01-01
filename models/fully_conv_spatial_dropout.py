from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras.layers import Dense, Conv1D, Input, SpatialDropout1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequence)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    topic_pred = Dense(nb_labels, activation='softmax', name='topic')(x)

    model = Model(inputs=sequence_input, outputs=topic_pred, name='fully_conv_spatial_dropout')

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model
