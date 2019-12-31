from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv1D, Input, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequence)
    x = BatchNormalization()(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = BatchNormalization()(x)
    f = Conv1D(128, 5, activation='relu')(x)
    x = GlobalAveragePooling1D()(f)
    softmax_layer = Dense(nb_labels, activation='softmax', name='topic')
    topic_pred = softmax_layer(x)
    cam = tf.einsum('btf,fl->blt', f, softmax_layer.kernel)

    model = Model(inputs=sequence_input, outputs=[topic_pred, cam], name='keras_example_model')

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss=['categorical_crossentropy', rationale_loss], optimizer='adam', metrics=['acc'])

    return model


def rationale_loss(r_true, r_pred):
    return 0
