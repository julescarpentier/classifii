from __future__ import absolute_import, division, print_function, unicode_literals

from os import path

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Input, BatchNormalization, GlobalAveragePooling1D, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

IMAGE_PATH = 'output/fully_conv_with_rationales.png'


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequence)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv1D(128, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    f = Conv1D(128, 5, activation='relu', padding='same')(x)

    x = GlobalAveragePooling1D()(f)
    softmax_layer = Dense(nb_labels, activation='softmax', name='topic')
    topic_pred = softmax_layer(x)

    w = softmax_layer.kernel
    # batch x tokens x feats / feats x labels -> batch x labels x tokens
    cam = Lambda(lambda t: tf.einsum('btf,fl->blt', t, w), name='cam')(f)

    model = Model(inputs=sequence_input, outputs=[topic_pred, cam], name='fully_conv_with_rationales')

    if not path.exists(IMAGE_PATH):
        plot_model(model, to_file=IMAGE_PATH, show_shapes=True)

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss=['categorical_crossentropy', rationale_loss], optimizer='adam', metrics=['acc'])

    return model


def rationale_loss(r_true, r_pred):
    loss = tf.reduce_sum(r_true * (r_true - tf.sigmoid(r_pred)))

    return loss / tf.reduce_sum(r_true)
