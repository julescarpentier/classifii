from __future__ import absolute_import, division, print_function, unicode_literals

from os import path

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Activation, Dense, Conv1D, Input, BatchNormalization, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

IMAGE_PATH = './images/fully_conv_rationales.png'


def cam_einsum(x, w):
    return tf.einsum('btf,fl->blt', x, w)


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
    f = Activation('relu')(x)

    x = GlobalAveragePooling1D()(f)
    softmax_layer = Dense(nb_labels, activation='softmax', name='topic')
    topic_pred = softmax_layer(x)

    cam = Lambda(cam_einsum, arguments={'w': softmax_layer.kernel}, name='cam')(f)

    model = Model(inputs=sequence_input, outputs=[topic_pred, cam], name='fully_conv_rationales')

    if not path.exists(IMAGE_PATH):
        plot_model(model, to_file=IMAGE_PATH, show_shapes=True)

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss=['categorical_crossentropy', rationale_loss], optimizer='adam', metrics=['acc'])

    return model


def rationale_loss(r_true, r_pred):
    return k.cast_to_floatx(0)
