from __future__ import absolute_import, division, print_function, unicode_literals

from os import path

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Activation, Dense, Conv1D, Input, BatchNormalization, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

IMAGE_PATH = './images/fully_conv_rationales.png'


def cam_einsum(x, w):
    return tf.einsum('btf,fl->blt', x, w)  # batch x tokens x feats / feats x labels -> batch x labels x tokens


def get_model(embedding_layer, max_sequence_length, nb_labels):
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32', name='sequence')
    embedded_sequence = embedding_layer(sequence_input)

    l_conv1 = Conv1D(128, 5, activation='relu', padding='same')(embedded_sequence)
    l_batch1 = BatchNormalization()(l_conv1)
    l_conv2 = Conv1D(128, 5, activation='relu', padding='same')(l_batch1)
    l_batch2 = BatchNormalization()(l_conv2)
    last_conv = Conv1D(128, 5, activation='relu', padding='same')(l_batch2)
    # last_batch = BatchNormalization()(last_conv) ?
    l_gap = GlobalAveragePooling1D()(last_conv)
    softmax_layer = Dense(nb_labels, activation='softmax', name='topic')
    last_layer_output = softmax_layer(l_gap)

    # CAM
    w = softmax_layer.kernel
    cam = cam_einsum(last_conv, w)

    # Matrice de justificatifs test (débuggage), on s'est arrêté là avant les vacs
    # zeros_matrix = np.zeros.zeros.zeros((32, 20, input_length))

    model = Model(inputs=sequence_input, outputs=[last_layer_output, cam], name='fully_conv_rationales')

    # if not path.exists(IMAGE_PATH):
    #     plot_model(model, to_file=IMAGE_PATH, show_shapes=True)

    return model


def get_compiled_model(embedding_layer, max_sequence_length, nb_labels):
    model = get_model(embedding_layer, max_sequence_length, nb_labels)

    model.compile(loss=['categorical_crossentropy', rationale_loss], optimizer='adam', metrics=['acc'])

    return model


def rationale_loss(r_true, r_pred):
    return k.cast_to_floatx(0)
