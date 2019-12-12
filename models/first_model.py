import pandas as pd
import tensorflow as tf
import keras
from keras import Input


def get_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', input_shape=(None, 1)))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
