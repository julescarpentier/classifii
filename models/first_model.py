import pandas as pd
import tensorflow as tf
import keras

def gen_train(df_train):
    for index, values in df_train.iterrows():
        yield (
            values.tokens,
            values.target
        )

def get_model():
    # ---- Sequential call ----#

    # Inputs
    inputs = Input(some_shape, dtype=some_dtype)

    # Layers
    layer = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Dense(32, activation='sigmoid')(layer)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    return model


