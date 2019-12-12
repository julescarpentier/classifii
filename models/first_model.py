import pandas as pd
from FastText1 import *
import tensorflow as tf
import keras
from keras import Input


class VocabLookup(keras.layers.Layer):
    """This layers simply wraps a lookup table."""

    def __init__(self, vocab, oov_buckets, **kwargs):
        super(VocabLookup, self).__init__(**kwargs)
        table_initializer = tf.lookup.KeyValueTensorInitializer(
            keys=vocab,
            values=range(len(vocab)),
            value_dtype=tf.int64
        )
        self.table = tf.lookup.StaticVocabularyTable(
            initializer=table_initializer,
            num_oov_buckets=OOV_BUCKETS,
        )

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, '')

    def call(self, inputs, mask=None):
        return self.table.lookup(inputs)


class CustomEmbedding(keras.layers.Layer):
    """This custom Embedding layer zeroes out the vector representation of padding tokens."""

    def __init__(self, embeddings_array, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.embeddings_array = embeddings_array

    def build(self, input_shape):
        self.embeddings = tf.Variable(self.embeddings_array, trainable=True)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        x = tf.nn.embedding_lookup(self.embeddings, inputs)
        x = tf.ragged.boolean_mask(x, mask)
        return x.to_tensor()

def get_model():

    model = tf.keras.Sequential()
    # A mettre : embeddedText layer

    model.add(VocabLookup(vocab=mots, oov_buckets=1000)) # prend les indices
    model.add(CustomEmbedding(embeddings_array=vect))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', input_shape=(None, 1)))
    model.add(tf.keras.layers.Dense(32, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
