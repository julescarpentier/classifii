import tensorflow as tf
from tensorflow_core.python.keras.layers.embeddings import Embedding


def create_embedding_model(len_word_index, embedding_dim, embedding_matrix, input_length, nb_layer):
    embedding_layer = Embedding(len_word_index + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=input_length, mask_zero=True, trainable=True)
    model = tf.keras.Sequential()
    model.add(embedding_layer)

    for i in range(nb_layer):
        model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(7))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
