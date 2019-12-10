from models.data_parsing import *
from models.first_model import *

df_train = get_dataframe()

print(df_train)


def gen_train():
    for index, values in df_train.iterrows():
        yield (
            values.tokens,
            values.target
        )


ds_train = tf.data.Dataset.from_generator(
    gen_train,
    output_types=(tf.string, tf.int32)
).padded_batch(
    32,
    padded_shapes=([None], [])
)

for value in ds_train.take(2):
    print(value)

print("MODELE")

model = get_model()

print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("ENTRAINEMENT")
# ENCORE DES ERREURS ICI
history = model.fit(ds_train.shuffle(10000).batch(512),
                    epochs=5,
                    # validation_data=validation_data.batch(512),
                    verbose=1)


# print_stats()

# hist_length()

# pop_classes()
