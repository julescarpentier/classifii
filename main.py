from models.data_parsing import *
from models.first_model import *

df_train = get_dataframe()
# df_train.values.reshape()
# pd.melt(df_train)

print(df_train.shape)


# print(df_train)

# df_train = df_train.pivot()

df_train.pop('text')
df_train.pop('label')
df_train.pop('length')

# print(df_train.shape)
# print(df_train)

nrows = df_train.shape[0]
ncols = df_train.shape[1]

df_train = df_train.values.reshape(nrows, ncols, 1)
# df_train = pd.DataFrame(df_train) MARCHE PAS CAR DFTRAIN N'EST PLUS EN DIMENSION 2 ET UN DATAFRAME EST CENSE ETRE DE DIM 2



print(df_train)
# print(df_train.dtypes)
# print_stats(df_train)

# target = df_train.pop('target')
# df_train = df_train.pop('tokens')
# df_train = tf.convert_to_tensor(df_train, dtype=tf.float32)
# dataset = tf.data.Dataset.from_tensor_slices((df_train.values, target.values))

# dataset = tf.data.Dataset.from_tensor_slices((tf.cast(df_train.to_numpy(), tf.string),
                                             # tf.cast(target.to_numpy(), tf.int64)))

# for feat, target in dataset.take(5):
#     print('Features: {}, Target: {}'.format(feat, target))


def gen_train():
    for index, values in df_train.iterrows():
        yield (
            values.tokens,
            values.target
        )


# dataset : découpé en batch de taille 32

ds_train = tf.data.Dataset.from_generator(
    gen_train,
    output_types=(tf.string, tf.int64),
    output_shapes=((None,), ())).padded_batch(
    32,
    padded_shapes=([None], [])
)


for value in ds_train.take(5):
    print(value)

print("MODELE")

model = get_model()
print(model.summary())

print("ENTRAINEMENT")
# ENCORE DES ERREURS ICI
history = model.fit(ds_train,
                    epochs=5)

# print_stats()

# hist_length()

# pop_classes()
