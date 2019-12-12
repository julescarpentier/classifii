from models.data_parsing import *
from models.first_model import *
from FastText1 import *

# print(type(model))

df_train = get_dataframe()
print(df_train)
print(df_train.shape)

# df_train.pop('text')
# df_train.pop('label')
# df_train.pop('length')
# nrows = df_train.shape[0]
# ncols = df_train.shape[1]
# df_train = df_train.values.reshape(nrows, ncols, 1)
# df_train = pd.DataFrame(df_train) MARCHE PAS CAR DFTRAIN N'EST PLUS EN DIMENSION 2 ET UN DATAFRAME EST CENSE ETRE DE DIM 2

# Générateur
def gen_train():
    for index, values in df_train.iterrows():
        yield (
            values.tokens,
            values.target
        )


# Dataset : découpé en batch de taille 32
ds_train = tf.data.Dataset.from_generator(
    gen_train,
    output_types=(tf.string, tf.int64),
    output_shapes=((None,), ())).padded_batch(
    32,
    padded_shapes=([None], [])
)


for value in ds_train.take(3):
    print(value)

print("MODELE")

model = get_model()
print(model.summary())

# print("ENTRAINEMENT")
# # ENCORE DES ERREURS ICI
# history = model.fit(ds_train,
#                     epochs=5)

# print_stats()

# hist_length()

# pop_classes()
