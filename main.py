from models.data_parsing import *
from models.first_model import *


print(get_dataframe())

ds_train = tf.data.Dataset.from_generator(
    gen_train,
    output_types=(tf.string, tf.int32)
).padded_batch(
    32,
    padded_shapes=([None], [])
)



# print_stats()

#hist_length()

#pop_classes()

