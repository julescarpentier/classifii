import pandas as pd
import tensorflow as tf


def gen_train(df_train):
    for index, values in df_train.iterrows():
        yield (
            values.tokens,
            values.target
        )
