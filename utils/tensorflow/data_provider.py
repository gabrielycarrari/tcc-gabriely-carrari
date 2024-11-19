import tensorflow as tf

from ..data_provider import DataProvider as dataProvider

class DataProvider(dataProvider, tf.keras.utils.Sequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
