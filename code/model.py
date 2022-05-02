import numpy as np
import tensorflow as tf

class PoetryModel(tf.keras.model):
    def __init__(self):
        super(PoetryModel, self).__init__()

        self.encoder = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
