from keras import layers
import config
import tensorflow as tf


class BoardPrediction(layers.Layer):

    def __init__(self):
        super(BoardPrediction, self).__init__()
        # self.linear = layers.Dense(14, activation="softmax")
        self.linear = layers.Dense(14)
        self.activation = layers.Activation('softmax', dtype='float32')
        # Total Classes: 14
        # 1 for empty
        # 1-7 for white
        # 8-13 for black
        # 14 for mask

    def __call__(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs)
        return outputs

