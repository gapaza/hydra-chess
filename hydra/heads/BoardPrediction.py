from keras import layers
import config
import tensorflow as tf


class BoardPrediction(layers.Layer):

    def __init__(self):
        super(BoardPrediction, self).__init__()
        self.linear = layers.Dense(12, use_bias=True)

    def __call__(self, inputs):
        outputs = tf.reshape(inputs, (-1, 8, 8, 256))
        outputs = self.linear(outputs)
        return outputs

