from keras import layers
import config
import tensorflow as tf


class BoardModeling(layers.Layer):

    def __init__(self):
        super(BoardModeling, self).__init__()
        self.linear = layers.Dense(config.board_modality_classes)
        self.activation = layers.Activation('softmax', dtype='float32')

    def __call__(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs)
        return outputs

