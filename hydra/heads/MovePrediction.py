from keras import layers
import config
from tensorflow import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(package="Hydra", name="MovePrediction")
class MovePrediction(layers.Layer):

    def __init__(self):
        super(MovePrediction, self).__init__(name='move_prediction_head')
        self.next_move_avg = layers.GlobalAveragePooling1D(name='move_prediction_avg_pooling')
        self.move_prediction_dense = layers.Dense(1024, activation="mish", name='move_prediction_dense')
        self.next_move_prediction = layers.Dense(config_new.vocab_size, name='move_prediction_output')
        self.activation = layers.Activation('softmax', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        move_encoding, board_encoding = inputs
        next_move = tf.concat([move_encoding, board_encoding], axis=1)
        next_move = self.next_move_avg(next_move)
        next_move = self.move_prediction_dense(next_move)
        next_move = self.next_move_prediction(next_move)
        next_move = self.activation(next_move)
        return next_move

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
