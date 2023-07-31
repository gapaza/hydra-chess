from keras import layers
import config
import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Hydra", name="MoveRanking")
class MoveRanking(layers.Layer):

    def __init__(self):
        super(MoveRanking, self).__init__(name='move_ranking_head')
        self.next_move_avg = layers.GlobalAveragePooling1D(name='move_ranking_avg_pooling')
        # self.next_move_dense_1 = layers.Dense(256, name='move_ranking_dense_1', activation='relu')
        # self.next_move_dense_2 = layers.Dense(512, name='move_ranking_dense_2', activation='relu')
        # self.next_move_dense_3 = layers.Dense(1024, name='move_ranking_dense_3')
        self.next_move_prediction = layers.Dense(config.vocab_size, name='move_ranking_output')
        self.activation = layers.Activation('softplus', dtype='float32')  # softplus | leakyrelu | relu (vanishing gradients problem)


    def call(self, inputs, training=False, mask=None):
        move_encoding, board_encoding = inputs
        next_move = tf.concat([move_encoding, board_encoding], axis=1)
        next_move = self.next_move_avg(next_move)
        # next_move = self.next_move_dense_1(next_move)
        # next_move = self.next_move_dense_2(next_move)
        # next_move = self.next_move_dense_3(next_move)
        # next_move = tf.nn.leaky_relu(next_move)
        # if training:
        #     next_move = tf.nn.dropout(next_move, rate=0.5)
        next_move = self.next_move_prediction(next_move)
        next_move = self.activation(next_move)
        return next_move

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
