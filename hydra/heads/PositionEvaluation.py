from keras import layers
import config
import tensorflow as tf


class PositionEvaluation(layers.Layer):

    def __init__(self):
        super(PositionEvaluation, self).__init__()
        self.dense_1 = layers.Dense(128, activation='relu')
        self.eval_layer = layers.Dense(1, name='position_evaluation_head')
        self.activation = layers.Activation('tanh', dtype='float32')


    def __call__(self, inputs):
        move_predictions, board_predictions = inputs

        # Perform average pooling across the token axis
        # (batch, 128, 1963) --> (batch, 1963)
        move_aggregated = tf.reduce_mean(move_predictions, axis=1)
        board_aggregated = tf.reduce_mean(board_predictions, axis=1)

        # Concatenate the aggregated vectors
        combined = tf.concat([move_aggregated, board_aggregated], axis=1)

        # Pass through the dense layer
        # combined = self.dense_1(combined)

        # Pass through the dense layer
        position_evaluation = self.eval_layer(combined)
        position_evaluation = self.activation(position_evaluation)

        return position_evaluation