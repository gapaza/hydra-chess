from keras import layers
import config
import tensorflow as tf


class PositionEvaluation(layers.Layer):

    def __init__(self):
        super(PositionEvaluation, self).__init__()
        self.global_average_pooling = layers.GlobalAveragePooling1D()
        self.dense_1 = layers.Dense(128, activation='linear')
        self.eval_layer = layers.Dense(1, name='position_evaluation_head')
        self.activation = layers.Activation('softsign', dtype='float32')  # 'tanh' 'softsign'

    def __call__(self, inputs):
        move_predictions, board_predictions = inputs

        # Perform average pooling across the token axis
        # (batch, 128, 1963) --> (batch, 1963)
        # (batch, 64, 28) --> (batch, 28)
        # move_aggregated = tf.reduce_mean(move_predictions, axis=1)
        # board_aggregated = tf.reduce_mean(board_predictions, axis=1)

        move_aggregated = self.global_average_pooling(move_predictions)
        board_aggregated = self.global_average_pooling(board_predictions)


        # Concatenate the aggregated vectors
        combined = tf.concat([move_aggregated, board_aggregated], axis=1)

        # Pass through the dense layer
        combined = self.dense_1(combined)
        combined = tf.keras.activations.mish(combined)

        # Pass through the dense layer
        position_evaluation = self.eval_layer(combined)
        position_evaluation = self.activation(position_evaluation)
        # Cast posiiton_evaluation to float32
        # position_evaluation = tf.cast(position_evaluation, dtype='float32')
        # position_evaluation = tf.keras.activations.softsign(position_evaluation)

        return position_evaluation

