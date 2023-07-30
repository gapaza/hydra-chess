from keras import layers
import config
import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="Hydra", name="GameModeling")
class GameModeling(layers.Layer):

    def __init__(self):
        super(GameModeling, self).__init__(name='game_modeling_head')

        # 1. Move Modeling
        self.move_modeling_head = layers.Dense(config.vocab_size, name="move_modeling_output")
        self.move_modeling_activation = layers.Activation('softmax', dtype='float32')

        # 2. Board Modeling
        self.board_modeling_head = layers.Dense(config.board_modality_classes, name='board_modeling_output')
        self.board_modeling_activation = layers.Activation('softmax', dtype='float32')

        # 3. Evaluation Modeling
        self.move_modeling_pooling = layers.GlobalAveragePooling1D(name='move_modeling_pooling')
        self.board_modeling_pooling = layers.GlobalAveragePooling1D(name='board_modeling_pooling')
        self.eval_modeling_dense = layers.Dense(128, activation="mish", name='eval_modeling_dense')
        self.eval_modeling_head = layers.Dense(1, name='eval_modeling_output')
        self.eval_modeling_activation = layers.Activation('softsign', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        move_encoding, board_encoding = inputs

        # 1. Move Modeling
        move_predictions = self.move_modeling_head(move_encoding)
        move_predictions = self.move_modeling_activation(move_predictions)

        # 2. Board Modeling
        board_predictions = self.board_modeling_head(board_encoding)
        board_predictions = self.board_modeling_activation(board_predictions)

        # 3. Evaluation Modeling
        move_aggregate = self.move_modeling_pooling(move_predictions)
        board_aggregate = self.board_modeling_pooling(board_predictions)
        combined_aggregate = tf.concat([move_aggregate, board_aggregate], axis=1)
        eval_intermediate = self.eval_modeling_dense(combined_aggregate)
        eval_predictions = self.eval_modeling_head(eval_intermediate)
        eval_predictions = self.eval_modeling_activation(eval_predictions)

        return move_predictions, board_predictions, eval_predictions


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)













