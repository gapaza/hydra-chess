from keras import layers
import config
import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="Hydra", name="PositionModeling")
class PositionModeling(layers.Layer):

    def __init__(self):
        super(PositionModeling, self).__init__(name='position_modeling_head')

        # 1. Move Modeling
        self.move_modeling_head = layers.Dense(config.vocab_size, name="move_modeling_output")
        self.move_modeling_activation = layers.Activation('softmax', dtype='float32')

        # 2. Board Modeling
        self.board_modeling_head = layers.Dense(config.board_modality_classes, name='board_modeling_output')
        self.board_modeling_activation = layers.Activation('softmax', dtype='float32')


    def call(self, inputs, training=False, mask=None):
        move_encoding, board_encoding = inputs

        # 1. Move Modeling
        move_predictions = self.move_modeling_head(move_encoding)
        move_predictions = self.move_modeling_activation(move_predictions)

        # 2. Board Modeling
        board_predictions = self.board_modeling_head(board_encoding)
        board_predictions = self.board_modeling_activation(board_predictions)

        return move_predictions, board_predictions


    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)













