from keras import layers
import config
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Hydra", name="MoveRanking")
class MoveRanking(layers.Layer):

    def __init__(self):
        super(MoveRanking, self).__init__(name='move_ranking_head')
        self.modality_fusion = layers.Concatenate(axis=1, name='move_ranking_modality_fusion')
        self.next_move_avg = layers.GlobalAveragePooling1D(name='move_ranking_avg_pooling')
        self.next_move_prediction = layers.Dense(config_new.vocab_size, name='move_ranking_output')
        self.activation = layers.Activation('relu', dtype='float32')  # softplus | leakyrelu | relu (vanishing gradients problem)


    def call(self, inputs, training=False, mask=None):
        move_encoding, board_encoding = inputs
        next_move = self.modality_fusion([move_encoding, board_encoding])
        next_move = self.next_move_avg(next_move)
        next_move = self.next_move_prediction(next_move)
        next_move = self.activation(next_move)
        return next_move

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
