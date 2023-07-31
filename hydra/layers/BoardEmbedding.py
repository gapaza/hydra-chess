import tensorflow as tf
from keras import layers
import numpy as np
import config
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Hydra", name="BoardEmbedding")
class BoardEmbedding(layers.Layer):

    def __init__(self, positional, **kwargs):
        super(BoardEmbedding, self).__init__(name='board_embedding_block', **kwargs)
        self.supports_masking = True
        self.positional = positional

        # --> Parameters
        self.embed_dim = config.embed_dim
        self.board_seq_length = config.board_seq_length
        self.board_modality_classes = config.board_modality_classes

        # --> Token Embeddings
        self.token_embeddings = layers.Embedding(
            self.board_modality_classes,  self.embed_dim, name="board_embedding", mask_zero=False,
        )

        # --> Position Embeddings
        self.token_position_embeddings = layers.Embedding(
            input_dim=self.board_seq_length,
            output_dim= self.embed_dim,
            weights=[self.get_pos_encoding_matrix(self.board_seq_length,  self.embed_dim)],
            name="board_positional_embedding",
        )


    def call(self, inputs, training=False, mask=None):
        board_embedding = self.token_embeddings(inputs)
        if not self.positional:
            return board_embedding
        board_position_embeddings = self.token_position_embeddings(tf.range(start=0, limit=self.board_seq_length, delta=1))
        board_embedding = board_embedding + board_position_embeddings
        return board_embedding

    def get_pos_encoding_matrix(self, max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def get_config(self):
        base_config = super().get_config()
        config = {
            'positional': self.positional,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        positional = config.pop("positional")
        return cls(positional, **config)
