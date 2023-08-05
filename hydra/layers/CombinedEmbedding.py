import tensorflow as tf
from keras import layers
import numpy as np
import config
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Hydra", name="CombinedEmbedding")
class CombinedEmbedding(layers.Layer):

    def __init__(self, **kwargs):
        super(CombinedEmbedding, self).__init__(name='combined_embedding', **kwargs)
        self.supports_masking = True

        # --> Parameters
        self.embed_dim = config.embed_dim
        self.seq_length = config.seq_length + config.board_seq_length
        self.vocab_size = config.vocab_size

        # --> Board Embedding
        self.board_embeddings = layers.Embedding(
            config.board_modality_classes, self.embed_dim, name="board_embedding", mask_zero=False,
        )

        # --> Move Embedding
        self.move_embeddings = layers.Embedding(
            self.vocab_size, self.embed_dim, name="move_embedding", mask_zero=False,
        )

        # --> Position Embeddings
        self.token_position_embeddings = layers.Embedding(
            input_dim=self.seq_length,
            output_dim=self.embed_dim,
            weights=[self.get_pos_encoding_matrix(self.seq_length, self.embed_dim)],
            name="positional_embedding",
        )

    def call(self, inputs, training=False, mask=None):
        move_tokens, board_tokens = inputs
        move_embeddings = self.move_embeddings(move_tokens)
        board_embeddings = self.board_embeddings(board_tokens)
        combined_embeddings = tf.concat([move_embeddings, board_embeddings], axis=1)
        combined_position_embeddings = self.token_position_embeddings(tf.range(start=0, limit=self.seq_length, delta=1))
        combined = combined_embeddings + combined_position_embeddings
        return combined

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
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


