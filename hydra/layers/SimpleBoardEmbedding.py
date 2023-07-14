import tensorflow as tf
from keras import layers
import numpy as np
import config


class SimpleBoardEmbedding(layers.Layer):

    def __init__(self, name, positional=False):
        super(SimpleBoardEmbedding, self).__init__(name=name)
        self.flatten = layers.Flatten()

        # --> Token Embeddings
        self.token_embeddings = layers.Embedding(
            14, config.embed_dim, name="board_embedding", mask_zero=False,
        )

        # --> Masking Layer
        self.masking_layer = layers.Masking(mask_value=1000)

        self.positional = positional
        self.token_position_embeddings = layers.Embedding(
            input_dim=64,
            output_dim=config.embed_dim,
            weights=[self.get_pos_encoding_matrix(64, config.embed_dim)],
            name="position_embedding",
        )




    def __call__(self, inputs):
        board_embedding = self.flatten(inputs)
        board_embedding = self.token_embeddings(board_embedding)
        board_embedding = self.masking_layer(board_embedding)
        if not self.positional:
            return board_embedding

        board_position_embeddings = self.token_position_embeddings(tf.range(start=0, limit=64, delta=1))
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




