import tensorflow as tf
from keras import layers
import numpy as np
import config



class D2_PositionalEmbedding(layers.Layer):

    def __init__(self):
        super(D2_PositionalEmbedding, self).__init__()

        # 64 Board Squares
        self.total_length = 64

        # Embedding Layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.total_length,
            output_dim=config.embed_dim,
            weights=[self.get_2d_pos_encoding()],
            name="2d_position_embedding",
        )

    def __call__(self, inputs):
        positional_inputs = tf.range(start=0, limit=self.total_length, delta=1)
        positional_embeddings = self.embedding_layer(positional_inputs)
        positional_inputs = inputs + positional_embeddings
        return positional_inputs


    def get_2d_pos_encoding(self):
        # Compute the 2D positions for an 8x8 chessboard flattened into a 1D sequence
        # Issue with this is that squares (2, 4) and (4, 2) would have the same positional encoding
        positions = [(i // 8, i % 8) for i in range(self.total_length)]
        pos_enc = np.array(
            [
                [pos_i / np.power(10000, 2 * (j // 2) / config.embed_dim) + pos_j / np.power(10000, 2 * (j // 2) / config.embed_dim) for j in range(config.embed_dim)]
                for pos_i, pos_j in positions
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc


