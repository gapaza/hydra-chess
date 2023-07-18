import tensorflow as tf
from keras import layers
import numpy as np
import config



class PositionalEmbedding(layers.Layer):

    def __init__(self):
        super(PositionalEmbedding, self).__init__()

        # Total length for both modalities
        self.total_length = 64 + config.seq_length

        # Embedding Layer
        self.embedding_layer = layers.Embedding(
            input_dim=self.total_length,
            output_dim=config.embed_dim,
            weights=[self.get_pos_encoding_matrix()],
            name="position_embedding",
        )

    def __call__(self, inputs):
        positional_embeddings = self.embedding_layer(tf.range(start=0, limit=self.total_length, delta=1))
        positional_inputs = inputs + positional_embeddings
        return positional_inputs


    def get_pos_encoding_matrix(self):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / config.embed_dim) for j in range(config.embed_dim)]
                if pos != 0
                else np.zeros(config.embed_dim)
                for pos in range(self.total_length)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc
