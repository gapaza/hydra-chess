import tensorflow as tf
from keras import layers
import numpy as np
import config


class SimpleBoardEmbedding(layers.Layer):

    def __init__(self, name):
        super(SimpleBoardEmbedding, self).__init__(name=name)
        self.flatten = layers.Flatten()

        # --> Token Embeddings
        self.token_embeddings = layers.Embedding(
            14, config.embed_dim, name="board_embedding", mask_zero=False
        )

        # --> Masking Layer
        self.masking_layer = layers.Masking(mask_value=1e9)




    def __call__(self, inputs):
        board_embedding = self.flatten(inputs)
        board_embedding = self.token_embeddings(board_embedding)
        board_embedding = self.masking_layer(board_embedding)
        return board_embedding









