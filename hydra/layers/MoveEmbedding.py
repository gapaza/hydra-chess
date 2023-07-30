import tensorflow as tf
from keras import layers
import numpy as np
import config
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Hydra", name="MoveEmbedding")
class MoveEmbedding(layers.Layer):

    def __init__(self):
        super(MoveEmbedding, self).__init__(name='move_embedding_block')
        self.supports_masking = True

        # --> Parameters
        self.embed_dim = config_new.embed_dim
        self.seq_length = config_new.seq_length
        self.vocab_size = config_new.vocab_size

        # --> Token Embeddings
        self.token_embeddings = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            name="move_embedding",
            mask_zero=False
        )

        # --> Position Embeddings
        self.token_position_embeddings = layers.Embedding(
            input_dim=self.seq_length,
            output_dim=self.embed_dim,
            weights=[self.get_pos_encoding_matrix(self.seq_length, self.embed_dim)],
            name="move_positional_embedding",
        )

    def call(self, inputs, training=False, mask=None):
        token_embeddings = self.token_embeddings(inputs)
        token_position_embeddings = self.token_position_embeddings(tf.range(start=0, limit=self.seq_length, delta=1))
        move_embedding = token_embeddings + token_position_embeddings
        return move_embedding

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

    def compute_mask(self, inputs, mask=None):
        mask = tf.math.logical_not(tf.math.equal(inputs, config_new.padding_token_id))
        return mask

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




