from keras import layers
import tensorflow as tf
import config


class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --> Position Embeddings
        self.position_embedding = layers.Embedding(
            input_dim=config.vt_num_patches, output_dim=config.embed_dim
        )

    def __call__(self, encoded_patches):
        positions = tf.range(start=0, limit=config.vt_num_patches, delta=1)
        encoded_positions = self.position_embedding(positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches
