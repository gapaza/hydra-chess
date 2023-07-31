import keras
import tensorflow as tf
import config

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.DualDecoder import DualDecoder
from hydra.layers.CombinedEmbedding import CombinedEmbedding

# --> Keras Layers
from keras_nlp.layers import TransformerEncoder



@keras.saving.register_keras_serializable(package="Hydra", name="EncoderBase")
class EncoderBase(tf.keras.Model):
    def __init__(self):
        super().__init__(name='encoder_base')
        self.supports_masking = True
        self.dense_dim = config.dense_dim
        self.num_heads = config.heads
        self.positional = False

        # --> Combined Embedding
        self.combined_embedding = CombinedEmbedding()

        # --> Transformer Blocks
        normalize_first = False
        self.encoder_1 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_2 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_3 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_4 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_5 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_6 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_7 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.encoder_8 = TransformerEncoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)


    def call(self, inputs, training=False, mask=None):
        board_inputs, move_inputs = inputs

        # 1. Combined Positional Embedding
        combined_embedding = self.combined_embedding([move_inputs, board_inputs], training=training)

        # 3. Encoders
        combined_encoding = self.encoder_1(combined_embedding, training=training)
        combined_encoding = self.encoder_2(combined_encoding, training=training)
        combined_encoding = self.encoder_3(combined_encoding, training=training)
        combined_encoding = self.encoder_4(combined_encoding, training=training)
        combined_encoding = self.encoder_5(combined_encoding, training=training)
        combined_encoding = self.encoder_6(combined_encoding, training=training)
        combined_encoding = self.encoder_7(combined_encoding, training=training)
        combined_encoding = self.encoder_8(combined_encoding, training=training)

        # 4. Split combined encoding into move and board encoding
        split_idx = config.seq_length
        move_encoding = combined_encoding[:, :split_idx, :]
        board_encoding = combined_encoding[:, split_idx:, :]

        return move_encoding, board_encoding

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




