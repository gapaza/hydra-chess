import keras
import tensorflow as tf
import config

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.DualDecoder import DualDecoder





@keras.saving.register_keras_serializable(package="Hydra", name="HydraBase")
class HydraBase(tf.keras.Model):
    def __init__(self):
        super().__init__(name='hydra_base')
        self.supports_masking = True
        self.dense_dim = config_new.dense_dim
        self.num_heads = config_new.heads

        # --> Move Embedding
        self.move_embedding = MoveEmbedding()

        # --> Board Embedding
        self.board_embedding = BoardEmbedding()

        # --> Transformer Blocks
        normalize_first = True
        self.decoder_1 = DualDecoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.decoder_2 = DualDecoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.decoder_3 = DualDecoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)
        self.decoder_4 = DualDecoder(self.dense_dim, self.num_heads, normalize_first=normalize_first)


    def call(self, inputs, training=False, mask=None):
        board_inputs, move_inputs = inputs

        # 1. Board Embedding
        board_embedding = self.board_embedding(board_inputs, training=training)

        # 2. Move Embedding
        move_embedding = self.move_embedding(move_inputs, training=training)
        move_mask = self.move_embedding.compute_mask(move_inputs)

        # 3. Custom Decoders
        move_encoding, board_encoding = self.decoder_1([move_embedding, board_embedding], training=training, move_mask=move_mask)
        move_encoding, board_encoding = self.decoder_2([move_encoding, board_encoding], training=training, move_mask=move_mask)
        move_encoding, board_encoding = self.decoder_3([move_encoding, board_encoding], training=training, move_mask=move_mask)
        move_encoding, board_encoding = self.decoder_4([move_encoding, board_encoding], training=training, move_mask=move_mask)

        return move_encoding, board_encoding

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




