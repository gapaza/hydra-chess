import keras
from keras import layers
from keras_nlp.layers import TransformerDecoder

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.ModalityFusion import ModalityFusion
from hydra.layers.VisualEncoder import VisualEncoder
from hydra.layers.PositionalEmbedding import PositionalEmbedding


import config


class HydraEncoder(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(HydraEncoder, self).__init__(*args, **kwargs)

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=False)

        # --> Board Embedding
        self.board_embedding = BoardEmbedding('board_embedding')

        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Position Embeddings
        self.positional_embedding = PositionalEmbedding()

        # --> Encoders
        self.encoder_stack = keras.Sequential([
            VisualEncoder(),
            # VisualEncoder(),
            # VisualEncoder(),
            # VisualEncoder()
        ], name='encoder_stack')


    def __call__(self, board_inputs, move_inputs, mask=None, split=False):

        # 1. Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # 2. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 3. Combine Board and Move Embeddings
        combined_embedding = self.modality_fusion(board_embedding, move_embedding)

        # 4. Positional Embedding
        combined_positional_embedding = self.positional_embedding(combined_embedding)

        # 5. Encoder Stack
        encoder_outputs = self.encoder_stack(combined_positional_embedding)

        # 6. Split Output
        if split:
            split_idx = config.vt_num_patches
            encoder_board_output = encoder_outputs[:, :split_idx, :]
            encoder_move_output = encoder_outputs[:, split_idx:, :]
            return encoder_board_output, encoder_move_output
        else:
            return encoder_outputs










