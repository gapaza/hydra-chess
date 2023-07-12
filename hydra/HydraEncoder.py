import keras
from keras import layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.BoardEmbedding import BoardEmbedding
from hydra.layers.SimpleBoardEmbedding import SimpleBoardEmbedding
from hydra.layers.ModalityFusion import ModalityFusion
from hydra.layers.VisualEncoder import VisualEncoder
from hydra.layers.PositionalEmbedding import PositionalEmbedding

from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction
from hydra.heads.BoardPrediction import BoardPrediction


import config


class HydraEncoder(layers.Layer):

    def __init__(self, mode='ft', *args, **kwargs):
        super(HydraEncoder, self).__init__(*args, **kwargs)
        self.mode = mode

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=False)

        # --> Board Embedding
        # self.board_embedding = BoardEmbedding('board_embedding')
        self.board_embedding = SimpleBoardEmbedding('board_embedding')

        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Position Embeddings
        self.positional_embedding = PositionalEmbedding()

        # --> Encoders
        self.encoder_stack = keras.Sequential([

            # Stack of 12
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),

            # Stack of 12
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),
            # TransformerEncoder(config.encoder_dense_dim, config.encoder_heads),



        ], name='encoder_stack')

        # --> Output Heads
        self.next_move_prediction_head = MovePrediction()
        self.mask_span_prediction_head = MoveMaskPrediction()
        self.board_prediction_head = BoardPrediction()


    def __call__(self, board_inputs, move_inputs, mask=None):

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
        split_idx = config.vt_num_patches
        encoder_board_output = encoder_outputs[:, :split_idx, :]
        encoder_move_output = encoder_outputs[:, split_idx:, :]

        # 7. Pass through output head
        if self.mode == 'pt':
            return self.mask_span_prediction_head(encoder_move_output), self.board_prediction_head(encoder_board_output)
        elif self.mode == 'ft':
            return self.next_move_prediction_head(encoder_outputs)











