import keras
from keras import layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.SimpleBoardEmbedding import SimpleBoardEmbedding

from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MovePredictionSoftmax import MovePredictionSoftmax
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction



import config


class HydraDecoder(layers.Layer):

    def __init__(self, mode='ft', *args, **kwargs):
        super(HydraDecoder, self).__init__(*args, **kwargs)
        self.mode = mode

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=True)

        # --> Board Embedding
        # self.board_embedding = BoardEmbedding('board_embedding')
        self.board_embedding = SimpleBoardEmbedding('board_embedding', positional=True)

        # Stack of 18 decoders
        self.decoder_1 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_2 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_3 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_4 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_5 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_6 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_7 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        self.decoder_8 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_9 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_10 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_11 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_12 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_13 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_14 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_15 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_16 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_17 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)
        # self.decoder_18 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=True)

        # --> Output Heads
        self.next_move_prediction_head = MovePrediction()
        self.next_move_prediciton_softmax = MovePredictionSoftmax()
        self.mask_span_prediction_head = MoveMaskPrediction()


    def __call__(self, board_inputs, move_inputs, mask=None):

        # 1. Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # 2. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 3. Decoder Stack
        decoder_outputs = self.decoder_1(move_embedding, board_embedding)
        decoder_outputs = self.decoder_2(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_3(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_4(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_5(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_6(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_7(decoder_outputs, board_embedding)
        decoder_outputs = self.decoder_8(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_9(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_10(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_11(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_12(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_13(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_14(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_15(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_16(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_17(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_18(decoder_outputs, board_embedding)

        # 4. Output Heads
        if config.dc_mode == 'pt':
            return self.mask_span_prediction_head(decoder_outputs)
        elif config.dc_mode == 'ft':
            return self.next_move_prediciton_softmax(decoder_outputs)












