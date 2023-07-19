import keras
from keras import layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder

from hydra.heads.BoardPrediction import BoardPrediction
# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.SimpleBoardEmbedding import SimpleBoardEmbedding

from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MovePredictionSoftmax import MovePredictionSoftmax
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction



import config


class HydraDecoder(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(HydraDecoder, self).__init__(*args, **kwargs)

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=True)

        # --> Board Embedding
        self.board_embedding = SimpleBoardEmbedding('board_embedding', positional=True)

        # Board Encoders
        self.encoder_1 = TransformerEncoder(config.de_dense_dim, config.de_heads)
        self.encoder_2 = TransformerEncoder(config.de_dense_dim, config.de_heads)
        self.encoder_3 = TransformerEncoder(config.de_dense_dim, config.de_heads)
        self.encoder_4 = TransformerEncoder(config.de_dense_dim, config.de_heads)

        # Stack of 18 decoders
        normalize_first = True
        self.decoder_1 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        self.decoder_2 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        self.decoder_3 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        self.decoder_4 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_5 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_6 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_7 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_8 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_9 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_10 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_11 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_12 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_13 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_14 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_15 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_16 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_17 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_18 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_19 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_20 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_21 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_22 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_23 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_24 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_25 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_26 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_27 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_28 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_29 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_30 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_31 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)
        # self.decoder_32 = TransformerDecoder(config.de_dense_dim, config.de_heads, normalize_first=normalize_first)

        # --> Output Heads
        self.mask_span_prediction_head = MoveMaskPrediction()
        self.next_move_ranking_head = MovePrediction()
        self.next_move_prediction_head = MovePredictionSoftmax()
        self.board_prediction_head = BoardPrediction()


    def __call__(self, board_inputs, move_inputs, mask=None):

        # 1. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 2. Board Encoder
        board_encoding = self.encoder_1(board_embedding)
        board_encoding = self.encoder_2(board_encoding)
        board_encoding = self.encoder_3(board_encoding)
        board_encoding = self.encoder_4(board_encoding)

        # 2. Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # 3. Decoder Stack
        decoder_outputs = self.decoder_1(move_embedding, board_encoding)
        decoder_outputs = self.decoder_2(decoder_outputs, board_encoding)
        decoder_outputs = self.decoder_3(decoder_outputs, board_encoding)
        decoder_outputs = self.decoder_4(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_5(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_6(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_7(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_8(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_9(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_10(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_11(decoder_outputs, board_encoding)
        # decoder_outputs = self.decoder_12(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_13(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_14(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_15(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_16(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_17(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_18(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_19(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_20(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_21(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_22(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_23(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_24(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_25(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_26(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_27(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_28(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_29(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_30(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_31(decoder_outputs, board_embedding)
        # decoder_outputs = self.decoder_32(decoder_outputs, board_embedding)



        # 4. Output Heads
        if 'pt' in config.model_mode:
            if 'dual' in config.dc_mode:
                return self.mask_span_prediction_head(decoder_outputs), self.board_prediction_head(board_encoding)
            else:
                return self.mask_span_prediction_head(decoder_outputs)
        elif 'ft' in config.model_mode:
            if 'ndcg' in config.model_mode:
                return self.next_move_ranking_head(decoder_outputs)
            elif 'classify' in config.model_mode:
                return self.next_move_prediction_head(decoder_outputs)













