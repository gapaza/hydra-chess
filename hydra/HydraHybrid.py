import keras
from keras import layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder

# --> Custom Layers
from hydra.layers.MoveEmbedding import MoveEmbedding
from hydra.layers.SimpleBoardEmbedding import SimpleBoardEmbedding

from hydra.layers.HybridDecoder import HybridDecoder
from hydra.layers.ModalityFusion import ModalityFusion


from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MovePredictionSoftmax import MovePredictionSoftmax
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction
from hydra.heads.BoardPrediction import BoardPrediction




import config


class HydraHybrid(layers.Layer):

    def __init__(self, *args, **kwargs):
        super(HydraHybrid, self).__init__(*args, **kwargs)

        # --> Move Embedding
        self.move_embedding = MoveEmbedding(positional=True)

        # --> Board Embedding
        self.board_embedding = SimpleBoardEmbedding('board_embedding', positional=True)

        # --> Stack of 16 Custom Decoders
        normalize_first = True
        self.decoder_1 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_2 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_3 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_4 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_5 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_6 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_7 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_8 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_9 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_10 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_11 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_12 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_13 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_14 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_15 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)
        self.decoder_16 = HybridDecoder(config.hy_dense_dim, config.hy_heads, normalize_first=normalize_first)


        # --> Modality Fusion
        self.modality_fusion = ModalityFusion()

        # --> Output Heads
        self.mask_span_prediction_head = MoveMaskPrediction()
        self.next_move_ranking_head = MovePrediction()
        self.next_move_prediction_head = MovePredictionSoftmax()
        self.board_prediction_head = BoardPrediction()



    def __call__(self, board_inputs, move_inputs, mask=None):

        # 1. Board Embedding
        board_embedding = self.board_embedding(board_inputs)

        # 2. Move Embedding
        move_embedding = self.move_embedding(move_inputs)

        # 3. Hybrid Decoders
        # small: 4 decoders
        # normal: 8 decoders
        # large: 16 decoders
        # xlarge: 32 decoders
        # ultra: 64 decoders


        # Stack of 16 Custom Decoders
        move_encoding, board_encoding = self.decoder_1(move_embedding, board_embedding)
        move_encoding, board_encoding = self.decoder_2(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_3(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_4(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_5(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_6(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_7(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_8(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_9(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_10(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_11(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_12(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_13(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_14(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_15(move_encoding, board_encoding)
        move_encoding, board_encoding = self.decoder_16(move_encoding, board_encoding)


        # 7. Pass through output head
        if 'pt' in config.model_mode:
            return self.mask_span_prediction_head(move_encoding), self.board_prediction_head(board_encoding)
        elif 'ft' in config.model_mode:
            if 'single' in config.hy_mode:
                combined_encoding = self.modality_fusion(move_encoding, board_encoding)
                if 'ndcg' in config.model_mode:
                    return self.next_move_ranking_head(combined_encoding)
                elif 'classify' in config.model_mode:
                    return self.next_move_prediction_head(combined_encoding)
            elif 'dual' in config.hy_mode:
                if 'ndcg' in config.model_mode:
                    return self.next_move_ranking_head(move_encoding)
                elif 'classify' in config.model_mode:
                    return self.next_move_prediction_head(move_encoding)

















