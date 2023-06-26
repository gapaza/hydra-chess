from keras import layers


class ModalityFusion(layers.Layer):

    def __init__(self):
        super(ModalityFusion, self).__init__()

        # --> Modality Fusion
        self.modality_fusion = layers.Concatenate(axis=1, name='modality_fusion')


    def __call__(self, board_embedding, move_embedding):
        modality_fusion = self.modality_fusion([board_embedding, move_embedding])
        return modality_fusion