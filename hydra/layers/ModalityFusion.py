from keras import layers
import tensorflow as tf

class ModalityFusion(layers.Layer):

    def __init__(self):
        super(ModalityFusion, self).__init__()
        self.supports_masking = True

        # --> Modality Fusion
        self.modality_fusion = layers.Concatenate(axis=1, name='modality_fusion')


    def __call__(self, board_embedding, move_embedding):
        # print('move_embedding: ', move_embedding.shape, move_embedding._keras_mask)
        # batch_size = tf.shape(board_embedding)[0]
        # mask_patch_tokenization = tf.ones((1, 64), dtype=bool)
        # mask_patch_tokenization = tf.broadcast_to(mask_patch_tokenization, [batch_size, 64])
        # print(mask_patch_tokenization)
        # board_embedding._keras_mask = mask_patch_tokenization
        # print('board_embedding: ', board_embedding.shape, board_embedding._keras_mask)


        modality_fusion = self.modality_fusion([board_embedding, move_embedding])
        return modality_fusion