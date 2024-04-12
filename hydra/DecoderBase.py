import keras
import tensorflow as tf
import config




@keras.saving.register_keras_serializable(package="Hydra", name="DecoderBase")
class DecoderBase(tf.keras.Model):

    def __init__(self):
        super().__init__(name='decoder_base')
        self.supports_masking = True
        self.dense_dim = config.dense_dim
        self.num_heads = config.heads
        self.positional = False














