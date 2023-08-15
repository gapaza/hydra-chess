from keras import layers
import config
from tensorflow import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(package="Hydra", name="Router")
class Router(layers.Layer):

    def __init__(self, **kwargs):
        super(Router, self).__init__(name='router_head', **kwargs)
        self.router_weights = layers.Dense(config.num_experts, name='router_weights', dtype='float32')

    def call(self, inputs, training=False, mask=None):
        token_inputs, expert_capacity = inputs

        router_logits = self.router_weights(token_inputs)
        router_probs = tf.nn.softmax(router_logits, axis=-1)

        # This is simply the sequence length
        tokens_per_group = router_probs.shape[1]



        return None










    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


































