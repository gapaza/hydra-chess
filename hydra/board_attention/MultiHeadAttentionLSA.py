from keras import layers
import tensorflow as tf
import config
import math



class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_length = config.vt_num_patches + config.seq_length

        # Diagonal Attention Mask
        self.diag_attn_mask = 1 - tf.eye(self.total_length)
        self.diag_attn_mask = tf.cast([self.diag_attn_mask], dtype=tf.int8)

        # Learnable Temperature Scaling
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)


    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, self.diag_attn_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores
