import tensorflow as tf
from tensorflow import keras
import config

from keras.layers import MultiHeadAttention
from keras_nlp.utils.keras_utils import clone_initializer

from hydra.layers.MultiHeadSequentialCrossAttention import MultiHeadSequentialCrossAttention


@keras.saving.register_keras_serializable(package="Hydra", name="StateSequenceDecoder")
class StateSequenceDecoder(keras.layers.Layer):

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        name=None,
        **kwargs,
    ):
        # Work around for model saving, we need to ensure our model is built
        # immediately after restoring from config.
        decoder_sequence_shape = kwargs.pop("decoder_sequence_shape", None)
        encoder_sequence_shape = kwargs.pop("encoder_sequence_shape", None)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True
        self._decoder_sequence_shape = None
        self._encoder_sequence_shape = None

        if decoder_sequence_shape:
            self.build(decoder_sequence_shape, encoder_sequence_shape)



    def build(
        self,
        decoder_sequence_shape,
        encoder_sequence_shape=None,
    ):
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = decoder_sequence_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=decoder_sequence_shape,
                value=decoder_sequence_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=decoder_sequence_shape,
                value_shape=decoder_sequence_shape,
            )
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Cross attention layers are NOT optional.
        self._cross_attention_layer = None
        self._cross_attention_layer = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        if hasattr(self._cross_attention_layer, "_build_from_signature"):
            self._cross_attention_layer._build_from_signature(
                query=encoder_sequence_shape,
                value=encoder_sequence_shape,
            )
        else:
            self._cross_attention_layer.build(
                query_shape=encoder_sequence_shape,
                value_shape=encoder_sequence_shape,
            )
        self._cross_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._cross_attention_layernorm.build(encoder_sequence_shape)
        self._cross_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )




        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )
        # Create layers based on input shape.
        self.built = True























