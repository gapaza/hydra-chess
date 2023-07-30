
"""Transformer decoder block implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras
from keras_nlp.utils.keras_utils import clone_initializer


import config


@keras.saving.register_keras_serializable(package="Hydra", name="DualDecoder")
class DualDecoder(keras.layers.Layer):

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
        self._input_shape = kwargs.pop("build_input_shape", None)
        self._has_cross_attention = kwargs.pop("has_cross_attention", False)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self._built = False
        self.supports_masking = True


    def _build(self, move_input_shape, board_input_shape):
        self._built = True
        self._move_input_shape = move_input_shape
        self._board_input_shape = board_input_shape

        hidden_dim = config.embed_dim
        head_dim = int(hidden_dim // config.heads)
        
        # Move Self Attention Layers
        self._move_self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._move_self_attention_layer._build_from_signature(
            query=move_input_shape,
            value=move_input_shape,
        )
        self._move_self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._move_self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Board Self Attention Layers
        self._board_self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._board_self_attention_layer._build_from_signature(
            query=board_input_shape,
            value=board_input_shape,
        )
        self._board_self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._board_self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Move Cross Attention Layers
        self._move_cross_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._move_cross_attention_layer._build_from_signature(
            query=move_input_shape,
            value=move_input_shape,
        )
        self._move_cross_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._move_cross_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Board Cross Attention Layer
        self._board_cross_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._board_cross_attention_layer._build_from_signature(
            query=board_input_shape,
            value=board_input_shape,
        )
        self._board_cross_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._board_cross_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Move Feedforward layers.
        self._move_feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._move_feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._move_feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._move_feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )


        # Board Feedforward layers.
        self._board_feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._board_feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._board_feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._board_feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )
        

        return 0

    def call(self, inputs, training=False, move_mask=None):
        move_sequence, board_sequence = inputs

        if not self._built:
            self._build(move_sequence.shape, board_sequence.shape)

        x_move = move_sequence
        x_board = board_sequence

        ### Masking
        # 1. Self-Attention: Moves attending to moves
        # - mask out moves that are padding (False elements in move_mask)
        # - matrix of size (move_seq_len, move_seq_len) --> (128, 128)
        # 2. Cross-Attention: Moves attending to board
        # - no masking needed, all board squares visible
        # 3. Cross-Attention: Board attending to moves
        # - mask out moves that are padding (False elements in move_mask)
        # - matrix of size (board_seq_len, move_seq_len) --> (65, 128)
        move_self_attention_mask = None
        board_cross_attention_mask = None
        if move_mask is not None:
            move_mask_expanded = tf.expand_dims(move_mask, axis=-1)
            move_self_attention_mask = tf.math.logical_and(move_mask_expanded, tf.transpose(move_mask_expanded, perm=[0, 2, 1]))
            move_self_attention_mask = tf.cast(move_self_attention_mask, tf.int16)

            move_mask_expanded_2 = tf.expand_dims(move_mask, axis=1)
            board_cross_attention_mask = tf.repeat(move_mask_expanded_2, repeats=config.board_seq_length, axis=1)
            board_cross_attention_mask = tf.cast(board_cross_attention_mask, tf.int16)


        # 1. Compute Self Attention
        residual_move = x_move
        residual_board = x_board
        if self.normalize_first:
            x_move = self._move_self_attention_layernorm(x_move)
            x_board = self._board_self_attention_layernorm(x_board)

        # print('--> MOVE SELF ATTENTION INPUT SHAPES:', x_move.shape, move_self_attention_mask.shape)
        x_move = self._move_self_attention_layer(
            query=x_move,  # (batch, 128, 256)
            value=x_move,  # (batch, 128, 256)
            key=x_move,    # (batch, 128, 256)
            attention_mask=move_self_attention_mask  # (batch, 128, 128)
        )
        x_board = self._board_self_attention_layer(
            query=x_board,
            value=x_board,
            key=x_board
        )
        if training is True:
            x_move = self._move_self_attention_dropout(x_move)
            x_board = self._board_self_attention_dropout(x_board)
        x_move = x_move + residual_move
        x_board = x_board + residual_board
        if not self.normalize_first:
            x_move = self._move_self_attention_layernorm(x_move)
            x_board = self._board_self_attention_layernorm(x_board)


        # 2. Compute Cross Attention
        residual_move = x_move
        residual_board = x_board
        if self.normalize_first:
            x_move = self._move_cross_attention_layernorm(x_move)
            x_board = self._board_cross_attention_layernorm(x_board)
        x_move = self._move_cross_attention_layer(
            query=x_move,
            value=x_board,
            key=x_board,
        )
        x_board = self._board_cross_attention_layer(
            query=x_board,  # (batch, 65, 256)
            value=x_move,   # (batch, 128, 256)
            key=x_move,     # (batch, 128, 256)
            attention_mask=board_cross_attention_mask  # (batch, 65, 128)
        )
        if training is True:
            x_move = self._move_cross_attention_dropout(x_move)
            x_board = self._board_cross_attention_dropout(x_board)
        x_move = x_move + residual_move
        x_board = x_board + residual_board
        if not self.normalize_first:
            x_move = self._move_cross_attention_layernorm(x_move)
            x_board = self._board_cross_attention_layernorm(x_board)
            
            
        # 3. Compute Feed Forward
        residual_move = x_move
        residual_board = x_board
        if self.normalize_first:
            x_move = self._move_feedforward_layernorm(x_move)
            x_board = self._board_feedforward_layernorm(x_board)
        x_move = self._move_feedforward_intermediate_dense(x_move)
        x_move = self._move_feedforward_output_dense(x_move)
        x_board = self._board_feedforward_intermediate_dense(x_board)
        x_board = self._board_feedforward_output_dense(x_board)
        if training is True:
            x_move = self._move_feedforward_dropout(x_move)
            x_board = self._board_feedforward_dropout(x_board)
        x_move = x_move + residual_move
        x_board = x_board + residual_board
        if not self.normalize_first:
            x_move = self._move_feedforward_layernorm(x_move)
            x_board = self._board_feedforward_layernorm(x_board)

        return x_move, x_board


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
                "build_input_shape": self._input_shape,
                "has_cross_attention": self._has_cross_attention,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)