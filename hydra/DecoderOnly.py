import keras
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import SinePositionEncoding
# from keras_nlp.layers import RotaryEmbedding


@keras.saving.register_keras_serializable(package="DecoderOnly", name="DecoderOnly")
class DecoderOnly(tf.keras.Model):
    def __init__(self):
        super().__init__(name='hydra_base')
        self.supports_masking = True
        self.dense_dim = config.dense_dim
        self.num_heads = config.heads
        self.positional = True

        # Move Embeddings
        self.move_embedding_layer = TokenAndPositionEmbedding(
            config.vocab_size,
            config.seq_length,
            config.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=config.dropout)
        self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=config.dropout)

        # Move Prediction Head
        self.move_prediction_head = keras.layers.Dense(
            config.vocab_size,
            name="move_prediction_head",
            activation="linear",
        )

    def call(self, inputs, training=False):
        # Move Embeddings
        move_embeddings = self.move_embedding_layer(inputs)

        # Decoder Stack
        decoder_1_output = self.decoder_1(move_embeddings, training=training)
        decoder_2_output = self.decoder_2(decoder_1_output, training=training)

        # Move Prediction Head
        move_predictions = self.move_prediction_head(decoder_2_output)

        return move_predictions

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")

    pt_perplexity_tracker = keras_nlp.metrics.Perplexity(name="perplexity", from_logits=True)

    def train_step(self, inputs):
        input_sequences, target_sequences = inputs
        with tf.GradientTape() as tape:
            # Forward Pass
            predictions = self(input_sequences, training=True)
            loss = self.pt_loss_fn(target_sequences, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.pt_loss_tracker.update_state(loss)
        self.pt_perplexity_tracker.update_state(target_sequences, predictions)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    def test_step(self, inputs):
        input_sequences, target_sequences = inputs
        predictions = self(input_sequences, training=False)
        loss = self.pt_loss_fn(target_sequences, predictions)
        self.pt_loss_tracker.update_state(loss)
        self.pt_perplexity_tracker.update_state(target_sequences, predictions)

        return {"loss": self.pt_loss_tracker.result(), "perplexity": self.pt_perplexity_tracker.result()}

    @property
    def metrics(self):
        return [self.pt_loss_tracker, self.pt_perplexity_tracker]














