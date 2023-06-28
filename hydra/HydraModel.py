import tensorflow as tf
import tensorflow_ranking as tfr
from keras import layers
import os
from keras.utils import plot_model
import config
from hydra.HydraEncoder import HydraEncoder



def build_model(mode):
    board_inputs = layers.Input(shape=(8, 8, 12,), name="board")
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")
    hydra = HydraEncoder(mode=mode)
    output = hydra(board_inputs, move_inputs)
    model = HydraModel([board_inputs, move_inputs], output, name=config.model_name)

    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.plots_dir, config.model_name + '-' + config.mode + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)

    return model



class HydraModel(tf.keras.Model):

    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    board_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    board_loss_tracker = tf.keras.metrics.Mean(name="board_loss")
    board_accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(name="board_accuracy")

    ft_loss_fn = tfr.keras.losses.ApproxNDCGLoss(name='loss')
    ft_loss_tracker = tf.keras.metrics.Mean(name="loss")
    ft_precision_tracker = tfr.keras.metrics.PrecisionMetric(name="accuracy", topn=3)


    ##################
    ### Train Step ###
    ##################

    def train_step(self, inputs):
        if config.mode == 'pt':
            return self.pt_train_step(inputs)
        elif config.mode == 'pt2':
            return self.pt2_train_step(inputs)
        elif config.mode == 'ft':
            return self.ft_train_step(inputs)

    def pt_train_step(self, inputs):
        features, labels, sample_weight, board = inputs
        with tf.GradientTape() as tape:
            predictions = self([board, features], training=True)
            loss = self.pt_loss_fn(labels, predictions, sample_weight=sample_weight)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.pt_loss_tracker.update_state(loss, sample_weight=sample_weight)
        self.pt_accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)
        return {"loss": self.pt_loss_tracker.result(), "accuracy": self.pt_accuracy_tracker.result()}


    def pt2_train_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        with tf.GradientTape() as tape:
            move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=True)
            move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
            board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)
            loss = move_loss + board_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.pt_loss_tracker.update_state(move_loss, sample_weight=move_seq_sample_weights)
        self.board_loss_tracker.update_state(board_loss, sample_weight=board_tensor_sample_weights)

        self.pt_accuracy_tracker.update_state(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        self.board_accuracy_tracker.update_state(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)
        return {
            "loss": self.pt_loss_tracker.result(),
            "accuracy": self.pt_accuracy_tracker.result(),
            "board_loss": self.board_loss_tracker.result(),
            "board_accuracy": self.board_accuracy_tracker.result(),
        }




    def ft_train_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor = inputs
        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.ft_loss_fn(relevancy_scores, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.ft_loss_tracker.update_state(loss)
        self.ft_precision_tracker.update_state(relevancy_scores, predictions)
        return {"loss": self.ft_loss_tracker.result(), "accuracy": self.ft_precision_tracker.result()}



    ##################
    ### Train Step ###
    ##################

    def test_step(self, inputs):
        if config.mode == 'pt':
            return self.pt_test_step(inputs)
        elif config.mode == 'pt2':
            return self.pt2_test_step(inputs)
        elif config.mode == 'ft':
            return self.ft_test_step(inputs)

    def pt_test_step(self, inputs):
        features, labels, sample_weight, board = inputs
        predictions = self([board, features], training=False)
        loss = self.pt_loss_fn(labels, predictions, sample_weight=sample_weight)
        self.pt_loss_tracker.update_state(loss, sample_weight=sample_weight)
        self.pt_accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)
        return {"loss": self.pt_loss_tracker.result(), "accuracy": self.pt_accuracy_tracker.result()}

    def pt2_test_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=False)
        move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)
        loss = move_loss + board_loss

        self.pt_loss_tracker.update_state(move_loss, sample_weight=move_seq_sample_weights)
        self.board_loss_tracker.update_state(board_loss, sample_weight=board_tensor_sample_weights)

        self.pt_accuracy_tracker.update_state(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        self.board_accuracy_tracker.update_state(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)

        return {
            "loss": self.pt_loss_tracker.result(),
            "accuracy": self.pt_accuracy_tracker.result(),
            "board_loss": self.board_loss_tracker.result(),
            "board_accuracy": self.board_accuracy_tracker.result(),
        }



    def ft_test_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor = inputs
        predictions = self([board_tensor, previous_moves], training=False)
        loss = self.ft_loss_fn(relevancy_scores, predictions)
        self.ft_loss_tracker.update_state(loss)
        self.ft_precision_tracker.update_state(relevancy_scores, predictions)
        return {"loss": self.ft_loss_tracker.result(), "accuracy": self.ft_precision_tracker.result()}

    @property
    def metrics(self):
        if config.mode == 'pt':
            return [self.pt_loss_tracker, self.pt_accuracy_tracker]
        elif config.mode == 'pt2':
            return [self.pt_loss_tracker, self.pt_accuracy_tracker, self.board_loss_tracker, self.board_accuracy_tracker]
        elif config.mode == 'ft':
            return [self.ft_loss_tracker, self.ft_precision_tracker]



