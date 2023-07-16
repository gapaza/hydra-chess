import tensorflow as tf
import tensorflow_ranking as tfr
from keras import layers
import os
from keras.utils import plot_model
import config
from hydra.HydraEncoder import HydraEncoder
from hydra.HydraDecoder import HydraDecoder

from preprocess.FT_DatasetGenerator import FT_DatasetGenerator

def build_model_encoder(mode):

    # 1. Inputs
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")
    board_inputs = layers.Input(shape=(8, 8), name="board")

    # 2. Model
    hydra = HydraEncoder(mode=mode)
    output = hydra(board_inputs, move_inputs)
    model = HydraModel([board_inputs, move_inputs], output, name=config.model_name)

    # 3. Visualize
    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.plots_dir, config.model_name + '-' + config.mode + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)

    return model


def build_model_decoder(mode):

    # 1. Inputs
    board_inputs = layers.Input(shape=(8, 8), name="board")
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")

    # 2. Model
    hydra = HydraDecoder(mode=mode)
    output = hydra(board_inputs, move_inputs)
    model = HydraModel([board_inputs, move_inputs], output, name=config.model_name)

    # 3. Visualize
    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.plots_dir, config.model_name + '-' + config.mode + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)

    return model







class HydraModel(tf.keras.Model):

    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    board_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    board_loss_tracker = tf.keras.metrics.Mean(name="board_loss")
    board_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="board_accuracy")

    move_pred_weight = 1.0
    board_pred_weight = 1.0


    ft_loss_fn = tfr.keras.losses.ApproxNDCGLoss(name='loss')
    ft_loss_tracker = tf.keras.metrics.Mean(name="loss")
    ft_precision_tracker = tfr.keras.metrics.PrecisionMetric(name="accuracy", topn=3)
    ft_precision_tracker_t1 = tfr.keras.metrics.PrecisionMetric(name="accuracy_t1", topn=1)




    ###############################
    ### Softmax Move Prediciton ###
    ###############################

    dc_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    dc_loss_tracker = tf.keras.metrics.Mean(name="loss")
    dc_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")



    ##################
    ### Train Step ###
    ##################

    def train_step(self, inputs):
        if config.mode == 'pt':
            return self.pt_train_step(inputs)
        elif config.mode == 'ft':
            return self.ft_train_step(inputs)
        elif config.mode == 'dc':
            return self.dc_train_step(inputs)


    def pt_train_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        with tf.GradientTape() as tape:
            move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=True)
            move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
            board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)

            move_loss = self.optimizer.get_scaled_loss(move_loss)
            board_loss = self.optimizer.get_scaled_loss(board_loss)
            loss = (self.move_pred_weight * move_loss) + (self.board_pred_weight * board_loss)

        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
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
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.ft_loss_fn(relevancy_scores, predictions)
            loss = self.optimizer.get_scaled_loss(loss)

        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.ft_loss_tracker.update_state(loss)
        self.ft_precision_tracker.update_state(relevancy_scores, predictions)
        self.ft_precision_tracker_t1.update_state(relevancy_scores, predictions)

        return {
            "loss": self.ft_loss_tracker.result(),
            "accuracy": self.ft_precision_tracker.result(),
            "accuracy_t1": self.ft_precision_tracker_t1.result()
        }








    def dc_train_step(self, inputs):
        if config.dc_mode == 'pt':
            return self.dc_pt_train_step(inputs)
        elif config.dc_mode == 'ft':
            return self.dc_ft_train_step(inputs)


    def dc_pt_train_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        board_tensor_labels = tf.reshape(board_tensor_labels, (-1, 8, 8))
        with tf.GradientTape() as tape:
            move_predictions = self([board_tensor_labels, move_seq_masked], training=True)
            move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
            move_loss = self.optimizer.get_scaled_loss(move_loss)
            loss = move_loss
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.pt_loss_tracker.update_state(move_loss, sample_weight=move_seq_sample_weights)
        self.pt_accuracy_tracker.update_state(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        return {
            "loss": self.pt_loss_tracker.result(),
            "accuracy": self.pt_accuracy_tracker.result(),
        }

    def dc_ft_train_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        label_indices = tf.argmax(relevancy_scores, axis=-1)

        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.dc_loss_fn(label_indices, predictions)
            loss = self.optimizer.get_scaled_loss(loss)
        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.dc_loss_tracker.update_state(loss)
        self.dc_accuracy_tracker.update_state(label_indices, predictions)

        return {
            "loss": self.dc_loss_tracker.result(),
            "accuracy": self.dc_accuracy_tracker.result(),
        }




    #################
    ### Test Step ###
    #################

    def test_step(self, inputs):
        if config.mode == 'pt':
            return self.pt_test_step(inputs)
        elif config.mode == 'ft':
            return self.ft_test_step(inputs)
        elif config.mode == 'dc':
            return self.dc_test_step(inputs)


    def pt_test_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=False)
        move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)
        move_loss = self.optimizer.get_scaled_loss(move_loss)
        board_loss = self.optimizer.get_scaled_loss(board_loss)
        loss = (self.move_pred_weight * move_loss) + (self.board_pred_weight * board_loss)

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
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        predictions = self([board_tensor, previous_moves], training=False)
        loss = self.ft_loss_fn(relevancy_scores, predictions)
        loss = self.optimizer.get_scaled_loss(loss)
        self.ft_loss_tracker.update_state(loss)
        self.ft_precision_tracker.update_state(relevancy_scores, predictions)
        self.ft_precision_tracker_t1.update_state(relevancy_scores, predictions)
        return {
            "loss": self.ft_loss_tracker.result(),
            "accuracy": self.ft_precision_tracker.result(),
            "accuracy_t1": self.ft_precision_tracker_t1.result()
        }










    def dc_test_step(self, inputs):
        if config.dc_mode == 'pt':
            return self.dc_pt_train_step(inputs)
        elif config.dc_mode == 'ft':
            return self.dc_ft_test_step(inputs)


    def dc_pt_test_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        board_tensor_labels = tf.reshape(board_tensor_labels, (-1, 8, 8))
        move_predictions = self([board_tensor_labels, move_seq_masked], training=False)
        move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        move_loss = self.optimizer.get_scaled_loss(move_loss)
        loss = move_loss
        self.pt_loss_tracker.update_state(move_loss, sample_weight=move_seq_sample_weights)
        self.pt_accuracy_tracker.update_state(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        return {
            "loss": self.pt_loss_tracker.result(),
            "accuracy": self.pt_accuracy_tracker.result(),
        }

    def dc_ft_test_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        label_indices = tf.argmax(relevancy_scores, axis=-1)

        predictions = self([board_tensor, previous_moves], training=False)
        loss = self.dc_loss_fn(label_indices, predictions)
        loss = self.optimizer.get_scaled_loss(loss)
        self.dc_loss_tracker.update_state(loss)
        self.dc_accuracy_tracker.update_state(label_indices, predictions)
        return {
            "loss": self.dc_loss_tracker.result(),
            "accuracy": self.dc_accuracy_tracker.result(),
        }


    @property
    def metrics(self):
        if config.mode == 'pt':
            return [self.pt_loss_tracker, self.pt_accuracy_tracker, self.board_loss_tracker]
        elif config.mode == 'ft':
            return [self.ft_loss_tracker, self.ft_precision_tracker, self.ft_precision_tracker_t1]
        elif config.mode == 'dc':
            if config.dc_mode == 'pt':
                return [self.pt_loss_tracker, self.pt_accuracy_tracker]
            elif config.dc_mode == 'ft':
                return [self.dc_loss_tracker, self.dc_accuracy_tracker]


