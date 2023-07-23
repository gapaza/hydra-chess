import tensorflow as tf
import tensorflow_ranking as tfr
from keras import layers
import os
from keras.utils import plot_model
import config
from hydra.HydraHybrid import HydraHybrid



def build_model():

    # 1. Inputs
    move_inputs = layers.Input(shape=(config.seq_length,), name="moves")
    board_inputs = layers.Input(shape=(8, 8), name="board")

    # 2. Model
    hydra = HydraHybrid()
    output = hydra(board_inputs, move_inputs)
    model = HydraHybridModel([board_inputs, move_inputs], output, name=config.model_name)

    # 3. Visualize
    model.summary(expand_nested=True)
    model_img_file = os.path.join(config.plots_dir, config.model_name + '-' + config.model_mode + '.png')
    plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)

    return model



class HydraHybridModel(tf.keras.Model):

    ###############################
    ### Move and Board Modeling ###
    ###############################

    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    board_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    board_loss_tracker = tf.keras.metrics.Mean(name="board_loss")
    board_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="board_accuracy")

    move_pred_weight = 3.0
    board_pred_weight = 1.0

    ############################
    ### NDCG Move Prediction ###
    ############################

    ft_ndcg_loss_fn = tfr.keras.losses.ApproxNDCGLoss(name='loss', reduction=tf.keras.losses.Reduction.NONE)
    ft_ndcg_loss_tracker = tf.keras.metrics.Mean(name="loss")
    ft_ndcg_precision_tracker = tfr.keras.metrics.PrecisionMetric(name="accuracy", topn=3)
    ft_ndcg_precision_tracker_t1 = tfr.keras.metrics.PrecisionMetric(name="accuracy_t1", topn=1)


    ###############################
    ### Softmax Move Prediction ###
    ###############################

    ft_classify_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    ft_classify_loss_tracker = tf.keras.metrics.Mean(name="loss")
    ft_classify_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    ##################
    ### Train Step ###
    ##################

    def train_step(self, inputs):
        if 'pt' in config.model_mode:
            return self.pt_train_step(inputs)
        elif 'ft' in config.model_mode:
            if 'ndcg' in config.model_mode:
                return self.ft_train_step_ndcg(inputs)
            elif 'classify' in config.model_mode:
                return self.ft_train_step_classify(inputs)

    def pt_train_step(self, inputs):
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        with tf.GradientTape() as tape:
            move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=True)

            distributed_move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
            distributed_board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)
            move_loss = tf.nn.compute_average_loss(distributed_move_loss, global_batch_size=config.global_batch_size)
            board_loss = tf.nn.compute_average_loss(distributed_board_loss, global_batch_size=config.global_batch_size)
            loss = (self.move_pred_weight * move_loss) + (self.board_pred_weight * board_loss)
            loss = self.optimizer.get_scaled_loss(loss)


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

    def ft_train_step_classify(self, inputs):
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        label_indices = tf.argmax(relevancy_scores, axis=-1)
        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.ft_classify_loss_fn(label_indices, predictions)

            # DISTRIBUTED TRAINING
            if config.distributed is True:
                loss = tf.nn.compute_average_loss(loss, global_batch_size=config.global_batch_size)

            # MIXED PRECISION
            loss = self.optimizer.get_scaled_loss(loss)

        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.ft_classify_loss_tracker.update_state(loss)
        self.ft_classify_accuracy_tracker.update_state(label_indices, predictions)
        return {
            "loss": self.ft_classify_loss_tracker.result(),
            "accuracy": self.ft_classify_accuracy_tracker.result(),
        }

    def ft_train_step_ndcg(self, inputs):
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        with tf.GradientTape() as tape:
            predictions = self([board_tensor, previous_moves], training=True)
            loss = self.ft_ndcg_loss_fn(relevancy_scores, predictions)

            # DISTRIBUTED TRAINING
            if config.distributed is True:
                loss = tf.nn.compute_average_loss(loss, global_batch_size=config.global_batch_size)

            # MIXED PRECISION
            loss = self.optimizer.get_scaled_loss(loss)

        trainable_vars = self.trainable_variables
        scaled_gradients = tape.gradient(loss, trainable_vars)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.ft_ndcg_loss_tracker.update_state(loss)
        self.ft_ndcg_precision_tracker.update_state(relevancy_scores, predictions)
        self.ft_ndcg_precision_tracker_t1.update_state(relevancy_scores, predictions)
        return {
            "loss": self.ft_ndcg_loss_tracker.result(),
            "accuracy": self.ft_ndcg_precision_tracker.result(),
            "accuracy_t1": self.ft_ndcg_precision_tracker_t1.result()
        }

    #################
    ### Test Step ###
    #################

    def test_step(self, inputs):
        if 'pt' in config.model_mode:
            return self.pt_test_step(inputs)
        elif 'ft' in config.model_mode:
            if 'ndcg' in config.model_mode:
                return self.ft_test_step_ndcg(inputs)
            elif 'classify' in config.model_mode:
                return self.ft_test_step_classify(inputs)

    def pt_test_step(self, inputs):

        # PREDICTIONS
        move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = inputs
        move_predictions, board_predictions = self([board_tensor_masked, move_seq_masked], training=False)

        # LOSS
        move_loss = self.pt_loss_fn(move_seq_labels, move_predictions, sample_weight=move_seq_sample_weights)
        board_loss = self.board_loss_fn(board_tensor_labels, board_predictions, sample_weight=board_tensor_sample_weights)

        # DISTRIBUTED TRAINING
        move_loss = tf.nn.compute_average_loss(move_loss, global_batch_size=config.global_batch_size)
        board_loss = tf.nn.compute_average_loss(board_loss, global_batch_size=config.global_batch_size)

        # UPDATE TRACKERS
        self.pt_loss_tracker.update_state(move_loss, sample_weight=move_seq_sample_weights)
        self.board_loss_tracker.update_state(board_loss, sample_weight=board_tensor_sample_weights)
        self.pt_accuracy_tracker.update_state(move_seq_labels, move_predictions,
                                              sample_weight=move_seq_sample_weights)
        self.board_accuracy_tracker.update_state(board_tensor_labels, board_predictions,
                                                 sample_weight=board_tensor_sample_weights)
        return {
            "loss": self.pt_loss_tracker.result(),
            "accuracy": self.pt_accuracy_tracker.result(),
            "board_loss": self.board_loss_tracker.result(),
            "board_accuracy": self.board_accuracy_tracker.result(),
        }

    def ft_test_step_classify(self, inputs):

        # PREDICTIONS
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        label_indices = tf.argmax(relevancy_scores, axis=-1)
        predictions = self([board_tensor, previous_moves], training=False)

        # LOSS
        loss = self.ft_classify_loss_fn(label_indices, predictions)

        # DISTRIBUTED TRAINING
        if config.distributed is True:
            loss = tf.nn.compute_average_loss(loss, global_batch_size=config.global_batch_size)

        # UPDATE TRACKERS
        self.ft_classify_loss_tracker.update_state(loss)
        self.ft_classify_accuracy_tracker.update_state(label_indices, predictions)
        return {
            "loss": self.ft_classify_loss_tracker.result(),
            "accuracy": self.ft_classify_accuracy_tracker.result(),
        }

    def ft_test_step_ndcg(self, inputs):

        # PREDICTIONS
        previous_moves, relevancy_scores, board_tensor, sample_weights = inputs
        predictions = self([board_tensor, previous_moves], training=False)

        # LOSS
        loss = self.ft_ndcg_loss_fn(relevancy_scores, predictions)

        # DISTRIBUTED TRAINING
        if config.distributed is True:
            loss = tf.nn.compute_average_loss(loss, global_batch_size=config.global_batch_size)

        # UPDATE TRACKERS
        self.ft_ndcg_loss_tracker.update_state(loss)
        self.ft_ndcg_precision_tracker.update_state(relevancy_scores, predictions)
        self.ft_ndcg_precision_tracker_t1.update_state(relevancy_scores, predictions)
        return {
            "loss": self.ft_ndcg_loss_tracker.result(),
            "accuracy": self.ft_ndcg_precision_tracker.result(),
            "accuracy_t1": self.ft_ndcg_precision_tracker_t1.result()
        }

    @property
    def metrics(self):
        if 'pt' in config.model_mode:
            return [self.pt_loss_tracker, self.pt_accuracy_tracker, self.board_loss_tracker]
        elif 'ft' in config.model_mode:
            if 'ndcg' in config.model_mode:
                return [self.ft_ndcg_loss_tracker, self.ft_ndcg_precision_tracker,
                        self.ft_ndcg_precision_tracker_t1]
            elif 'classify' in config.model_mode:
                return [self.ft_classify_loss_tracker, self.ft_classify_accuracy_tracker]


















