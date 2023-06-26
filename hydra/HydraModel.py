import tensorflow as tf
import tensorflow_ranking as tfr
from keras import layers
import config
from hydra.HydraEncoder import HydraEncoder
from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveMaskPrediction import MoveMaskPrediction













class HydraModel(tf.keras.Model):

    def __init__(self, mode='pt', **kwargs):
        super(HydraModel, self).__init__(**kwargs)
        self.mode = mode

        # --> Inputs Layers
        self.encoder = HydraEncoder()

        # --> Output Heads
        self.next_move_prediction_head = MovePrediction()
        self.mask_span_prediction_head = MoveMaskPrediction()

    def call(self, inputs, training=False):
        board_inputs = inputs[0]
        move_inputs = inputs[1]
        encoder_board_output, encoder_move_output = self.encoder(board_inputs, move_inputs, split=True)
        if self.mode == 'pt':
            return self.mask_span_prediction_head(encoder_move_output)
        elif self.mode == 'ft':
            return self.next_move_prediction_head(encoder_move_output)
        else:
            raise Exception('Invalid mode:', self.mode)


    #  _______           _         _                 _
    # |__   __|         (_)       (_)               | |
    #    | | _ __  __ _  _  _ __   _  _ __    __ _  | |      ___    ___   _ __
    #    | || '__|/ _` || || '_ \ | || '_ \  / _` | | |     / _ \  / _ \ | '_ \
    #    | || |  | (_| || || | | || || | | || (_| | | |____| (_) || (_) || |_) |
    #    |_||_|   \__,_||_||_| |_||_||_| |_| \__, | |______|\___/  \___/ | .__/
    #                                         __/ |                      | |
    #                                        |___/                       |_|


    pt_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    pt_loss_tracker = tf.keras.metrics.Mean(name="loss")
    pt_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    ft_loss_fn = tfr.keras.losses.ApproxNDCGLoss(name='loss')
    ft_loss_tracker = tf.keras.metrics.Mean(name="loss")
    ft_precision_tracker = tfr.keras.metrics.PrecisionMetric(name="accuracy", topn=3)


    ##################
    ### Train Step ###
    ##################

    def train_step(self, inputs):
        if self.mode == 'pt':
            return self.pt_train_step(inputs)
        elif self.mode == 'ft':
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
        if self.mode == 'pt':
            return self.pt_test_step(inputs)
        elif self.mode == 'ft':
            return self.ft_test_step(inputs)

    def pt_test_step(self, inputs):
        features, labels, sample_weight, board = inputs
        predictions = self([board, features], training=False)
        loss = self.pt_loss_fn(labels, predictions, sample_weight=sample_weight)
        self.pt_loss_tracker.update_state(loss, sample_weight=sample_weight)
        self.pt_accuracy_tracker.update_state(labels, predictions, sample_weight=sample_weight)
        return {"loss": self.pt_loss_tracker.result(), "accuracy": self.pt_accuracy_tracker.result()}

    def ft_test_step(self, inputs):
        previous_moves, relevancy_scores, board_tensor = inputs
        predictions = self([board_tensor, previous_moves], training=False)
        loss = self.ft_loss_fn(relevancy_scores, predictions)
        self.ft_loss_tracker.update_state(loss)
        self.ft_precision_tracker.update_state(relevancy_scores, predictions)
        return {"loss": self.ft_loss_tracker.result(), "accuracy": self.ft_precision_tracker.result()}

    @property
    def metrics(self):
        if self.mode == 'pt':
            return [self.pt_loss_tracker, self.pt_accuracy_tracker]
        elif self.mode == 'ft':
            return [self.ft_loss_tracker, self.ft_precision_tracker]



