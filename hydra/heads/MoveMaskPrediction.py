from keras import layers
import config


class MoveMaskPrediction(layers.Layer):

    def __init__(self):
        super(MoveMaskPrediction, self).__init__()

        self.move_mask_prediction = layers.Dense(config.vocab_size, name="mask_prediction_head")
        self.activation = layers.Activation('softmax', dtype='float32')

    def __call__(self, inputs):
        move_mask = self.move_mask_prediction(inputs)
        move_mask = self.activation(move_mask)
        return move_mask


















