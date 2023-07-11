from keras import layers
import config


class MovePrediction(layers.Layer):

    def __init__(self):
        super(MovePrediction, self).__init__()

        self.next_move_avg = layers.GlobalAveragePooling1D()
        self.next_move_prediction = layers.Dense(config.vocab_size, name='next_move_prediction')
        self.activation = layers.Activation('relu', dtype='float32')


    def __call__(self, inputs):
        next_move = self.next_move_avg(inputs)
        next_move = self.next_move_prediction(next_move)
        next_move = self.activation(next_move)
        return next_move

