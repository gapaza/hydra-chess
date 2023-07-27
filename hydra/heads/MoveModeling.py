from keras import layers
import config


class MoveModeling(layers.Layer):

    def __init__(self):
        super(MoveModeling, self).__init__()

        # 1. Required Layers
        self.move_mask_prediction = layers.Dense(config.vocab_size, name="mask_prediction_head")
        self.activation = layers.Activation('softmax', dtype='float32')

        # 2. Additional Layers
        # self.dense_1 = layers.Dense(config.embed_dim, activation='relu')
        # self.norm_1 = layers.LayerNormalization()
        # self.move_mask_prediction = layers.Dense(config.vocab_size, name="mask_prediction_head")
        # self.activation = layers.Activation('softmax', dtype='float32')

    def __call__(self, inputs):
        move_mask = inputs

        # move_mask = self.dense_1(move_mask)
        move_mask = self.move_mask_prediction(move_mask)
        move_mask = self.activation(move_mask)

        return move_mask


















