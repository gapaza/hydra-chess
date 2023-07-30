from hydra.Hydra import Hydra
from hydra.HydraBase import HydraBase

from hydra.heads.GameModeling import GameModeling
from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveRanking import MoveRanking
from hydra.checkpoints.SaveCheckpoint import SaveCheckpoint
from hydra.utils import plot_history

import tensorflow as tf
import config
from keras.utils import plot_model
import os
from keras import layers


def get_model(head, checkpoint_path=None):
    if checkpoint_path:
        hydra, hydra_base, hydra_head = transfer_model(head, checkpoint_path)
    else:
        hydra, hydra_base, hydra_head = new_model(head)

    # 1. Build model
    # implicit_build(hydra, summary=False)

    # 2. Visualize
    hydra.summary(expand_nested=True)
    model_img_file = os.path.join(config_new.plots_dir, config_new.model_name + '-' + config_new.model_mode + '.png')
    plot_model(hydra, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=True)

    # 3. Return
    return hydra, hydra_base, hydra_head




def transfer_model(head, checkpoint_path):
    print('--> RESTORING MODEL')
    hydra_model, hydra_base, hydra_head = new_model(head)
    implicit_build(hydra_model)

    # --> IF USING WEIGHTS
    # hydra_model.load_weights(checkpoint_path)

    # --> IF USING CHECKPOINT
    checkpoint = tf.train.Checkpoint(hydra_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    # --> FREEZE LAYERS
    if config_new.tl_freeze_base:
        hydra_model.hydra_base.trainable = False

    return hydra_model, hydra_base, hydra_head



def new_model(head):
    hydra_base = get_base()
    hydra_head = get_head(head)
    hydra_model = Hydra(hydra_base, hydra_head)
    return hydra_model, hydra_base, hydra_head


def get_base():
    return HydraBase()


def get_head(head):
    hydra_head = None
    if head == 'game-modeling':
        hydra_head = GameModeling()
    elif head == 'move-prediction':
        hydra_head = MovePrediction()
    elif head == 'move-ranking':
        hydra_head = MoveRanking()
    return hydra_head


def implicit_build(model, summary=False):
    board_input = tf.zeros((1, 65))
    move_input = tf.zeros((1, 128))
    model([board_input, move_input])
    if summary is True:
        model.summary()






# def get_base(base_path=None):
#     if base_path:
#         base_model = HydraBase()
#         base_model.load_weights(config_new.tl_hydra_base_weights_save)
#
#         if config_new.tl_freeze_base is True:
#             base_model.trainable = False
#         print('--> TRAINABLE BASE:', base_model.trainable)
#
#         return base_model
#     else:
#         return HydraBase()




# transfer_model(head, checkpoint_path)
# exit(0)
#
# hydra_base = HydraBase()
# hydra_head = get_head(head)
# hydra = Hydra(hydra_base, hydra_head)
# checkpoint = tf.train.Checkpoint(hydra)
# checkpoint.restore(transfer_model).expect_partial()
# implicit_build(hydra)
# hydra.hydra_base.trainable = not config_new.tl_freeze_base
# hydra.summary()
#
#
# # # Create new model
# # extracted_layers = hydra.layers[:-1]
# # extracted_layers.append(get_head("move-prediction"))
# # new_model = keras.Sequential(extracted_layers)
# # implicit_build(new_model)
#
#
# # new_model.summary()
# # hydra.summary()
# # exit(0)
#
# return hydra, hydra_base, hydra_head







