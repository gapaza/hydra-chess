from hydra.Hydra import Hydra
from hydra.HydraBase import HydraBase
from hydra.EncoderBase import EncoderBase
from hydra.HydraInterface import HydraInterface

from hydra.heads.GameModeling import GameModeling
from hydra.heads.PositionModeling import PositionModeling

from hydra.heads.MovePrediction import MovePrediction
from hydra.heads.MoveRanking import MoveRanking
from hydra.checkpoints.SaveCheckpoint import SaveCheckpoint
from hydra.utils import plot_history
from hydra.schedulers.LinearWarmup import LinearWarmup
from hydra.schedulers.LinearWarmupCosineDecay import LinearWarmupCosineDecay

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
        implicit_build(hydra)

    # 1. Build model
    # implicit_build(hydra, summary=False)

    # 2. Visualize
    hydra.summary(expand_nested=True)
    model_img_file = os.path.join(config.plots_dir, config.model_name + '-' + config.model_mode + '.png')
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
    if config.tl_freeze_base:
        hydra_model.hydra_base.trainable = False
        hydra_model.hydra_base.decoder_4.trainable = True

    return hydra_model, hydra_base, hydra_head



def new_model(head):
    hydra_base = get_base()
    hydra_head = get_head(head)
    hydra_model = Hydra(hydra_base, hydra_head)
    return hydra_model, hydra_base, hydra_head


def get_base():
    if config.model_base == 'custom':
        return HydraBase()
    elif config.model_base == 'encoder':
        return EncoderBase()


def get_head(head):
    hydra_head = None
    if head == 'game-modeling':
        hydra_head = GameModeling()
    elif head == 'move-prediction':
        hydra_head = MovePrediction()
    elif head == 'move-ranking':
        hydra_head = MoveRanking()
    elif head == 'position-modeling':
        hydra_head = PositionModeling()
    return hydra_head


def implicit_build(model, summary=False):
    board_input = tf.zeros((1, 65))
    move_input = tf.zeros((1, 128))
    model([board_input, move_input])
    if summary is True:
        model.summary()











