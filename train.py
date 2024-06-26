import argparse
import config
import platform
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


from hydra import HydraEncoderModel
from hydra import HydraDecoderModel
from hydra import HydraHybridModel

from hydra.callbacks.ValidationCallback import ValidationCallback
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.PT_Eval_DatasetGenerator import PT_Eval_DatasetGenerator
import tensorflow_addons as tfa

from hydra.schedulers.PretrainingScheduler import PretrainingScheduler
from hydra.schedulers.FinetuningScheduler import FinetuningScheduler
from hydra.schedulers.LinearWarmupCosineDecay import LinearWarmupCosineDecay
from hydra.schedulers.LinearWarmup import LinearWarmup


#
#   _______           _         _                 _    _        _
#  |__   __|         (_)       (_)               | |  | |      | |
#     | | _ __  __ _  _  _ __   _  _ __    __ _  | |__| |  ___ | | _ __    ___  _ __  ___
#     | || '__|/ _` || || '_ \\| || '_ \\ / _` | |  __  | / _\\| || '_ \  / _ \| '__|/ __|
#     | || |  | (_| || || | | || || | | || (_| | | |  | ||  __/| || |_) ||  __/| |   \__ \
#     |_||_|   \__,_||_||_| |_||_||_| |_| \__, | |_|  |_| \___||_|| .__/  \___||_|   |___/
#                                          __/ |                  | |
#                                         |___/                   |_|
#


def get_dataset():
    dataset_generator, epochs = None, None
    train_dataset, val_dataset = None, None
    if 'pt' in config.model_mode:
        if 'eval' in config.model_mode:
            dataset_generator = PT_Eval_DatasetGenerator(
                config.pt_mixed_eval_4mil
            )
        else:
            dataset_generator = PT_DatasetGenerator(
                config.pt_megaset_bw
            )
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=2048 * 100,
            val_buffer=256
        )
        epochs = config.pt_epochs
    elif 'ft' in config.model_mode:
        dataset_generator = DC_DatasetGenerator(
            # config.ft_lc0_standard_large_128_mask_dir
            config.ft_lichess_tactics
            # config.ft_lichess_mates
        )
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=2048 * 10,
            val_buffer=256
        )
        epochs = config.ft_epochs
    print('Datasets Fetched...')
    return train_dataset, val_dataset, epochs


def get_optimizer():

    # 1. Set Learning Rate
    learning_rate = None
    if 'pt' in config.model_mode:
        learning_rate = 0.0008
    elif 'ft' in config.model_mode:
        learning_rate = 0.00008

    # 2. Create Optimizer
    if platform.system() != 'Darwin':
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        # optimizer = tfa.optimizers.AdaBelief(learning_rate=learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        jit_compile = True
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        jit_compile = False

    if config.distributed is True:
        jit_compile = False

    return optimizer, jit_compile


def get_checkpoints():
    checkpoints = []

    # Save Checkpoint
    checkpoint = ModelCheckpoint(config.model_save_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoints.append(checkpoint)

    return checkpoints


def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#
#   _______           _
#  |__   __|         (_)
#     | | _ __  __ _  _  _ __
#     | || '__|/ _` || || '_ \
#     | || |  | (_| || || | | |
#     |_||_|   \__,_||_||_| |_|
#


def train_distributed():
    with config.mirrored_strategy.scope():
        train()

def train():

    # 1. Build Model
    model = None
    if config.model_type == 'encoder':
        model = HydraEncoderModel.build_model()
    elif config.model_type == 'decoder':
        model = HydraDecoderModel.build_model()
    elif config.model_type == 'hybrid':
        model = HydraHybridModel.build_model()

    # 2. Load Weights
    if config.tl_enabled is True:
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore(config.tl_load_checkpoint).expect_partial()

    # 3. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 4. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 5. Get Datasets
    train_dataset, val_dataset, epochs = get_dataset()

    steps_per_epoch = None
    validation_steps = None
    if config.distributed is True:
        train_dataset = train_dataset.repeat(epochs)
        val_dataset = val_dataset.repeat(epochs)
        train_dataset = config.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = config.mirrored_strategy.experimental_distribute_dataset(val_dataset)
        print('-- Distributed Training Enabled --')



    if 'pt' in config.model_mode:
        steps_per_epoch = 60000  # 12500, 25000 | 58000 (128 batch)
        validation_steps = 3000  # 750, 1500 | 3000 (128 batch)
    elif 'ft' in config.model_mode:
        steps_per_epoch = 1000  # 1.8mil / 128 = 14062.5
        validation_steps = 100  # 1000


    # 6. Get Checkpoints
    checkpoints = get_checkpoints()

    # 7. Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=checkpoints
    )

    # 8. Plot History
    plot_history(history)



if __name__ == "__main__":
    if config.distributed is False:
        train()
    else:
        train_distributed()
