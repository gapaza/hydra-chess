import argparse
import config
import platform
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


from hydra import HydraEncoderModel
from hydra import HydraDecoderModel


from hydra.callbacks.ValidationCallback import ValidationCallback
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator
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
    if 'pt' in config.model_mode:
        dataset_generator = PT_DatasetGenerator(
            config.pt_megaset_pt3_dataset_64_30p_int16
            # config.pt_millionsbase_pt3_dataset_large_64_30p
        )
        epochs = config.pt_epochs
    elif 'ft' in config.model_mode:
        dataset_generator = DC_DatasetGenerator(
            config.ft_lc0_standard_large_128_mask_dir
        )
        epochs = config.ft_epochs
    train_dataset, val_dataset = dataset_generator.load_datasets()
    return train_dataset, val_dataset, epochs


def get_optimizer():

    # 1. Set Learning Rate
    learning_rate = None
    if 'pt' in config.model_mode:
        # learning_rate = LinearWarmupCosineDecay(
        #     warmup_steps=1000.,
        #     decay_steps=100000.,
        #     target_warmup=0.0005,
        #     target_decay=0.00005
        # )
        # learning_rate = LinearWarmup(
        #     warmup_steps=1000.,
        #     target_warmup=0.0005
        # )
        learning_rate = 0.0003
        # 0.001 2000 .0408
    elif 'ft' in config.model_mode:
        learning_rate = LinearWarmup(
            warmup_steps=400.,
            target_warmup=0.0001
        )
        # learning_rate = LinearWarmupCosineDecay(
        #     warmup_steps=400.,
        #     decay_steps=100000.,
        #     target_warmup=0.0001,
        #     target_decay=0.00005
        # )
        # decoder ndcg | 1000 0.0001: 0.0556 0.1586 --> 0.0556 0.1586
        # decoder ndcg | 2000 0.0002:


        # learning_rate = LinearWarmupCosineDecay(
        #     warmup_steps=400.,
        #     decay_steps=9000.,
        #     target_warmup=0.0001,
        #     target_decay=0.00001
        # )
        # decoder ndcg | 1000 10000 0.0001 0.00001: 0.0773 0.1586 --> 0.0773 0.1586 --> 0.0773 0.1586

    # 2. Create Optimizer
    if platform.system() != 'Darwin':
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
        # optimizer = tfa.optimizers.AdaBelief(learning_rate=learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        jit_compile = True
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        jit_compile = False
    return optimizer, jit_compile


def get_checkpoints():
    checkpoints = []

    # Save Checkpoint
    checkpoint = ModelCheckpoint(config.model_save_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoints.append(checkpoint)

    # Validation Checkpoint
    # train_dataset, val_dataset, epochs = get_dataset()
    # checkpoint = ValidationCallback(val_dataset, config.model_save_dir, save=True)
    # checkpoints.append(checkpoint)
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

def train():

    # 1. Build Model
    model = None
    if config.model_type == 'encoder':
        model = HydraEncoderModel.build_model(config.model_mode)
    elif config.model_type == 'decoder':
        model = HydraDecoderModel.build_model(config.model_mode)

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
    # train_dataset = train_dataset.take(2000)
    # val_dataset = val_dataset.take(100)

    # 6. Get Checkpoints
    checkpoints = get_checkpoints()

    # 7. Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=checkpoints
    )

    # 8. Plot History
    plot_history(history)



if __name__ == "__main__":
    train()
