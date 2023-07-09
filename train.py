import argparse
import config
import platform
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from hydra.HydraModel import build_model

from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Hydra')

    parser.add_argument('--mode', type=str, help='Mode to run the script in')

    # Parse the arguments
    args = parser.parse_args()

    # If mode is not given or is not one of the valid modes, print an error and exit
    if args.mode not in ['pt', 'ft']:
        parser.error('Invalid mode! Mode should be "pt" or "ft".')

    return args


# """
#   _______           _         _                 _    _        _
#  |__   __|         (_)       (_)               | |  | |      | |
#     | | _ __  __ _  _  _ __   _  _ __    __ _  | |__| |  ___ | | _ __    ___  _ __  ___
#     | || '__|/ _` || || '_ \\| || '_ \\ / _` | |  __  | / _\\| || '_ \  / _ \| '__|/ __|
#     | || |  | (_| || || | | || || | | || (_| | | |  | ||  __/| || |_) ||  __/| |   \__ \
#     |_||_|   \__,_||_||_| |_||_||_| |_| \__, | |_|  |_| \___||_|| .__/  \___||_|   |___/
#                                          __/ |                  | |
#                                         |___/                   |_|
# """


def get_dataset():
    dataset_generator, epochs = None, None
    if config.mode == 'pt':
        dataset_generator = PT_DatasetGenerator(
            config.pt_millionsbase_pt3_dataset_med_64_30p
        )
        epochs = config.pt_epochs
    elif config.mode == 'ft':
        dataset_generator = FT_DatasetGenerator(
            config.ft_lc0_standard_large_ft2_64
        )
        epochs = config.ft_epochs
    train_dataset, val_dataset = dataset_generator.load_datasets()
    return train_dataset, val_dataset, epochs


def get_optimizer():

    # 1. Set Learning Rate
    learning_rate = None
    if config.mode == 'pt':
        learning_rate = 0.001
    elif config.mode == 'ft':
        learning_rate = 0.0001

    # 2. Create Optimizer
    if platform.system() != 'Darwin':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        jit_compile = True
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        jit_compile = False
    return optimizer, jit_compile


def get_checkpoints():
    model_name = config.model_name + '-' + config.mode
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return [checkpoint]

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# """
#   _______           _
#  |__   __|         (_)
#     | | _ __  __ _  _  _ __
#     | || '__|/ _` || || '_ \
#     | || |  | (_| || || | | |
#     |_||_|   \__,_||_||_| |_|
# """

def train():
    args = parse_arguments()
    config.mode = args.mode

    # 1. Build Model
    model = build_model(args.mode)

    # 2. Load Weights
    if config.tl_enabled is True:
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore('/home/ubuntu/hydra-chess/models/hydra-pt3-backup').expect_partial()

    # 3. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 4. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 5. Get Datasets
    train_dataset, val_dataset, epochs = get_dataset()

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