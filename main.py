import argparse
import time
import config
import platform
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from hydra.checkpoints import PlotCallback

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

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()



def main():
    args = parse_arguments()
    config.mode = args.mode

    # Model
    model = build_model(args.mode)

    # Load Weights
    if config.tl_enabled is True:
        model.load_weights(config.tl_load_weights)

    # Learning Rate
    if args.mode == 'pt':
        learning_rate = 0.001
    elif args.mode == 'ft':
        learning_rate = 0.0001
    else:
        learning_rate = 0.001

    # Optimizer
    if platform.system() != 'Darwin':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        jit_compile = True
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        jit_compile = False

    # Compile
    model.compile(optimizer=optimizer, jit_compile=jit_compile)
    if args.mode == 'pt':
        print('Pretraining...')
        pretrain(model)
    elif args.mode == 'pt2':
        print('Pretraining...')
        pretrain2(model)
    elif args.mode == 'ft':
        print('Fine-Tuning...')
        fine_tune(model)

    # Save Weights
    if not os.path.exists(config.tl_write_dir):
        os.makedirs(config.tl_write_dir)
    model.save_weights(config.tl_write_path)


def pretrain(model):
    # Load datasets
    dataset_generator = PT_DatasetGenerator(config.pt_millionsbase_chesscom_dataset)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-pt'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback(model_name)
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset, callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


def pretrain2(model):
    # Load datasets
    dataset_generator = PT_DatasetGenerator(config.pt_millionsbase_pt2_dataset)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-pt2'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback(model_name)
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset,
                        callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)



def fine_tune(model):
    # Load datasets
    dataset_generator = FT_DatasetGenerator(config.ft_lc0_standard_200k_legal_dir)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-ft'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback(model_name)
    history = model.fit(training_dataset, epochs=config.ft_epochs, validation_data=validation_dataset, callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


if __name__ == "__main__":
    main()
