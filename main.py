import argparse
from keras import layers
import config
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from hydra.checkpoints import PlotCallback

from hydra.HydraModel import HydraModel
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

    # Create model
    model = HydraModel(mode=args.mode, name="hydra")
    # optimizer = tf.keras.optimizers.Adam()
    optimizer = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=optimizer, jit_compile=False)

    if args.mode == 'pt':
        print('Pretraining...')
        pretrain(model)
    elif args.mode == 'ft':
        print('Fine-Tuning...')
        fine_tune(model)

    # model.summary(expand_nested=True)
    # model_img_file = os.path.join(config.models_dir, config.model_name + '-' + args.mode + '.png')
    # plot_model(model, to_file=model_img_file, show_shapes=True, show_layer_names=True, expand_nested=False)



def pretrain(model):
    # Load datasets
    dataset_generator = PT_DatasetGenerator(config.pt_millionsbase_dataset)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-pt'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback(model_name)
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset, callbacks=[checkpoint, plot_checkpoint])

    # --> Plot Training History
    plot_history(history)


def fine_tune(model):
    # Load datasets
    dataset_generator = FT_DatasetGenerator(config.pt_lc0_standard_dir)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-ft'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    plot_checkpoint = PlotCallback(model_name)
    history = model.fit(training_dataset, epochs=config.ft_epochs, validation_data=validation_dataset, callbacks=[checkpoint, plot_checkpoint])

    # --> Plot Training History
    plot_history(history)


if __name__ == "__main__":
    main()
