import argparse
import config
import platform
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from hydra.HydraModel import build_model_encoder
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Hydra')

    parser.add_argument('--mode', type=str, help='Mode to run the script in')

    # Parse the arguments
    args = parser.parse_args()

    # If mode is not given or is not one of the valid modes, print an error and exit
    if args.mode not in ['pt', 'pt2', 'pt3', 'ft', 'ft2']:
        parser.error('Invalid mode! Mode should be "pt", "pt2", "pt3", "ft", or "ft2".')

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
    model = build_model_encoder(args.mode)

    # Load Weights
    if config.tl_enabled is True:
        checkpoint = tf.train.Checkpoint(model)
        checkpoint.restore('/home/ubuntu/hydra-chess/models/hydra-pt3-backup').expect_partial()


    # Learning Rate
    if args.mode == 'pt':
        learning_rate = 0.001
    elif args.mode == 'ft':
        learning_rate = 0.0001
    elif args.mode == 'ft2':
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
        print('Multi-Task Pretraining...')
        pretrain2(model)
    elif args.mode == 'pt3':
        print('Multi-Task Pretraining...')
        pretrain3(model)
    elif args.mode == 'ft' or args.mode == 'ft2':
        print('Fine-Tuning...')
        fine_tune(model)

    # Save Weights
    # if not os.path.exists(config.tl_write_dir):
    #     os.makedirs(config.tl_write_dir)
    # model.save_weights(config.tl_write_path, save_format='h5')


def pretrain(model):
    # Load datasets
    dataset_generator = PT_DatasetGenerator(config.pt_millionsbase_chesscom_dataset)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # --> Train Model
    model_name = config.model_name + '-pt'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset, callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


def pretrain2(model):
    # Load datasets
    dataset_generator_1 = PT_DatasetGenerator(config.pt_millionsbase_pt2_dataset)
    training_dataset_1, validation_dataset_1 = dataset_generator_1.load_datasets()

    dataset_generator_2 = PT_DatasetGenerator(config.pt_chesscom_pt2_dataset)
    training_dataset_2, validation_dataset_2 = dataset_generator_2.load_datasets()

    print('Concatinating Datasets...')
    training_dataset = training_dataset_1.concatenate(training_dataset_2)
    validation_dataset = validation_dataset_1.concatenate(validation_dataset_2)

    # --> Train Model
    model_name = config.model_name + '-pt2'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset, callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


def pretrain3(model):
    # Load datasets
    dataset_generator = PT_DatasetGenerator(config.pt_millionsbase_pt3_dataset_med_64_30p)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # Limit val dataset
    # validation_dataset = validation_dataset.take(10000)


    # --> Train Model
    model_name = config.model_name + '-pt3'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(training_dataset, epochs=config.pt_epochs, validation_data=validation_dataset,
                        callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


def fine_tune(model):
    # Load datasets
    dataset_generator = FT_DatasetGenerator(config.ft_lc0_standard_small_ft2_64)
    training_dataset, validation_dataset = dataset_generator.load_datasets()

    # training_dataset = training_dataset.unbatch().batch(16)
    # validation_dataset = validation_dataset.unbatch().batch(16)
    # print('Unbatching finished')


    # --> Train Model
    model_name = config.model_name + '-ft2'
    model_file = os.path.join(config.models_dir, model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    history = model.fit(training_dataset, epochs=config.ft_epochs, validation_data=validation_dataset, callbacks=[checkpoint])

    # --> Plot Training History
    plot_history(history)


if __name__ == "__main__":
    main()
