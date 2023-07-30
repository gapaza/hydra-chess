import config
import platform
import tensorflow as tf
import tensorflow_addons as tfa

import hydra
from hydra import SaveCheckpoint

from preprocess.generators.PT_Eval_DatasetGenerator import PT_Eval_DatasetGenerator
from preprocess.generators.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.generators.EvaluationsDatasetGenerator import EvaluationsDatasetGenerator


#
#   _______           _         _
#  |__   __|         (_)       (_)
#     | | _ __  __ _  _  _ __   _  _ __    __ _
#     | || '__|/ _` || || '_ \ | || '_ \  / _` |
#     | || |  | (_| || || | | || || | | || (_| |
#     |_||_|   \__,_||_||_| |_||_||_| |_| \__, |
#                                          __/ |
#                                         |___/
#

def train():


    # 1. Build Model
    model, model_base, model_head = hydra.get_model(
        config.model_mode,
        checkpoint_path=config.tl_full_model_path,
    )


    # 2. Get Optimizer
    optimizer, jit_compile = get_optimizer()


    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)


    # 4. Get Datasets
    train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps = get_dataset()


    # 5. Get Checkpoints
    checkpoints = get_checkpoints()


    # 6. Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=checkpoints
    )


    # 7. Plot History
    hydra.plot_history(history)







#
#   _    _        _
#  | |  | |      | |
#  | |__| |  ___ | | _ __    ___  _ __  ___
#  |  __  | / _ \| || '_ \  / _ \| '__|/ __|
#  | |  | ||  __/| || |_) ||  __/| |   \__ \
#  |_|  |_| \___||_|| .__/  \___||_|   |___/
#                   | |
#                   |_|
#

def get_dataset():
    dataset_generator, epochs = None, None
    train_dataset, val_dataset = None, None
    steps_per_epoch, validation_steps = None, None
    if config.train_mode == 'pt':
        dataset_generator = PT_Eval_DatasetGenerator(config.pt_dataset)
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=config.pt_train_buffer,
            val_buffer=config.pt_val_buffer
        )
        epochs = config.pt_epochs
        steps_per_epoch = config.pt_steps_per_epoch
        validation_steps = config.pt_val_steps
    elif config.train_mode == 'ft':
        # dataset_generator = DC_DatasetGenerator(config.ft_dataset)
        dataset_generator = EvaluationsDatasetGenerator(config.ft_dataset)
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=config.ft_train_buffer,
            val_buffer=config.ft_train_buffer
        )
        epochs = config.ft_epochs
        steps_per_epoch = config.ft_steps_per_epoch
        validation_steps = config.ft_val_steps
    print('Datasets Fetched...')

    # --> Distributed Training
    if config.distributed is True:
        train_dataset = train_dataset.repeat(epochs)
        val_dataset = val_dataset.repeat(epochs)
        train_dataset = config.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        val_dataset = config.mirrored_strategy.experimental_distribute_dataset(val_dataset)
        print('-- Distributed Training Enabled --')

    return train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps


def get_optimizer():

    # 1. Set Learning Rate
    learning_rate = None
    if config.train_mode == 'pt':
        learning_rate = config.pt_learning_rate

    elif config.train_mode == 'ft':
        # learning_rate = config.ft_learning_rate
        learning_rate = hydra.LinearWarmup(target_warmup=0.0005, warmup_steps=1000)

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

    model_checkpoint = SaveCheckpoint(config.tl_hydra_full_save, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoints.append(model_checkpoint)

    return checkpoints


















def train_distributed():
    with config.mirrored_strategy.scope():
        train()


if __name__ == "__main__":
    if config.distributed is False:
        train()
    else:
        train_distributed()




