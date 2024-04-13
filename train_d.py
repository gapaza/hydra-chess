import config
import platform
import tensorflow as tf
import tensorflow_addons as tfa

import hydra
from hydra import SaveCheckpoint

from preprocess.generators.PT_Eval_DatasetGenerator import PT_Eval_DatasetGenerator
from preprocess.generators.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.generators.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.generators.EvaluationsDatasetGenerator import EvaluationsDatasetGenerator
from preprocess.generators.DecoderOnly_DG import DecoderOnly_DG

from preprocess.utils import rebatch_dataset


curr_dataset = config.pt_baseline_short

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
    checkpoint_path = config.tl_decoder_save
    model = hydra.decoder_only_model(checkpoint_path=None)

    # 2. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 4. Get Datasets
    train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps = get_dataset()
    # train_dataset = train_dataset.take(1000)

    # 5. Get Checkpoints
    checkpoints = get_checkpoints()


    # 6. Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        # steps_per_epoch=steps_per_epoch,
        # validation_steps=validation_steps,
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
        # dataset_generator = PT_Eval_DatasetGenerator(config.pt_dataset)
        # dataset_generator = PT_DatasetGenerator(config.pt_baseline)
        dataset_generator = DecoderOnly_DG(curr_dataset)
        # train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
        #     train_buffer=config.pt_train_buffer,
        #     val_buffer=config.pt_val_buffer
        # )
        train_dataset, val_dataset = dataset_generator.load_datasets()

        # # call position_modeling python function
        # train_dataset = dataset_generator.map_positions(train_dataset, save_path='train_dataset_pos')
        # val_dataset = dataset_generator.map_positions(val_dataset, save_path='val_dataset_pos')
        # exit(0)






        epochs = config.pt_epochs
        steps_per_epoch = config.pt_steps_per_epoch
        validation_steps = config.pt_val_steps
    elif config.train_mode == 'ft':
        # dataset_generator = DC_DatasetGenerator(config.ft_dataset)
        dataset_generator = EvaluationsDatasetGenerator(config.ft_dataset)
        # train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
        #     train_buffer=config.ft_train_buffer,
        #     val_buffer=config.ft_train_buffer
        # )
        train_dataset, val_dataset = dataset_generator.load_datasets()
        epochs = config.ft_epochs
        steps_per_epoch = config.ft_steps_per_epoch
        validation_steps = config.ft_val_steps
    print('Datasets Fetched...')


    return train_dataset, val_dataset, epochs, steps_per_epoch, validation_steps


def get_optimizer():
    jit_compile = False
    learning_rate = 0.001
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(
        0.0,
        10000,
        alpha=0.1,
        warmup_target=learning_rate,
        warmup_steps=500
    )
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    if config.mixed_precision is True:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer, jit_compile



def get_checkpoints():
    checkpoints = []
    model_checkpoint = SaveCheckpoint(config.tl_hydra_full_save, monitor='loss', verbose=1, save_best_only=False, mode='min', save_weights_only=True)
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




