import config
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
import tensorflow as tf
import os



def cast_to_int32(*features):
    return tuple(tf.cast(tensor, tf.int32) for tensor in features)

def cast_to_int16(*features):
    return tuple(tf.cast(tensor, tf.int16) for tensor in features)

def cast_dataset():
    generator = PT_DatasetGenerator(config.pt_megaset_denoising_64)
    train_dataset, val_dataset = generator.load_datasets()


    # 1. Cast Datasets
    # print('Casting datasets...')
    # train_dataset = train_dataset.map(cast_to_int16)
    # val_dataset = val_dataset.map(cast_to_int16)

    # # 1.1 Rebatch if necessary
    train_dataset = train_dataset.unbatch().batch(256)
    val_dataset = val_dataset.unbatch().batch(256)

    print('Shuffling datasets...')
    # train_dataset = train_dataset.shuffle(28000)
    # val_dataset = val_dataset.shuffle(3000)

    # 2. Save Datasets
    print('Saving datasets...')
    train_save = os.path.join(config.pt_megaset_denoising_256, 'train_dataset')
    val_save = os.path.join(config.pt_megaset_denoising_256, 'val_dataset')
    train_dataset.save(train_save)
    val_dataset.save(val_save)


    return 0








def cast_ft(position, relevancy_scores, board, move_weights):
    return (
        tf.cast(position, tf.int16),
        tf.cast(relevancy_scores, tf.float32),
        tf.cast(board, tf.int16),
        tf.cast(move_weights, tf.float16)
    )

def cast_dataset_ft():
    generator = DC_DatasetGenerator(config.ft_lc0_standard_large_128_mask_dir)
    train_dataset, val_dataset = generator.load_datasets()

    # 1. Cast Datasets
    print('Casting datasets...')
    # train_dataset = train_dataset.map(cast_ft)
    # val_dataset = val_dataset.map(cast_ft)

    # # 1.1 Rebatch if necessary
    train_dataset = train_dataset.unbatch().batch(config.global_batch_size)
    val_dataset = val_dataset.unbatch().batch(config.global_batch_size)

    print('Shuffling datasets...')
    # train_dataset = train_dataset.shuffle(7000)
    # val_dataset = val_dataset.shuffle(800)

    # 2. Save Datasets
    print('Saving datasets...')
    train_save = os.path.join(config.ft_lc0_standard_large_256_mask_dir, 'train_dataset')
    val_save = os.path.join(config.ft_lc0_standard_large_256_mask_dir, 'val_dataset')
    train_dataset.save(train_save)
    val_dataset.save(val_save)







if __name__ == '__main__':
    cast_dataset()
    # cast_dataset_ft()
