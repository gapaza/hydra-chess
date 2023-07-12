import config
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
import tensorflow as tf
import os



def cast_to_int32(*features):
    return tuple(tf.cast(tensor, tf.int32) for tensor in features)

def cast_to_int16(*features):
    return tuple(tf.cast(tensor, tf.int16) for tensor in features)

def cast_dataset():
    generator = PT_DatasetGenerator(config.pt_millionsbase_pt3_dataset_large_64_30p)
    train_dataset, val_dataset = generator.load_datasets()


    # 1. Cast Datasets
    print('Casting datasets...')
    train_dataset = train_dataset.map(cast_to_int16)
    val_dataset = val_dataset.map(cast_to_int16)

    # # 1.1 Rebatch if necessary
    # train_dataset = train_dataset.unbatch().batch(128)
    # val_dataset = val_dataset.unbatch().batch(128)

    print('Shuffling datasets...')
    train_dataset = train_dataset.shuffle(28000)
    # val_dataset = val_dataset.shuffle(3000)

    # 2. Save Datasets
    print('Saving datasets...')
    train_save = os.path.join(config.pt_millionsbase_pt3_dataset_large_64_30p_int16, 'train_dataset')
    val_save = os.path.join(config.pt_millionsbase_pt3_dataset_large_64_30p_int16, 'val_dataset')
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
    generator = FT_DatasetGenerator(config.ft_lc0_standard_large_ft2_64)
    train_dataset, val_dataset = generator.load_datasets()

    # 1. Cast Datasets
    print('Casting datasets...')
    train_dataset = train_dataset.map(cast_ft)
    val_dataset = val_dataset.map(cast_ft)

    print('Shuffling datasets...')
    train_dataset = train_dataset.shuffle(28000)
    val_dataset = val_dataset.shuffle(3000)

    # 2. Save Datasets
    print('Saving datasets...')
    train_save = os.path.join(config.ft_lc0_standard_large_ft2_64_int16, 'train_dataset')
    val_save = os.path.join(config.ft_lc0_standard_large_ft2_64_int16, 'val_dataset')
    train_dataset.save(train_save)
    val_dataset.save(val_save)


if __name__ == '__main__':
    cast_dataset_ft()
