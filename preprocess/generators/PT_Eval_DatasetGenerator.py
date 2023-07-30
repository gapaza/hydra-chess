import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm
import pickle
import tensorflow_datasets as tfds
import random
import chess
import warnings

import threading
import time
import sys


from preprocess.strategies import game_modeling
from preprocess.strategies import denoising_objective







class PT_Eval_DatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'dataset_4mil.pkl')

        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')


    ##########################
    ### 3. Procure Dataset ###
    ##########################

    def get_dataset(self, save=False, small=False):


        with open(self.game_file, 'rb') as f:
            eval_data = pickle.load(f)
        # random.shuffle(eval_data)


        # dict_keys(['move_sequences', 'masking_indices', 'masking_segments', 'absolute_scores', 'probability_scores', 'white_turn'])
        move_sequences = eval_data['move_sequences']
        masking_indices = eval_data['masking_indices']
        probability_scores = eval_data['probability_scores']

        req_padding = 0
        for idx, sequence in enumerate(move_sequences):
            masking_index = masking_indices[idx]
            probability_score = probability_scores[idx]
            if -1 in masking_index:
                req_padding += 1
                for i in range(len(masking_index)):
                    if masking_index[i] == -1:
                        masking_index[i] = masking_index[0]
            if -1 in probability_score:
                for i in range(len(probability_score)):
                    if probability_score[i] == -1:
                        probability_score[i] = probability_score[0]
        print('Padding Required: ', req_padding)




        # Split into train and val
        split_idx = int(len(move_sequences) * 0.9)
        train_move_sequences, val_move_sequences = move_sequences[:split_idx], move_sequences[split_idx:]
        train_masking_indices, val_masking_indices = masking_indices[:split_idx], masking_indices[split_idx:]
        train_probability_scores, val_probability_scores = probability_scores[:split_idx], probability_scores[split_idx:]

        # Convert to tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_move_sequences, train_masking_indices, train_probability_scores))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_move_sequences, val_masking_indices, val_probability_scores))

        # Save
        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset

    ###################
    ### Load / Save ###
    ###################

    def save_datasets(self, train_dataset, val_dataset):
        print("Saving train dataset...")
        train_dataset.save(self.train_dataset_dir)
        print("Saving val dataset...")
        val_dataset.save(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset

    def load_unsupervised_datasets(self, train_buffer=1024, val_buffer=256, batch_size=config.global_batch_size):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        train_dataset = train_dataset.shuffle(train_buffer)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(game_modeling.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(game_modeling.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        val_dataset = val_dataset.shuffle(val_buffer)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.map(game_modeling.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(game_modeling.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def load_val_dataset(self):
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return val_dataset


if __name__ == '__main__':
    generator = PT_Eval_DatasetGenerator(config.pt_mixed_eval_4mil)
    # generator.chunk_pgn_file()
    # generator.parse_dir_games()
    generator.get_dataset(save=True, small=False)











