import config
import os
import re
import json
import ast
import pickle
import tensorflow as tf
import zipfile
import random
import chess

from tqdm import tqdm
from preprocess import utils
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.strategies.move_ranking import move_ranking_batch, encode_batch
from preprocess.strategies.move_ranking_flat import move_ranking_batch_flat






class TacticDatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'bulk_games.txt')
        self.intermediate_file = os.path.join(self.dataset_dir, 'bulk_games.pkl')
        self.intermediate_file_tactics = os.path.join(self.dataset_dir, 'bulk_games_tactics.pkl')
        self.intermediate_file_mates = os.path.join(self.dataset_dir, 'bulk_games_mates.pkl')
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')
        self.archive_file = os.path.join(self.dataset_dir, self.dataset_name + '.zip')
        self.max_positions = 2000000
        self.unique_prev_moves = set()

    ################################
    ### 1. Gen Intermediate File ###
    ################################


    def parse_bulk_games(self):
        unique_short_evals = set()

        eval_data = []
        with open(self.game_file, 'r') as f:
            file_lines = f.readlines()
            for line in tqdm(file_lines[1:self.max_positions+1]):

                # 1. Parse line
                prev_moves, moves, cp_scores, best_score = utils.parse_ft_filter(line, mates=True, tactics=False)

                # 2. Parse Datapoint
                mask = True
                result = DC_DatasetGenerator.preprocess_datapoint(prev_moves, moves, cp_scores, best_score, unique_short_evals,
                                                   mask=mask)
                if result is None:
                    continue
                else:
                    eval_data.append(result)
        print('Number of evals:', len(eval_data))
        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(eval_data, f)

        return eval_data

    ##########################
    ### 2. Procure Dataset ###
    ##########################

    def get_datasets(self, save=False, small=False, mates=False, tactics=False):
        if mates is True:
            self.intermediate_file = self.intermediate_file_mates
        elif tactics is True:
            self.intermediate_file = self.intermediate_file_tactics


        if not os.path.exists(self.intermediate_file):
            print('Intermediate file not found... generating')
            self.parse_bulk_games()

        with open(self.intermediate_file, 'rb') as f:
            eval_data = pickle.load(f)
        random.shuffle(eval_data)

        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(eval_data) * 0.9)
        train_positions, val_positions = eval_data[:split_idx], eval_data[split_idx:]
        if small is True:
            train_positions, val_positions = train_positions[:100000], val_positions[:10000]

        # 4. Parse datasets
        print('Training Positions:', len(train_positions))
        train_dataset = self.parse_dataset(train_positions, buffer=50000)
        print('Validation Positions:', len(val_positions))
        val_dataset = self.parse_dataset(val_positions, buffer=50)

        # 5. Save datasets
        if save is True:
            self.save_datasets(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def parse_dataset(self, positions, buffer=1000):
        dataset = DC_DatasetGenerator.create_and_pad_dataset(positions, self.unique_prev_moves)
        return dataset

    ###################
    ### Load / Save ###
    ###################

    def save_datasets(self, train_dataset, val_dataset):
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset

    def load_val_dataset(self):
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return val_dataset

    def load_unsupervised_datasets(self, train_buffer=1024, val_buffer=256, batch_size=config.global_batch_size):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        train_dataset = train_dataset.shuffle(train_buffer)
        train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_ranking_batch_flat, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        val_dataset = val_dataset.shuffle(val_buffer)
        val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_ranking_batch_flat, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset







if __name__ == '__main__':
    generator = TacticDatasetGenerator(config.ft_lichess)
    # generator.parse_bulk_games()
    generator.get_datasets(save=True, small=False, mates=False, tactics=True)



