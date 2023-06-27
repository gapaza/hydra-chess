import config
import os
import re
import json
import ast
import pickle
import tensorflow as tf
import zipfile
import random

from tqdm import tqdm
from preprocess import utils
from preprocess.strategies.move_ranking import move_ranking_batch, encode_batch




class FT_DatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'bulk_games.txt')
        self.intermediate_file = os.path.join(self.dataset_dir, 'bulk_games.pkl')
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')
        self.archive_file = os.path.join(self.dataset_dir, self.dataset_name + '.zip')
        self.max_positions = 2000000


    ################################
    ### 1. Gen Intermediate File ###
    ################################

    def parse_bulk_games(self):
        eval_data = []
        with open(self.game_file, 'r') as f:
            file_lines = f.readlines()
            for line in tqdm(file_lines[1:self.max_positions+1]):

                # 1. Parse line
                prev_moves, moves, cp_scores, best_score = utils.parse_ft_line(line)
                if prev_moves is None:
                    continue

                # 2. Add mask token to end prev_moves
                # prev_moves.append("[mask]")

                # 3. Compute absolute differences, find max difference
                abs_diff_scores = [abs(score - best_score) for score in cp_scores]
                max_abs_diff = max(abs_diff_scores)

                # 4. Normalize absolute differences by max difference
                move_scores = [score / max_abs_diff if max_abs_diff != 0 else 0. for score in abs_diff_scores]

                # 5. Invert the scores so that a higher score is better.
                move_scores = [round(1 - score, 3) for score in move_scores]

                # 6. Replace 0.0 with 0.1, as 0.0 is reserved for non-evaluated moves
                move_scores = [x if x != 0. else 0.1 for x in move_scores]

                # 7. Sort moves wand norm_score together on norm_score with zip
                moves, move_scores = zip(*sorted(zip(moves, move_scores), key=lambda x: x[1], reverse=True))
                move_scores = list(move_scores)
                uci_moves = [move['move'] for move in moves]
                uci_moves_idx = [config.vocab.index(move) for move in uci_moves]

                # 8. Ensure there are at least top n moves
                while len(uci_moves) < 3:
                    padding_move = ''
                    uci_moves.append(padding_move)
                    uci_moves_idx.append(config.vocab.index(padding_move))
                    move_scores.append(0.0)

                # 9. Add to eval_data
                eval_data.append({
                    'candidate_scores': move_scores,
                    'candidate_moves': uci_moves,
                    'candidate_moves_idx': uci_moves_idx,
                    'prev_moves': ' '.join(prev_moves)
                })

        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(eval_data, f)

        return eval_data



    ##########################
    ### 2. Procure Dataset ###
    ##########################

    def get_datasets(self, save=False):
        if not os.path.exists(self.intermediate_file):
            print('Intermediate file not found... generating')
            self.parse_bulk_games()

        with open(self.intermediate_file, 'rb') as f:
            eval_data = pickle.load(f)
        random.shuffle(eval_data)

        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(eval_data) * 0.9)
        train_positions, val_positions = eval_data[:split_idx], eval_data[split_idx:]

        # 4. Parse datasets
        print('Training Positions:', len(train_positions))
        train_dataset = self.parse_dataset(train_positions)
        print('Validation Positions:', len(val_positions))
        val_dataset = self.parse_dataset(val_positions)

        # 5. Save datasets
        if save is True:
            self.save_datasets(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def parse_dataset(self, positions):
        dataset = self.create_and_pad_dataset(positions)
        dataset = dataset.batch(config.ft_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def create_and_pad_dataset(positions):
        all_previous_moves = []
        all_candidate_moves_idx = []
        all_candidate_scores = []
        for position in positions:
            all_previous_moves.append(position['prev_moves'])
            all_candidate_moves_idx.append(position['candidate_moves_idx'])
            all_candidate_scores.append(position['candidate_scores'])
        return tf.data.Dataset.from_tensor_slices((all_candidate_scores, all_previous_moves, all_candidate_moves_idx))

    ###################
    ### Load / Save ###
    ###################

    def save_datasets(self, train_dataset, val_dataset):
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)
        # with zipfile.ZipFile(self.archive_file, 'w') as zip_obj:
        #     zip_obj.write(self.train_dataset_dir)
        #     zip_obj.write(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset




if __name__ == '__main__':
    generator = FT_DatasetGenerator(config.ft_lc0_standard_dir)
    # generator.parse_bulk_games()
    generator.get_datasets(save=True)











