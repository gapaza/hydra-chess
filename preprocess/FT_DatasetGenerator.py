import config
import chess.pgn
from tqdm import tqdm
import os
import re
import json
import ast
import concurrent.futures
import numpy as np
import pickle
import itertools
import tensorflow as tf
import zipfile
import random

from tqdm import tqdm
from preprocess.strategies.move_ranking import move_ranking, move_ranking_batch, encode_batch




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
            for line in tqdm(file_lines[1:self.max_positions]):
                matches = re.findall(r'\[.*?\]', line)
                if len(matches) != 2:
                    print('Error parsing line:', line)
                    continue
                moves = json.loads(matches[0])
                prev_moves = ast.literal_eval(matches[1])
                try:
                    cp_scores = []
                    for move in moves:
                        eval_str = move['eval']
                        if 'Cp(' in eval_str:
                            cp_score = int(re.search(r"Cp\((.+?)\)", eval_str).group(1))
                        elif 'Mate(' in eval_str:
                            mate_score = int(re.search(r"Mate\((.+?)\)", eval_str).group(1))
                            # Assign a large score for checkmate evaluations.
                            cp_score = 10000 if mate_score > 0 else -10000
                        cp_scores.append(cp_score)
                except Exception as e:
                    print('Error parsing line:', line, e)
                    continue
                if 'WHITE' in moves[0]['eval']:
                    best_score = max(cp_scores)
                else:
                    best_score = min(cp_scores)

                # Compute absolute differences.
                abs_diff_scores = [abs(score - best_score) for score in cp_scores]

                # Normalize by dividing by the maximum absolute difference.
                max_abs_diff = max(abs_diff_scores)
                norm_scores = [score / max_abs_diff if max_abs_diff != 0 else 0. for score in abs_diff_scores]

                # Invert the scores so that a higher score is better.
                norm_scores_inv = [round(1 - score, 3) for score in norm_scores]

                # sort moves wand norm_score together on norm_score with zip
                # then unzip them into two lists
                moves, norm_scores_inv_sorted = zip(*sorted(zip(moves, norm_scores_inv), key=lambda x: x[1], reverse=True))
                uci_moves = [move['move'] for move in moves]

                # --> Outputs:
                # 1. norm_scores: list of normalized evaluation scores for current position
                # 2. uci_moves: list of candidate uci moves for current position
                # 3. prev_moves: list of previous uci moves leading up to current position
                eval_data.append({
                    'norm_scores': norm_scores_inv_sorted,
                    'uci_moves': uci_moves,
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
        print(len(eval_data))
        exit(0)
        # 3. Split files with 90% train and 10% validation
        split_idx = int(len(eval_data) * 0.9)
        train_positions, val_positions = eval_data[:split_idx], eval_data[split_idx:]

        # 4. Parse datasets
        print('Training Positions:', len(train_positions))
        train_dataset = self.parse_dataset(train_positions)
        print('Validation Positions:', len(val_positions))
        val_dataset = self.parse_dataset(val_positions)

        # 5. Save datasets
        if save:
            self.save_datasets(train_dataset, val_dataset)
        return train_dataset, val_dataset

    def parse_dataset(self, positions):
        dataset = self.create_and_pad_dataset(positions)
        dataset = dataset.batch(config.ft_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(3125)
        return dataset.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def create_and_pad_dataset(positions):
        all_candidate_scores = []
        all_previous_moves = []
        for position in positions:
            candidate_scores = [0.] * config.vocab_size
            for idx, candidate_move in enumerate(position['uci_moves']):
                score = position['norm_scores'][idx]
                if score == 0:
                    score = 0.1
                candidate_scores[config.vocab.index(candidate_move)] = score
            all_candidate_scores.append(candidate_scores)
            all_previous_moves.append(position['prev_moves'])
        return tf.data.Dataset.from_tensor_slices((all_candidate_scores, all_previous_moves))



    ###################
    ### Load / Save ###
    ###################

    def save_datasets(self, train_dataset, val_dataset):
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)
        with zipfile.ZipFile(self.archive_file, 'w') as zip_obj:
            zip_obj.write(self.train_dataset_dir)
            zip_obj.write(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset




if __name__ == '__main__':
    generator = FT_DatasetGenerator(config.ft_lc0_standard_dir)
    generator.get_datasets(save=True)











