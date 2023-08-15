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
from multiprocessing import Pool
from copy import deepcopy
from preprocess.strategies import move_prediction



def print_puzzle(puzzle):
    print('\n\nPuzzle:', puzzle['url'])
    for key, value in puzzle.items():
        if key == 'url':
            continue
        print(key, ':', value)

def handle_datapoint(puzzle):
    return PuzzlesDatasetGenerator.decompose_datapoint(puzzle)


class PuzzlesDatasetGenerator:
    def __init__(self, dataset_dir):
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'dataset_ft.pkl')
        self.intermediate_file = os.path.join(self.dataset_dir, 'intermediate.pkl')

        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')
        self.train_dataset_dir_unsupervised = os.path.join(self.dataset_dir, 'train_dataset_unsupervised')
        self.val_dataset_dir_unsupervised = os.path.join(self.dataset_dir, 'val_dataset_unsupervised')

        self.unique_prev_moves = set()

    def preprocess(self):
        f_1 = os.path.join(self.dataset_dir, 'lichess_db_puzzle_0_1mil.pkl')
        f_2 = os.path.join(self.dataset_dir, 'lichess_db_puzzle_1_2mil.pkl')

        with open(f_1, 'rb') as f:
            puzzles_1 = pickle.load(f)

        with open(f_2, 'rb') as f:
            puzzles_2 = pickle.load(f)

        all_puzzles = puzzles_1 + puzzles_2

        # all_puzzles = all_puzzles[:10000]

        # shuffle all_puzzles
        random.shuffle(all_puzzles)

        # split into train and validation
        train_split = 0.93
        train_puzzles = all_puzzles[:int(len(all_puzzles) * train_split)]
        val_puzzles = all_puzzles[int(len(all_puzzles) * train_split):]

        # Sort puzzles by rating
        train_puzzles.sort(key=lambda x: x['rating'])
        val_puzzles.sort(key=lambda x: x['rating'])




        pool = Pool(16)  # Number of CPUs

        # Train Datapoints
        print('Processing Train Datapoints')
        train_datapoints = []
        with tqdm(total=len(train_puzzles)) as progress_bar:
            for processed_datapoints in pool.imap(handle_datapoint, train_puzzles):
                progress_bar.update(1)
                if processed_datapoints is not None:
                    train_datapoints += processed_datapoints

        # Validation Datapoints
        print('Processing Validation Datapoints')
        val_datapoints = []
        with tqdm(total=len(val_puzzles)) as progress_bar:
            for processed_datapoints in pool.imap(handle_datapoint, val_puzzles):
                progress_bar.update(1)
                if processed_datapoints is not None:
                    val_datapoints += processed_datapoints


        datapoints = {
            'train': train_datapoints,
            'val': val_datapoints,
        }

        # Save datapoints in intermediate pickle file
        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(datapoints, f)





        return datapoints


    @staticmethod
    def decompose_datapoint(puzzle):
        # To get more data, use all the moves in the line
        def is_even(n):
            return n % 2 == 0

        ### Puzzle Keys
        # predict_idx = puzzle['predict_idx']
        # rating = puzzle['rating']
        url = puzzle['url']
        moves = puzzle['moves'].split(' ')
        line = puzzle['line'].split(' ')

        board = chess.Board()
        for move in moves:
            board.push_uci(move)

        points = []
        for idx, line_move in enumerate(line):
            if is_even(idx):
                # Get legal moves in position
                legal_moves = [move.uci() for move in board.legal_moves]
                legal_moves_scores = [0. for _ in legal_moves]
                legal_moves_idx = [config.vocab.index(move) for move in legal_moves]
                top_move = [line_move]
                top_move_scores = [1.]
                top_move_idx = [config.vocab.index(line_move)]
                prev_moves = deepcopy(moves)
                prev_moves.append('[mask]')
                points.append({
                    'prev_moves': ' '.join(prev_moves),        # used
                    'top_move': top_move,
                    'top_move_scores': top_move_scores,        # used
                    'top_move_idx': top_move_idx,              # used
                    'legal_moves': legal_moves,
                    'legal_moves_scores': legal_moves_scores,  # used
                    'legal_moves_idx': legal_moves_idx,        # used
                    'url': url,
                    'rating': puzzle['rating'],
                })
            moves.append(line_move)
            board.push_uci(line_move)
        return points







    def get_dataset(self):
        if os.path.exists(self.intermediate_file):
            with open(self.intermediate_file, 'rb') as f:
                datapoints = pickle.load(f)
        else:
            datapoints = self.preprocess()

        train_datapoints = datapoints['train']
        val_datapoints = datapoints['val']

        print('Train Datapoints:', len(train_datapoints))
        print('Val Datapoints:', len(val_datapoints))

        # Aggregate Train Data
        train_top_moves_cp_norm = []
        train_prev_moves = []
        train_top_moves_idx = []
        train_legal_moves_idx = []
        train_legal_moves_scores = []
        for datapoint in tqdm(train_datapoints):
            train_top_moves_cp_norm.append(datapoint['top_move_scores'])
            train_prev_moves.append(datapoint['prev_moves'])
            train_top_moves_idx.append(datapoint['top_move_idx'])
            train_legal_moves_idx.append(datapoint['legal_moves_idx'])
            train_legal_moves_scores.append(datapoint['legal_moves_scores'])


        # Pad Train Data
        max_length = max(len(lst) for lst in train_legal_moves_idx)
        train_legal_moves_idx = [lst + [lst[0]] * (max_length - len(lst)) for lst in train_legal_moves_idx]

        max_length = max(len(lst) for lst in train_legal_moves_scores)
        train_legal_moves_scores = [lst + [lst[0]] * (max_length - len(lst)) for lst in train_legal_moves_scores]



        # Aggregate Validation Data
        val_top_moves_cp_norm = []
        val_prev_moves = []
        val_top_moves_idx = []
        val_legal_moves_idx = []
        val_legal_moves_scores = []
        for datapoint in tqdm(val_datapoints):
            val_top_moves_cp_norm.append(datapoint['top_move_scores'])
            val_prev_moves.append(datapoint['prev_moves'])
            val_top_moves_idx.append(datapoint['top_move_idx'])
            val_legal_moves_idx.append(datapoint['legal_moves_idx'])
            val_legal_moves_scores.append(datapoint['legal_moves_scores'])


        # Pad Validation Data
        max_length = max(len(lst) for lst in val_legal_moves_idx)
        val_legal_moves_idx = [lst + [lst[0]] * (max_length - len(lst)) for lst in val_legal_moves_idx]

        max_length = max(len(lst) for lst in val_legal_moves_scores)
        val_legal_moves_scores = [lst + [lst[0]] * (max_length - len(lst)) for lst in val_legal_moves_scores]



        # Train Dataset
        print('Creating Train Dataset From Slices...')
        train_dataset = tf.data.Dataset.from_tensor_slices((train_top_moves_cp_norm, train_prev_moves,
                                                            train_top_moves_idx, train_legal_moves_idx,
                                                            train_legal_moves_scores))

        # Validation Dataset
        print('Creating Validation Dataset From Slices...')
        val_dataset = tf.data.Dataset.from_tensor_slices((val_top_moves_cp_norm, val_prev_moves,
                                                          val_top_moves_idx, val_legal_moves_idx,
                                                          val_legal_moves_scores))

        train_dataset, val_dataset = self.create_supervised_datasets(train_dataset, val_dataset)

        self.save_datasets(train_dataset, val_dataset)




    def create_supervised_datasets(self, train_dataset, val_dataset):
        print('Creating Supervised Datasets...')
        cpu_count = tf.data.AUTOTUNE
        # cpu_count = 32

        # train_dataset = train_dataset.shuffle(config.ft_train_buffer)
        train_dataset = train_dataset.batch(config.global_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # val_dataset = val_dataset.shuffle(config.ft_val_buffer)
        val_dataset = val_dataset.batch(config.global_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset


    def save_datasets(self, train_dataset, val_dataset):
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)





if __name__ == '__main__':
    generator = PuzzlesDatasetGenerator(config.ft_lichess_puzzles)
    # generator.preprocess()
    generator.get_dataset()



