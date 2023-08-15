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


from preprocess.strategies import move_prediction


def process_sequence(args):
    sequence_p, masking_index, top_move, top_moves_eval, unique_set = args
    return EvaluationsDatasetGenerator.preprocess_datapoint(
        sequence_p.split(' '),
        masking_index,
        top_move,
        top_moves_eval,
        unique_set
    )



class EvaluationsDatasetGenerator:
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




    def get_dataset(self, save=True, supervised=False):
        datapoints = self.get_datapoints()


        # Aggregate Data
        top_moves_cp_norm = []
        prev_moves = []
        top_moves_idx = []
        legal_moves_idx = []
        legal_moves_scores = []
        for datapoint in tqdm(datapoints):
            top_moves_cp_norm.append(datapoint['top_moves_cp_norm'])
            prev_moves.append(datapoint['prev_moves'])
            top_moves_idx.append(datapoint['top_moves_idx'])
            legal_moves_idx.append(datapoint['legal_moves_idx'])
            legal_moves_scores.append(datapoint['legal_moves_scores'])


        # Pad legal moves idx and scores
        # Pad all_legal_moves_idx and all_legal_moves_scores
        max_length = max(len(lst) for lst in legal_moves_idx)
        legal_moves_idx = [lst + [lst[0]] * (max_length - len(lst)) for lst in legal_moves_idx]

        max_length = max(len(lst) for lst in legal_moves_scores)
        legal_moves_scores = [lst + [lst[0]] * (max_length - len(lst)) for lst in legal_moves_scores]


        # Split Aggregate Data into Train and Val
        print('--> SPLITTING DATASETS')
        split_idx = int(len(datapoints) * 0.9)
        top_moves_cp_norm_train = top_moves_cp_norm[:split_idx]
        prev_moves_train = prev_moves[:split_idx]
        top_moves_idx_train = top_moves_idx[:split_idx]
        legal_moves_idx_train = legal_moves_idx[:split_idx]
        legal_moves_scores_train = legal_moves_scores[:split_idx]
        train_dataset = tf.data.Dataset.from_tensor_slices((top_moves_cp_norm_train, prev_moves_train, top_moves_idx_train, legal_moves_idx_train, legal_moves_scores_train))

        top_moves_cp_norm_val = top_moves_cp_norm[split_idx:]
        prev_moves_val = prev_moves[split_idx:]
        top_moves_idx_val = top_moves_idx[split_idx:]
        legal_moves_idx_val = legal_moves_idx[split_idx:]
        legal_moves_scores_val = legal_moves_scores[split_idx:]
        val_dataset = tf.data.Dataset.from_tensor_slices((top_moves_cp_norm_val, prev_moves_val, top_moves_idx_val, legal_moves_idx_val, legal_moves_scores_val))

        if supervised is True:
            train_dataset, val_dataset = self.create_supervised_datasets(train_dataset, val_dataset)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset



    def get_datapoints(self):

        # Open and return intermediate file if exists
        if os.path.exists(self.intermediate_file):
            with open(self.intermediate_file, 'rb') as f:
                datapoints = pickle.load(f)
            return datapoints


        with open(self.game_file, 'rb') as f:
            eval_data = pickle.load(f)

        print(eval_data.keys())
        print('NUM DATAPOINTS: ', len(eval_data['move_sequences']))

        # ['move_sequences', 'masking_indices', 'top_moves', 'top_moves_evals', 'top_lines', 'probability_scores']
        move_sequences = eval_data['move_sequences']
        masking_indices = eval_data['masking_indices']
        top_moves = eval_data['top_moves']
        top_moves_evals = eval_data['top_moves_evals']
        top_lines = eval_data['top_lines']
        probability_scores = eval_data['probability_scores']

        # Shuffle Data
        combined = list(zip(move_sequences, masking_indices, top_moves, top_moves_evals, top_lines, probability_scores))
        random.shuffle(combined)
        move_sequences, masking_indices, top_moves, top_moves_evals, top_lines, probability_scores = zip(*combined)


        # Preprocess Data Multiprocessing
        print('Creating Args...')
        pool = Pool(32)  # Number of CPUs
        args_list = [
            (sequence, masking_indices[idx], top_moves[idx], top_moves_evals[idx], self.unique_prev_moves)
            for idx, sequence in enumerate(move_sequences)]

        print('Preprocessing Multiple Processes...')
        datapoints = []
        with tqdm(total=len(move_sequences)) as progress_bar:
            for processed_datapoint in pool.imap(process_sequence, args_list):
                progress_bar.update(1)
                if processed_datapoint is not None:
                    datapoints.append(processed_datapoint)

        # Save datapoints in intermediate pickle file
        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(datapoints, f)

        return datapoints

    @staticmethod
    def preprocess_datapoint(game_moves, eval_idx, top_moves, top_moves_cp, unique_set=None):
        prev_moves = game_moves[:eval_idx]

        # 1. Validate moves exist and no duplicates
        if len(prev_moves) >= config.seq_length:
            return None

        if len(prev_moves) < 75 and unique_set is not None:
            if ' '.join(prev_moves) in unique_set:
                return None
            else:
                unique_set.add(' '.join(prev_moves))


        # 2. Find legal moves in position
        board = chess.Board()
        move_idx = 0
        try:
            for move in prev_moves:
                board.push_uci(move)
                move_idx += 1
        except Exception as ex:
            # print('EXCEPTION PUSHING MOVE:', ex)
            # print('MOVE:', prev_moves)
            # print('IDX:', move_idx)
            return None


        white_turn = board.turn == chess.WHITE
        legal_candidate_moves = [move.uci() for move in board.legal_moves]
        legal_candidate_moves_idx = [config.vocab.index(move) for move in legal_candidate_moves]
        legal_candidate_moves_scores = [0. for _ in legal_candidate_moves]

        # 3 Add mask, get top move indices
        prev_moves.append('[mask]')
        top_moves_idx = [config.vocab.index(move) for move in top_moves]

        # 4. Compute relevancy scores
        top_moves_cp_norm = EvaluationsDatasetGenerator.min_max_normalize(top_moves_cp)

        # 5. Ensure top_moves always has 5 elements
        while len(top_moves_idx) < 5:
            top_moves.append(top_moves[0])
            top_moves_idx.append(top_moves_idx[0])
            top_moves_cp_norm.append(top_moves_cp_norm[0])


        # 6. Return datapoint
        #  all_candidate_scores, all_previous_moves, all_candidate_moves_idx, all_legal_moves_idx, all_legal_moves_scores
        return {
            'top_moves_cp_norm': top_moves_cp_norm,  # all_candidate_scores
            'prev_moves': ' '.join(prev_moves),  # all_previous_moves
            'top_moves_idx': top_moves_idx,  # all_candidate_moves_idx
            'legal_moves_idx': legal_candidate_moves_idx,  # all_legal_moves_idx
            'legal_moves_scores': legal_candidate_moves_scores,  # all_legal_moves_scores

            'top_moves': top_moves,
            'white_turn': white_turn
        }


    @staticmethod
    def min_max_normalize(evals, range_min=0.2, range_max=1.0):
        min_eval = min(evals)
        max_eval = max(evals)

        # Normalization formula: (x - min) / (max - min)
        normalized_evals = [(x - min_eval) / (max_eval - min_eval) if max_eval > min_eval else 0.0 for x in evals]

        # Scaling to the range [range_min, range_max]
        scaled_evals = [x * (range_max - range_min) + range_min for x in normalized_evals]

        return scaled_evals


    def save_datasets(self, train_dataset, val_dataset):
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)

    def load_datasets(self):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return train_dataset, val_dataset


    def load_unsupervised_datasets(self, train_buffer=1024, val_buffer=256, batch_size=config.global_batch_size):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir_unsupervised)
        train_dataset = train_dataset.shuffle(train_buffer)
        train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.load(self.val_dataset_dir_unsupervised)
        val_dataset = val_dataset.shuffle(val_buffer)
        val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset


    def create_supervised_datasets(self, train_dataset, val_dataset):
        print('Creating Supervised Datasets...')
        cpu_count = tf.data.AUTOTUNE
        # cpu_count = 32

        train_dataset = train_dataset.shuffle(config.ft_train_buffer)
        train_dataset = train_dataset.batch(config.global_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = val_dataset.shuffle(config.ft_val_buffer)
        val_dataset = val_dataset.batch(config.global_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_prediction.move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset





if __name__ == '__main__':
    generator = EvaluationsDatasetGenerator(config.ft_evaluations)
    generator.get_dataset(save=True, supervised=True)





