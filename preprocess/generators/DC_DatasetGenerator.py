import config
import os
import pickle
import tensorflow as tf
import random
import chess

from tqdm import tqdm
from preprocess import utils
from preprocess.strategies.old.move_ranking_2d_board import encode_batch
from preprocess.strategies.move_prediction import move_ranking_batch




class DC_DatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'bulk_games.txt')
        self.intermediate_file = os.path.join(self.dataset_dir, 'bulk_games_tactics.pkl')
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')
        self.archive_file = os.path.join(self.dataset_dir, self.dataset_name + '.zip')
        self.max_positions = 2000000
        self.unique_prev_moves = set()



    ################################
    ### 1. Gen Intermediate File ###
    ################################

    @staticmethod
    def preprocess_datapoint(prev_moves, candidate_moves_info, candidate_moves_scores_cp, best_score_cp, unique_short_evals=None, mask=True):

        # 1. Verify datapoint fits sequence length
        if prev_moves is None or len(prev_moves) >= config.seq_length:
            return None

        # 2. If less than 75 moves, ensure unique
        if len(prev_moves) < 75 and unique_short_evals is not None:
            if ' '.join(prev_moves) in unique_short_evals:
                return None
            else:
                unique_short_evals.add(' '.join(prev_moves))

        # 3. Add mask token to end prev_moves, get legal uci moves in position
        board = chess.Board()
        for move in prev_moves:
            board.push_uci(move)
        legal_candidate_moves = [move.uci() for move in board.legal_moves]
        legal_candidate_moves_idx = [config.vocab.index(move) for move in legal_candidate_moves]
        legal_candidate_moves_scores = [10. for _ in legal_candidate_moves]
        if mask is True:
            prev_moves.append('[mask]')

        # 4. Compute absolute differences, find max difference
        abs_diff_scores = [abs(score - best_score_cp) for score in candidate_moves_scores_cp]
        max_abs_diff = max(abs_diff_scores)

        # 5. Normalize absolute differences by max difference
        candidate_moves_scores_dn = [score / max_abs_diff if max_abs_diff != 0 else 0. for score in abs_diff_scores]

        # 6. Invert the scores so that a higher score is better.
        candidate_moves_scores_dn = [round(1 - score, 3) for score in candidate_moves_scores_dn]
        candidate_moves_scores_dn = [score * 100. for score in candidate_moves_scores_dn]

        # 7. Replace 0.0 with 0.2, as 0.0 is reserved for non-evaluated moves
        candidate_moves_scores_dn = [x if x != 0. else 20. for x in candidate_moves_scores_dn]

        # 8. Sort moves and norm_score together on norm_score with zip
        candidate_moves_info, candidate_moves_scores_dn = zip(*sorted(zip(candidate_moves_info, candidate_moves_scores_dn), key=lambda x: x[1], reverse=True))
        candidate_moves_scores_dn = list(candidate_moves_scores_dn)
        candidate_moves = [move['move'] for move in candidate_moves_info]
        candidate_moves_idx = [config.vocab.index(move) for move in candidate_moves]

        # 9. Ensure there are at least top n moves
        while len(candidate_moves) < 3:
            padding_move = ''
            candidate_moves.append(padding_move)
            candidate_moves_idx.append(config.vocab.index(padding_move))
            candidate_moves_scores_dn.append(0.0)

        # 10. Return datapoint
        return {
            'candidate_scores': candidate_moves_scores_dn,
            'candidate_moves': candidate_moves,
            'candidate_moves_idx': candidate_moves_idx,
            'legal_moves_idx': legal_candidate_moves_idx,
            'legal_moves_scores': legal_candidate_moves_scores,
            'prev_moves': ' '.join(prev_moves)
        }




    def parse_bulk_games(self):
        unique_short_evals = set()

        eval_data = []
        with open(self.game_file, 'r') as f:
            file_lines = f.readlines()
            for line in tqdm(file_lines[1:self.max_positions+1]):

                # 1. Parse line
                prev_moves, moves, cp_scores, best_score = utils.parse_ft_line(line)

                # 2. Parse Datapoint
                mask = True
                result = self.preprocess_datapoint(prev_moves, moves, cp_scores, best_score, unique_short_evals, mask=mask)
                if result is None:
                    continue
                else:
                    eval_data.append(result)

        with open(self.intermediate_file, 'wb') as f:
            pickle.dump(eval_data, f)

        return eval_data



    ##########################
    ### 2. Procure Dataset ###
    ##########################

    def get_datasets(self, save=False, small=False):
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
        dataset = self.create_and_pad_dataset(positions, self.unique_prev_moves)
        return dataset


    @staticmethod
    def create_and_pad_dataset(positions, unique_prev_moves=set()):
        dupe_counter = 0
        long_counter = 0

        all_previous_moves = []
        all_san_previous_moves = []
        all_candidate_moves_idx = []
        all_candidate_scores = []
        all_legal_moves_idx = []
        all_legal_moves_scores = []
        for position in positions:
            prev_moves = position['prev_moves'].split(' ')


            # if prev_moves[-1] == '[mask]':
            #     prev_moves = prev_moves[:-1]

            if len(prev_moves) >= config.seq_length - 1:
                long_counter += 1
                continue

            if len(prev_moves) <= 70:
                if ' '.join(prev_moves) in unique_prev_moves:
                    print('Duplicate prev moves:', dupe_counter, ' '.join(prev_moves))
                    dupe_counter += 1
                    continue
                else:
                    unique_prev_moves.add(' '.join(prev_moves))

            # san_prev_moves = []
            # board = chess.Board()
            # for move in prev_moves:
            #     board.push_uci(move)
            #     san_move = board.san(move)
            #     san_prev_moves.append(san_move)
            # san_prev_moves = ' '.join(san_prev_moves)
            # all_san_previous_moves.append(san_prev_moves)



            prev_moves = ' '.join(prev_moves)
            all_previous_moves.append(prev_moves)
            all_candidate_moves_idx.append(position['candidate_moves_idx'])
            all_candidate_scores.append(position['candidate_scores'])
            all_legal_moves_idx.append(position['legal_moves_idx'])
            all_legal_moves_scores.append(position['legal_moves_scores'])



        print('Duplicate prev moves:', dupe_counter)
        print('Long datapoints skipped:', long_counter)

        # Pad all_legal_moves_idx and all_legal_moves_scores
        max_length = max(len(lst) for lst in all_legal_moves_idx)
        all_legal_moves_idx = [lst + [config.token2id['']] * (max_length - len(lst)) for lst in all_legal_moves_idx]

        max_length = max(len(lst) for lst in all_legal_moves_scores)
        all_legal_moves_scores = [lst + [0.0] * (max_length - len(lst)) for lst in all_legal_moves_scores]

        return tf.data.Dataset.from_tensor_slices((all_candidate_scores, all_previous_moves, all_candidate_moves_idx, all_legal_moves_idx, all_legal_moves_scores))

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
        train_dataset = train_dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        val_dataset = val_dataset.shuffle(val_buffer)
        val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset







if __name__ == '__main__':
    generator = DC_DatasetGenerator(config.ft_lichess)
    # generator.parse_bulk_games()
    generator.get_datasets(save=True, small=False)











