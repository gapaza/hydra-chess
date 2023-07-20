import tensorflow as tf
import tensorflow_ranking as tfr
import config
import os
import chess
import random


from preprocess.strategies.move_ranking import move_ranking_batch, encode_batch
from preprocess.strategies.move_ranking_flat import move_ranking_batch_flat, move_ranking_flat
from preprocess.strategies.window_masking import rand_window_multi, rand_window_batch_multi
from preprocess.strategies.py_utils import board_to_tensor_classes
from preprocess.strategies.dual_objective import dual_objective_batch, dual_objective
from preprocess.strategies.dual_objective_flat import dual_objective_flat_batch, dual_objective_flat, dual_objective_batch
from preprocess.strategies.denoising_objective import denoising_objective

from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator

from preprocess.strategies.move_ranking_flat import encode_batch as encode_ft_batch




class StrategyTesting:

    def __init__(self):
        self.test = 0

    def get_pt_uci_dataset(self, batch=True, batch_size=3):
        dataset_path = os.path.join(config.pt_millionsbase_dataset, 'chunks_uci', 'pgn_chunk_0_100000.txt')
        dataset = tf.data.TextLineDataset(dataset_path)
        if batch is True:
            dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            return dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            return dataset.map(config.encode_tf_old, num_parallel_calls=tf.data.AUTOTUNE)


    def get_ft_uci_dataset(self, batch_size=3):
        prev_moves = ['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3']
        candidate_moves_info = [{'move': 'b8c6'}, {'move': 'a7a6'}, {'move': 'g7g6'}]
        candidate_moves_scores_cp = [8.2, 4.3, 2.2]
        best_score_cp = max(candidate_moves_scores_cp)
        datapoint = DC_DatasetGenerator.preprocess_datapoint(prev_moves, candidate_moves_info, candidate_moves_scores_cp, best_score_cp)
        # for key, value in datapoint.items():
        #     print(key, value)
        dataset = DC_DatasetGenerator.create_and_pad_dataset([datapoint])
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.map(encode_ft_batch, num_parallel_calls=tf.data.AUTOTUNE)



    ###############
    ### Testing ###
    ###############

    def test_dual_objective_flat(self):
        dataset = self.get_pt_uci_dataset(batch=True, batch_size=3)
        dataset = dataset.map(dual_objective_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        first_element = next(iter(dataset.take(1)))
        for idx, element in enumerate(first_element):
            print('\n\n\nELEMENT', idx, ':', element)

    def test_denoising_objecive(self):
        dataset = self.get_pt_uci_dataset(batch=True, batch_size=5)
        dataset = dataset.map(denoising_objective, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        first_element = next(iter(dataset.take(1)))
        for idx, element in enumerate(first_element):
            print('\n\n\nELEMENT', idx, ':', element)





if __name__ == '__main__':
    st = StrategyTesting()
    st.test_denoising_objecive()




