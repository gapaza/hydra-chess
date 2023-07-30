import tensorflow as tf
import config
import os
import tensorflow_datasets as tfds

from preprocess.generators.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.generators.PT_Eval_DatasetGenerator import PT_Eval_DatasetGenerator
from preprocess.strategies.denoising_objective import preprocess_batch

from preprocess.generators.DC_DatasetGenerator import DC_DatasetGenerator

from preprocess.strategies.move_prediction import encode_batch as encode_ft_batch

from preprocess.strategies.old import window_pt_small
from preprocess.strategies import position_modeling








class StrategyTesting:

    def __init__(self):
        self.test = 0

    def get_pt_uci_dataset(self, batch=True, batch_size=3):
        dataset_path = os.path.join(config_new.pt_dataset, 'chunks_uci', 'chunk_0_100k.txt')
        dataset = tf.data.TextLineDataset(dataset_path)
        if batch is True:
            dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            return dataset.map(config_new.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            return dataset.map(config_new.encode_tf_old, num_parallel_calls=tf.data.AUTOTUNE)


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

    def test_pt_window_small(self, batch_size=3, bench=False):
        dataset = self.get_pt_uci_dataset(batch=True, batch_size=batch_size)
        dataset = dataset.map(window_pt_small.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.print_dataset_element(dataset)
        if bench is True:
            self.benchmark_dataset(dataset, batch_size)


    def test_pt_window_med(self, batch_size=3, bench=False):
        dataset = self.get_pt_uci_dataset(batch=True, batch_size=batch_size)
        dataset = dataset.map(window_pt.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.print_dataset_element(dataset)
        if bench is True:
            self.benchmark_dataset(dataset, batch_size)






    def test_pt_window_variable(self, batch_size=5, bench=True):
        dataset_generator = PT_DatasetGenerator(
            config_new.pt_dataset,
        )
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=2048,
            val_buffer=256,
            batch_size=batch_size,
        )
        self.print_dataset_element(train_dataset)
        # self.benchmark_dataset(train_dataset, batch_size=batch_size)

    def test_pt_window_eval(self, batch_size=3, bench=True):
        dataset_generator = PT_Eval_DatasetGenerator(
            config_new.pt_dataset
        )
        train_dataset, val_dataset = dataset_generator.load_unsupervised_datasets(
            train_buffer=2048,
            val_buffer=256,
            batch_size=batch_size,
        )
        self.print_dataset_element(train_dataset)
        return 0









    def test_pt_denoising_objecive(self, batch_size=3, bench=False):
        dataset = self.get_pt_uci_dataset(batch=True, batch_size=batch_size)
        dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.print_dataset_element(dataset)
        if bench is True:
            self.benchmark_dataset(dataset, batch_size)







    #################
    ### Debugging ###
    #################


    def benchmark_dataset(self, dataset, batch_size):
        print('Benchmarking dataset...')
        print(tfds.benchmark(dataset, batch_size=batch_size))

    def print_dataset_element(self, dataset):
        first_element = next(iter(dataset.take(1)))
        for idx, element in enumerate(first_element):
            print('\n\n\nELEMENT', idx, ':', element)





if __name__ == '__main__':
    st = StrategyTesting()
    # st.test_pt_window_small()
    # st.test_pt_window_med(bench=False, batch_size=3)
    # st.test_pt_denoising_objecive()
    # st.test_pt_window_variable()
    st.test_pt_window_eval()



