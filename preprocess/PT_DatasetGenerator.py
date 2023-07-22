import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm

import tensorflow_datasets as tfds
import random
import chess
import warnings

import threading
import multiprocessing
multiprocessing.set_start_method('fork')
import time
import sys




from preprocess.strategies import window_pt
from preprocess.strategies import denoising_objective







class PT_DatasetGenerator:
    def __init__(self, dataset_dir):

        # 1. Initialize
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.game_file = os.path.join(self.dataset_dir, 'bulk_games.pgn')
        self.archive_file = os.path.join(self.dataset_dir, self.dataset_name + '.zip')
        self.chunk_pgn_dir = os.path.join(self.dataset_dir, 'chunks_pgn')
        if not os.path.exists(self.chunk_pgn_dir):
            os.makedirs(self.chunk_pgn_dir)
        self.chunk_uci_dir = os.path.join(self.dataset_dir, 'chunks_uci')
        self.chunk_san_dir = os.path.join(self.dataset_dir, 'chunks_san')
        if not os.path.exists(self.chunk_uci_dir):
            os.makedirs(self.chunk_uci_dir)
        if not os.path.exists(self.chunk_san_dir):
            os.makedirs(self.chunk_san_dir)
        self.chunk_size = 100000

        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')


    ####################
    ### 1. Chunk PGN ###
    ####################

    def chunk_pgn_file(self):
        if os.listdir(self.chunk_pgn_dir):
            print("Chunks already exist. Skipping chunking.")
            return
        game_count = 0
        file_count = 0
        buffer = []
        progress_bar = tqdm(desc="Splitting PGN", unit="Games")
        new_game = True
        with open(self.game_file, 'r', errors="ignore") as pgn_file:
            curr_newline_count = 0
            for line in pgn_file:
                if new_game and line.strip() == '':
                    continue
                new_game = False
                buffer.append(line)
                if line.strip() == '':
                    curr_newline_count += 1
                    if curr_newline_count > 1:
                        game_count += 1
                        progress_bar.update(1)
                        if game_count == self.chunk_size:
                            file_name = os.path.join(self.chunk_pgn_dir, f"chunk_{file_count}_100k.pgn")
                            with open(file_name, 'w') as chunk_file:
                                chunk_file.writelines(buffer)
                            buffer = []
                            game_count = 0
                            file_count += 1
                        curr_newline_count = 0
                        new_game = True
        if buffer:
            file_name = os.path.join(self.chunk_pgn_dir, f"chunk_{file_count}_{game_count}.pgn")
            with open(file_name, 'w') as chunk_file:
                chunk_file.writelines(buffer)

    ####################
    ### 2. Chunk UCI ###
    ####################

    def parse_dir_games(self):
        if os.listdir(self.chunk_uci_dir):
            print("UCI Chunks already exist. Skipping chunking.")
            return
        game_files = os.listdir(self.chunk_pgn_dir)
        process_list = []
        for game_file in game_files:
            game_file_path = os.path.join(self.chunk_pgn_dir, game_file)
            game_file_name = game_file.split('.')[0]

            save_file = None
            if config.move_language == 'uci':
                save_file = os.path.join(self.chunk_uci_dir, game_file_name + '.txt')
            elif config.move_language == 'san':
                save_file = os.path.join(self.chunk_san_dir, game_file_name + '.txt')

            process = multiprocessing.Process(target=self.parse_games_linear, args=(game_file_path, save_file))
            process.start()
            process_list.append(process)
        for th in process_list:
            th.join()

    def parse_games_linear(self, game_file, save_file):
        print('Parsing', game_file, 'to', save_file)
        games = []

        # redirect stdout
        warnings.filterwarnings("ignore")  # Ignore warnings
        orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        # Iterate over each game,
        with open(game_file) as pgn_file:
            cnt = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break

                    parsed_moves = None
                    if config.move_language == 'uci':
                        parsed_moves = self.parse_game_moves_uci(game)
                    elif config.move_language == 'san':
                        parsed_moves = self.parse_game_moves_san(game)

                    if parsed_moves:
                        games.append(parsed_moves)
                        cnt += 1
                except Exception as e:
                    print('--> EXCEPTION: ', game_file)
                    continue
        # with open(save_file, 'wb') as f:
        #     pickle.dump(games, f)
        with open(save_file, 'w', encoding='utf-8') as f:
            for line in games:
                f.write(line + '\n')
        f.close()

        # Restore stdout
        sys.stdout = orig_stdout


        print('Finished parsing', game_file, 'to', save_file)

    def parse_game_moves_uci(self, game, draw_tokens=False):
        move_list = list(move.uci() for move in game.mainline_moves())
        if len(move_list) < 12 or any('@' in s for s in move_list):
            return None
        result = game.headers["Result"]
        if result == '1-0':
            move_list.append('[white]')
        elif result == '0-1':
            move_list.append('[black]')
        elif result == '1/2-1/2' and draw_tokens is True:
            move_list.append('[draw]')
        return ' '.join(move_list)


    def parse_game_moves_san(self, game):
        move_list = []
        board = chess.Board()

        warnings.filterwarnings("ignore")  # Ignore warnings
        orig_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        for move in game.mainline_moves():
            try:
                san_move = board.san(move)
                board.push(move)
            except Exception as e:
                if len(move_list) > 5:
                    return ' '.join(move_list)
                else:
                    return None
            move_list.append(san_move)

        move_str = ' '.join(move_list)
        return move_str

    ##########################
    ### 3. Procure Dataset ###
    ##########################

    def get_dataset(self, interleave=False, save=False, small=False):
        chunk_dir = None
        if config.move_language == 'uci':
            chunk_dir = self.chunk_uci_dir
        elif config.move_language == 'san':
            chunk_dir = self.chunk_san_dir


        # 1. Get and split move files
        if not os.listdir(chunk_dir):
            print("No UCI files. Skipping dataset creation.")
            return

        move_files = None
        if config.move_language == 'uci':
            move_files = self.load_uci_files()
        elif config.move_language == 'san':
            move_files = self.load_san_files()

        random.shuffle(move_files)
        split_idx = int(len(move_files) * 0.94)
        train_move_files, val_move_files = move_files[:split_idx], move_files[split_idx:]
        if small:
            train_move_files, val_move_files = train_move_files[:5], val_move_files[:1]
        else:
            self.balance_val_files(val_move_files, kill=True)

        print("Train files:", len(train_move_files))
        print("Val files:", len(val_move_files))

        if interleave:
            print("Interleaving train dataset...")
            train_dataset = self.parse_interleave_dataset(train_move_files)
            print("Interleaving val dataset...")
            val_dataset = self.parse_interleave_dataset(val_move_files)
        else:
            print("Parsing train dataset...")
            # train_dataset = self.parse_memory_dataset(train_move_files, buffer=2048*1000)
            train_dataset = tf.data.TextLineDataset(train_move_files)
            print("Parsing val dataset...")
            # val_dataset = self.parse_memory_dataset(val_move_files, buffer=256)
            val_dataset = tf.data.TextLineDataset(val_move_files)


        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset

    def balance_val_files(self, val_files, kill=False):
        cc_count = 0
        mil_count = 0
        for vfile in val_files:
            print(vfile)
            if 'cc' in vfile:
                cc_count += 1
            elif 'mil' in vfile:
                mil_count += 1
        if cc_count < 2 or mil_count < 2:
            if kill is True:
                exit(0)


    def parse_memory_dataset(self, move_files, buffer=1024):
        full_dataset = tf.data.TextLineDataset(move_files)
        full_dataset = full_dataset.shuffle(buffer)
        full_dataset = full_dataset.batch(config.pt_batch_size)
        full_dataset = full_dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        full_dataset = full_dataset.map(window_pt.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        return full_dataset.prefetch(tf.data.AUTOTUNE)

    def parse_interleave_dataset(self, move_files):
        def parse_fn(file_path):
            dataset = tf.data.TextLineDataset(file_path)
            dataset = dataset.batch(config.pt_batch_size)
            dataset = dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(window_pt.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(1024)
            return dataset.prefetch(tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.from_tensor_slices(move_files)
        dataset = dataset.interleave(
            parse_fn,
            cycle_length=tf.data.AUTOTUNE,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def load_uci_files(self):
        move_files = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                move_files.append(full_path)
        return move_files

    def load_san_files(self):
        move_files = []
        for file in os.listdir(self.chunk_san_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_san_dir, file)
                move_files.append(full_path)
        return move_files

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

    def load_unsupervised_datasets(self, train_buffer=1024, val_buffer=256):
        train_dataset = tf.data.Dataset.load(self.train_dataset_dir)
        train_dataset = train_dataset.shuffle(train_buffer)
        train_dataset = train_dataset.batch(config.global_batch_size)
        train_dataset = train_dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.map(window_pt.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        val_dataset = val_dataset.shuffle(val_buffer)
        val_dataset = val_dataset.batch(config.global_batch_size)
        val_dataset = val_dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.map(window_pt.preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def load_val_dataset(self):
        val_dataset = tf.data.Dataset.load(self.val_dataset_dir)
        return val_dataset


if __name__ == '__main__':
    # config.pt_megaset_dataset
    # config.pt_millionsbase_dataset
    generator = PT_DatasetGenerator(config.pt_megaset_dataset)
    # generator.chunk_pgn_file()
    # generator.parse_dir_games()
    generator.get_dataset(save=True, small=False)











