import config
import chess.pgn
import os
import tensorflow as tf
from tqdm import tqdm
from preprocess.strategies.dual_objective_flat import dual_objective_flat_batch

import chess


import threading
import multiprocessing
multiprocessing.set_start_method('fork')
import time






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
        if not os.path.exists(self.chunk_uci_dir):
            os.makedirs(self.chunk_uci_dir)
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
            save_file = os.path.join(self.chunk_uci_dir, game_file_name + '.txt')
            process = multiprocessing.Process(target=self.parse_games_linear, args=(game_file_path, save_file))
            process.start()
            process_list.append(process)
        for th in process_list:
            th.join()

    def parse_games_linear(self, game_file, save_file):
        print('Parsing', game_file, 'to', save_file)
        games = []
        # Iterate over each game,
        with open(game_file) as pgn_file:
            cnt = 0
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break
                    parsed_moves = self.parse_game_moves_uci(game)
                    # parsed_moves = self.parse_game_moves_san(game)
                    if parsed_moves:
                        games.append(parsed_moves)
                        cnt += 1
                except ValueError as e:
                    continue
        # with open(save_file, 'wb') as f:
        #     pickle.dump(games, f)
        with open(save_file, 'w', encoding='utf-8') as f:
            for line in games:
                f.write(line + '\n')
        f.close()
        print('Finished parsing', game_file, 'to', save_file)

    def parse_game_moves_uci(self, game):
        move_list = list(move.uci() for move in game.mainline_moves())
        move_str = ' '.join(move_list)
        if '@' in move_str or len(move_list) < 5:
            return None
        else:
            return move_str

    def parse_game_moves_san(self, game):
        move_list = []
        board = chess.Board()
        for move in game.mainline_moves():
            move_list.append(board.san(move))
            board.push(move)
        move_str = ' '.join(move_list)
        if '@' in move_str or len(move_list) < 5:
            return None
        else:
            return move_str


    ##########################
    ### 3. Procure Dataset ###
    ##########################

    def get_dataset(self, interleave=False, save=False, small=False):

        # 1. Get and split move files
        if not os.listdir(self.chunk_uci_dir):
            print("No UCI files. Skipping dataset creation.")
            return
        move_files = self.load_uci_files()
        split_idx = int(len(move_files) * 0.9)
        train_move_files, val_move_files = move_files[:split_idx], move_files[split_idx:]
        if small:
            train_move_files, val_move_files = train_move_files[:10], val_move_files[:1]
        print("Train files:", len(train_move_files))
        print("Val files:", len(val_move_files))

        if interleave:
            train_dataset = self.parse_interleave_dataset(train_move_files)
            val_dataset = self.parse_interleave_dataset(val_move_files)
        else:
            train_dataset = self.parse_memory_dataset(train_move_files)
            val_dataset = self.parse_memory_dataset(val_move_files)

        if save:
            self.save_datasets(train_dataset, val_dataset)

        return train_dataset, val_dataset

    def parse_memory_dataset(self, move_files):
        full_dataset = tf.data.TextLineDataset(move_files)
        full_dataset = full_dataset.batch(config.pt_batch_size)
        full_dataset = full_dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
        full_dataset = full_dataset.map(dual_objective_flat_batch, num_parallel_calls=tf.data.AUTOTUNE)
        full_dataset = full_dataset.shuffle(100)
        return full_dataset.prefetch(tf.data.AUTOTUNE)

    def parse_interleave_dataset(self, move_files):
        def parse_fn(file_path):
            dataset = tf.data.TextLineDataset(file_path)
            dataset = dataset.batch(config.pt_batch_size)
            dataset = dataset.map(config.encode_tf_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(dual_objective_flat_batch, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.shuffle(3125)
            return dataset.prefetch(tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.from_tensor_slices(move_files)
        dataset = dataset.interleave(
            parse_fn,
            cycle_length=tf.data.AUTOTUNE,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
        dataset = dataset.shuffle(50)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def load_uci_files(self):
        move_files = []
        for file in os.listdir(self.chunk_uci_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.chunk_uci_dir, file)
                move_files.append(full_path)
        return move_files

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


if __name__ == '__main__':
    generator = PT_DatasetGenerator(config.pt_millionsbase_dataset)
    generator.chunk_pgn_file()
    generator.parse_dir_games()
    generator.get_dataset(save=True, small=False)











