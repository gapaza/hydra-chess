import config
from preprocess.DC_DatasetGenerator import DC_DatasetGenerator
from preprocess.PT_DatasetGenerator import PT_DatasetGenerator
from preprocess.FT_DatasetGenerator import FT_DatasetGenerator
import tensorflow as tf
import os
import pickle







class DatasetRepair:

    def __init__(self):
        self.pretraining_evals_path = '/home/ubuntu/hydra-chess/datasets/pt/mixed-eval-1mil/dataset_prob.pkl'
        with open(self.pretraining_evals_path, 'rb') as f:
            evals = pickle.load(f)
        self.evals = evals


        # --> PGN Handling
        self.pgn_file_dir = '/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/chesscom/chunks_pgn_corrupt'
        self.repair_dir = self.pgn_file_dir + '_fixed'
        if not os.path.exists(self.repair_dir):
            os.makedirs(self.repair_dir)

    def fix_corrupted_pgns(self):
        pgn_files = os.listdir(self.pgn_file_dir)
        pgn_files.sort()
        for pgn_file in pgn_files:
            self.fix_corrupted_pgn(pgn_file)
            # break


    def fix_corrupted_pgn(self, pgn_file):
        print('Fixing:', pgn_file)
        pgn_corrupt_file = os.path.join(self.pgn_file_dir, pgn_file)
        pgn_fixed_file = os.path.join(self.repair_dir, pgn_file)
        with open(pgn_corrupt_file, 'r') as f1:
            lines = f1.readlines()
        with open(pgn_fixed_file, 'w') as f2:
            building_game = False
            current_game = []
            for idx, line in enumerate(lines):
                if building_game is False:
                    if line.startswith('['):
                        building_game = True
                        current_game.append(line)
                    else:
                        continue
                else:
                    if line.startswith('1.'):
                        # Game has moves, add all lines to the file
                        for game_line in current_game:
                            f2.write(game_line)
                        f2.write(line + '\n\n')
                        current_game = []
                        building_game = False
                    elif line.strip() in ['1-0', '0-1', '1/2-1/2']:
                        # Game was resigned before started, discard all lines
                        current_game = []
                        building_game = False
                    elif line.strip() in ['*']:
                        # Game was resigned before started, discard all lines
                        print('--> Found a *')
                        current_game = []
                        building_game = False
                    else:
                        # Normal header line, add it to the current game
                        current_game.append(line)
            # Write the last game if it has moves
            if current_game and not current_game[-1].strip() in ['1-0', '0-1', '1/2-1/2']:
                for game_line in current_game:
                    f2.write(game_line)
                if current_game[-1].startswith('1.'):
                    f2.write('\n\n')
        print('Done.')






if __name__ == '__main__':
    repair = DatasetRepair()
    # repair.fix_corrupted_pgns()
