import chess
import chess.pgn as chess_pgn
import numpy as np
import re
from tqdm import tqdm
import concurrent.futures
import json
import random
import os
import chess.engine
from chess.engine import PovScore
import pickle
import time
import config
import multiprocessing

# Import ORM, then tables
# from stockfishapi.stockfishapi.models import Evaluations




class StockfishAPI(multiprocessing.Process):

    def __init__(self, pgn_file, mode='evaluate', threads=8):
        multiprocessing.Process.__init__(self)
        self.mode = mode
        self.pgn_file = pgn_file
        self.self_play_pgn = os.path.join(config.games_dir, 'self_play.pgn')
        self.threads = threads

        # --> Limits
        self.nodes = 100000
        self.lines = 1

        # --> Database
        # self.db = DatabaseInterface()


    def __del__(self):
        self.engine.quit()


    def run(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': self.threads, "Hash": 1024})
        if self.mode == 'evaluate':
            self.evaluate()
        elif self.mode == 'self_play':
            self.self_play()

    def run_local(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)
        self.engine.configure({'Threads': self.threads, "Hash": 1024})
        self.evaluate()



    def self_play(self):
        # Create a new chess board + game
        game = chess.pgn.Game()
        node = game
        board = chess.Board()

        # Game loop
        uci_moves = []
        while not board.is_game_over():
            white_turn = (board.turn == chess.WHITE)

            # Determine if position has already been evaluated, play eval if so
            entry_id = self.db.fen_exists(board.fen(), white_turn)
            if type(entry_id) != int:
                move = entry_id.played_move
                board.push_uci(move)
                continue

            analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=1)

            top_moves, mate_exists, absolute_eval = self.extract_top_moves(analysis, white_turn)
            top_line = self.extract_top_line(analysis)

            move = analysis['pv'][0]
            uci_moves.append(move.uci())

            self.db.update_entry(entry_id, top_moves, top_line, move.uci(), mate_exists, absolute_eval)

            # Make the move on the board
            node = node.add_variation(move)
            board.push(move)

        # Game is over, print the result
        result = None
        if board.is_checkmate():
            result = 'checkmate: ' + ('white' if not board.turn else 'black') + ' wins'
        elif board.is_stalemate():
            result = 'stalemate'
        elif board.is_insufficient_material():
            result = 'insufficient material'
        elif board.is_seventyfive_moves():
            result = '75 moves rule'
        elif board.is_fivefold_repetition():
            result = 'fivefold repetition'
        elif board.is_variant_draw():
            result = 'variant draw'
        else:
            result = 'draw due to unknown reason'

        print(uci_moves)
        print(result)

        # Export the game to a PGN file
        with open(self.self_play_pgn, 'w') as f:
            print(game, file=f)

        return board, uci_moves


    def evaluate(self, max_games=None):
        print('--> EVAL PROCESS')
        pgn_file_obj = open(self.pgn_file)
        game = chess.pgn.read_game(pgn_file_obj)
        game_counter = 0
        # new tqdm progress bar
        # progress_bar = tqdm(total=100000)
        while game:

            # Index game
            game_moves = list(move.uci() for move in game.mainline_moves())
            game_result = None
            result = game.headers["Result"]
            if result == '1-0':
                game_result = '[white]'
            elif result == '0-1':
                game_result = '[black]'
            elif result == '1/2-1/2':
                game_result = '[draw]'
            game_id = self.db.get_or_add_game(game_moves, game_result)

            # 1. Evaluate game
            self.evaluate_game(game, game_id)

            # 2. Read next game
            game = chess.pgn.read_game(pgn_file_obj)

            # 3. Increment counter
            game_counter += 1
            # progress_bar.update(1)
            if max_games is not None and game_counter >= max_games:
                break

        print('Evaluation Finished...')



    def get_non_duplicate_index(self, game_id, low_bound, up_bound, threshold=10, segment_token=''):
        if up_bound < low_bound:  # If the range is empty, return None
            print('--> Empty range', low_bound, up_bound)
            return -1
        # First, check to see if game segment eval already exists
        if self.db.game_segment_eval_exists(game_id, segment_token):
            return -1
        for _ in range(threshold):
            idx = random.randint(low_bound, up_bound)
            if not self.db.game_move_eval_exists(game_id, idx):
                return idx
        print('--> Could not find a non-duplicate index')
        return -1

    def evaluate_game(self, game, game_id):
        new_positions_evaluated = 0
        duplicate_positions = 0

        # 1. Get moves and fresh board
        moves = list(move for move in game.mainline_moves())
        board = game.board()

        # Assume 3 phases of the game: opening, middlegame, endgame and 128 move limit
        total_moves = min(len(moves), 128)
        if total_moves < 13:
            return

        # Bounds allowing for 9 len masking window (chop off 5 moves from each end)
        if len(moves) >= 128:
            lower_bound = 5
            first_third = 44
            second_third = 83
            final_third = 122
        else:
            reduced_len = len(moves) - 10
            lower_bound = 5
            first_third = (reduced_len // 3) + 5
            second_third = (2 * reduced_len // 3) + 5
            final_third = reduced_len + 5


        # Randomly select an index from each phase of the game
        threshold = 10
        opening_index = self.get_non_duplicate_index(game_id, lower_bound, first_third, threshold, '[opening]')
        middlegame_index = self.get_non_duplicate_index(game_id, first_third, second_third, threshold, '[middlegame]')
        endgame_index = self.get_non_duplicate_index(game_id, second_third, final_third, threshold, '[endgame]')

        selected_indices = [opening_index, middlegame_index, endgame_index]

        # 2. Iterate over moves
        for idx, move in enumerate(moves):

            # Skip to the next move if this move is not one of the selected indices
            if idx not in selected_indices:
                board.push(move)
                continue

            segment_token = ''
            if idx == opening_index:
                segment_token = '[opening]'
            elif idx == middlegame_index:
                segment_token = '[middlegame]'
            elif idx == endgame_index:
                segment_token = '[endgame]'

            white_turn = (board.turn == chess.WHITE)
            curr_fen = board.fen()
            played_move_uci = move.uci()

            # 3. Check if position has already been evaluated: if entry_id is an int, continue
            entry_id = self.db.fen_exists(curr_fen, white_turn)
            if type(entry_id) != int:
                # Add GameEntry for current game
                self.db.add_game_move_evaluation(game_id, entry_id.id, idx, segment_token)
                duplicate_positions += 1
                board.push(move)
                continue
            else:
                new_positions_evaluated += 1

            # 4. Evaluate position
            try:
                analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)
                top_moves, mate_exists, absolute_eval = self.extract_top_moves(analysis, white_turn)
                top_line = self.extract_top_line(analysis)

                # 5. Save to database
                self.db.update_entry(entry_id, top_moves, top_line, played_move_uci, mate_exists, absolute_eval)
                self.db.add_game_move_evaluation(game_id, entry_id, idx, segment_token)
            except Exception as e:
                exception = e

            # 6. Push played move and continue
            board.push(move)


    # TODO: add game table functionality and join tables
    def evaluate_entire_game(self, game, game_id):
        new_positions_evaluated = 0
        duplicate_positions = 0

        # 1. Get moves and fresh board
        moves = list(move for move in game.mainline_moves())
        board = game.board()

        # 2. Iterate over moves
        for idx, move in enumerate(moves):

            white_turn = (board.turn == chess.WHITE)
            curr_fen = board.fen()
            played_move_uci = move.uci()

            # 3. Check if position has already been evaluated: if entry_id is an int, continue
            entry_id = self.db.fen_exists(curr_fen, white_turn)
            if type(entry_id) != int:
                duplicate_positions += 1
                board.push(move)
                continue
            else:
                new_positions_evaluated += 1

            # 4. Evaluate position
            analysis = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=self.lines)
            try:
                top_moves, mate_exists, absolute_eval = self.extract_top_moves(analysis, white_turn)
                top_line = self.extract_top_line(analysis)

                # 5. Save to database
                self.db.update_entry(entry_id, top_moves, top_line, played_move_uci, mate_exists, absolute_eval)
            except Exception as e:
                print(e, analysis)

            # 6. Push played move and continue
            board.push(move)

        # print('\nNew positions evaluated: ', new_positions_evaluated)
        # print('Duplicate positions: ', duplicate_positions)


    def extract_top_moves(self, analysis, white_turn):
        top_moves = []
        delivering_mate = False
        absolute_eval = 0
        for idx, line in enumerate(analysis):
            line_top_move = line["pv"][0].uci()

            # Relative Score
            if line["score"].relative.is_mate():
                if line["score"].relative.mate() > 0:
                    line_top_move_score = 10000  # use a very high value
                    delivering_mate = True
                else:
                    line_top_move_score = -10000  # use a very low value
            else:
                line_top_move_score = line["score"].relative.score()

            # Absolute Score
            if idx == 0:
                if line["score"].relative.is_mate():
                    if white_turn is True:
                        if line["score"].relative.mate() > 0:
                            absolute_eval = 10000
                        else:
                            absolute_eval = -10000
                    else:
                        if line["score"].relative.mate() > 0:
                            absolute_eval = -10000
                        else:
                            absolute_eval = 10000
                else:
                    absolute_eval = line["score"].white().score()

            top_moves.append((line_top_move, line_top_move_score))
        return top_moves, delivering_mate, absolute_eval

    def extract_top_line(self, analysis):
        top_line = analysis[0]["pv"]
        top_line = [move.uci() for move in top_line]
        return top_line





if __name__ == '__main__':
    api = StockfishAPI(config.games_file)
    api.start()
    # api.self_play()




