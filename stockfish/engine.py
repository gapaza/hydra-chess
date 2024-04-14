import chess
import chess.pgn as chess_pgn
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures
import json
import os
import chess.engine
from chess.engine import PovScore
import pickle
import time
import config

def test_engine():
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(config.stockfish_path)

    # Set the maximum number of games to process
    max_games = 1

    # Open the PGN file
    with open(config.games_file) as pgn_file:
        progress_bar = tqdm(range(max_games), desc="Processing games", unit="game")
        for _ in progress_bar:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_moves = list(move for move in game.mainline_moves())
            game_moves_uci = list(move.uci() for move in game_moves)

            # select a random move in the middle of the game
            random_move = np.random.randint(5, len(game_moves))

            board = game.board()
            for x in range(random_move):
                move = game_moves[x]
                board.push(move)
            print('Analyzing board')
            info = engine.analyse(board, chess.engine.Limit(time=5), multipv=5)
            print('Done analyzing board', info)
            for idx, variation in enumerate(info, start=1):
                move = variation.get("pv")[0]
                score = variation.get("score")
                if board.turn == chess.BLACK:
                    score = score.relative.score()
                print('Turn: ', board.turn) # true
                print(f"{idx}. {move} ({score}) {random_move}")

    engine.quit()


def analyze_position(board, depth=20, multipv=5):
    with chess.engine.SimpleEngine.popen_uci(config.stockfish_path) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        top_5_moves = []
        for idx, variation in enumerate(info, start=1):
            top_move = variation.get("pv")[0]
            score = variation.get("score")
            if board.turn == chess.BLACK:
                score = score.black().score()
            elif board.turn == chess.WHITE:
                score = score.white().score()
            top_5_moves.append({
                'move': top_move.uci(),
                'evaluation': score
            })
    return top_5_moves


def process_game_chunk(game_chunk):
    chunk_evaluation_data = []
    for game in game_chunk:
        moves = list(move.uci() for move in game.mainline_moves())
        position_data = []
        board = chess.Board()
        # rand_move = np.random.randint(1, len(moves))
        for idx1, move in enumerate(moves):
            print('Analyzing board: ', board.move_stack)
            # if idx1 == rand_move:
            top_5_moves = analyze_position(board.copy())
            position_data.append({
                'position': idx1,
                'top_5_moves': top_5_moves
            })
            board.push_uci(move)

        chunk_evaluation_data.append({
            'game_moves': moves,
            'position_data': position_data
        })
    return chunk_evaluation_data

def generate_stockfish_data_fast(games, num_workers=12):
    chunk_size = len(games) // num_workers
    game_chunks = [games[i:i + chunk_size] for i in range(0, len(games), chunk_size)]

    # --> Start workers
    evaluation_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_game_chunk, game_chunk) for game_chunk in game_chunks]
        for future in concurrent.futures.as_completed(futures):
            evaluation_data.extend(future.result())

    # --> Save data
    time_stamp = time.time()
    with open(os.path.join(config.evaluations_dir, 'stockfish_data_'+str(len(evaluation_data))+'_'+str(time_stamp)+'.pkl'), 'wb') as f:
        pickle.dump(evaluation_data, f)
    with open(os.path.join(config.evaluations_dir, 'stockfish_data_'+str(len(evaluation_data))+'_'+str(time_stamp)+'.json'), 'w') as f:
        json.dump(evaluation_data, f)
    return evaluation_data

















