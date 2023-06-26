import os
import chess.pgn
import pickle
import itertools
import tqdm
import config


def parse_all_tokens(game_file, max_games=1000000, save=True):
    possible_moves = set()
    with open(game_file, encoding='utf-8', errors='replace') as pgn_file:
        game_iterator = iter(lambda: chess.pgn.read_game(pgn_file), None)
        for game in itertools.islice(game_iterator, max_games):
            try:
                all_moves = game.mainline_moves()
                if not any(True for _ in all_moves):
                    continue
                for move in all_moves:
                    move_uci = move.uci()
                    possible_moves.add(move_uci)
            except ValueError as e:
                print(e)
                continue
    if save:
        token_file = os.path.join(config.tokens_dir, 'tokens_' + str(len(possible_moves)) + '.pkl')
        with open(token_file, 'wb') as f:
            pickle.dump(possible_moves, f)
    return possible_moves

def parse_position_tokens():
    unique_moves = set()
    # iterate over files in config.positions_file_dir
    for file in os.listdir(config.positions_load_dir):
        print('Parsing file', file)
        file_path = os.path.join(config.positions_load_dir, file)
        with open(file_path, 'rb') as games_file:
            games = pickle.load(games_file)
            for game in tqdm.tqdm(games):
                move_str = game['moves']
                game_moves = move_str.split(' ')
                unique_moves.update(game_moves)
            print(len(games))
    if '' in unique_moves:
        unique_moves.remove('')
    print(unique_moves)
    token_file = os.path.join(config.tokens_dir, 'tokens_' + str(len(unique_moves)) + '_chesscom.pkl')
    with open(token_file, 'wb') as f:
        pickle.dump(unique_moves, f)
    return unique_moves



def compare_tokens():
    f1 = os.path.join(config.tokens_dir, 'tokens_1966.pkl')
    f2 = os.path.join(config.tokens_dir, 'tokens_1968_chesscom.pkl')
    with open(f1, 'rb') as f:
        tokens1 = pickle.load(f)
    with open(f2, 'rb') as f:
        tokens2 = pickle.load(f)

    unique_tokens_1 = tokens1 - tokens2
    unique_tokens_2 = tokens2 - tokens1
    print('Unique tokens 1:', unique_tokens_1)
    print('Unique tokens 2:', unique_tokens_2)

    # merge token sets
    tokens1.update(tokens2)
    print('Merged tokens:', len(tokens1))

    # save merged tokens
    token_file = os.path.join(config.tokens_dir, 'tokens_' + str(len(tokens1)) + '_merged.pkl')
    with open(token_file, 'wb') as f:
        pickle.dump(tokens1, f)








if __name__ == '__main__':
    compare_tokens()
    # parse_position_tokens()
    # temp_file = os.path.join(config.games_file_dir, 'pgn_chunk_0_100000.pgn')
    # parse_all_tokens(config.games_file, max_games=1000000)