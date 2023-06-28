import chess
import numpy as np
import config


def get_board_tensor_at_move(move_tokens, move_idx):
    move_idx = move_idx.numpy()
    moves = [config.id2token[token_id] for token_id in move_tokens.numpy()]
    board = chess.Board()
    try:
        for i in range(move_idx+1):
            move = moves[i]
            if move == '[pos]':
                continue
            if move in ['[mask]', '']:
                break
            board.push_uci(move)
    except Exception as e:
        print('--> INVALID MOVE', e)
    return board_to_tensor(board)

def get_sequence_board_tensor(move_tokens):
    moves = [config.id2token[token_id] for token_id in move_tokens.numpy()]
    board = chess.Board()
    for move in moves:
        if move in ['', '[mask]']:
            break
        try:
            board.push_uci(move)
        except Exception as e:
            break
    return board_to_tensor(board)

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        piece_index = piece_to_index(piece)
        tensor[7 - rank, file, piece_index] = 1
    return tensor

def piece_to_index(piece):
    piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
    index = piece_order.index(piece.symbol().upper())
    # If the piece is black, add 6 to the index (to cover the range 0-11)
    if piece.color == chess.BLACK:
        index += 6
    return index

