import chess
import numpy as np
import config






##########################
### Old Board Encoding ###
##########################

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












##########################
### New Board Encoding ###
##########################


def get_sequence_board_tensor_classes(move_tokens):
    moves = [config.id2token[token_id] for token_id in move_tokens.numpy()]
    board = chess.Board()
    for move in moves:
        if move in ['', '[mask]']:
            break
        try:
            board.push_uci(move)
        except Exception as e:
            break
    return board_to_tensor_classes_no_mask(board)


def board_to_tensor_classes_no_mask(board):
    tensor = np.zeros((8, 8, 14))
    squares_used = np.full((8, 8), False)

    # 1. Add pieces, build labels
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        flipped_rank = flip_rank(rank)
        piece_class = piece_to_class(piece)
        tensor[flipped_rank, file, piece_class] = 1
        squares_used[flipped_rank, file] = True

    # 2. Account for empty squares
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if not squares_used[flipped_rank, file]:
                tensor[flipped_rank, file, 0] = 1

    return tensor








def get_board_tensor_classes_at_move(move_tokens, move_idx, board_mask):
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
        exception = e
    return board_to_tensor_classes(board, board_mask)

def board_to_tensor_classes(board, board_mask):
    board_mask = board_mask.numpy()
    tensor = np.zeros((8, 8, 14))

    # Progress Tensor
    squares_used = np.full((8, 8), False)

    # 1. Add mask, build weights
    weights = [0] * 64
    labels = [0] * 64
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if board_mask[flipped_rank, file]:
                tensor[flipped_rank, file, 13] = 1
                squares_used[flipped_rank, file] = True
                pos = flipped_rank * 8 + file
                weights[pos] = 1

    # 2. Add pieces, build labels
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        flipped_rank = flip_rank(rank)
        piece_class = piece_to_class(piece)
        pos = flipped_rank * 8 + file
        labels[pos] = piece_class
        if squares_used[flipped_rank, file]:
            continue  # means square is masked
        tensor[flipped_rank, file, piece_class] = 1
        squares_used[flipped_rank, file] = True

    # 3. Account for empty squares
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if not squares_used[flipped_rank, file]:
                tensor[flipped_rank, file, 0] = 1

    return tensor, labels, weights




###########################
### Flat Board Encoding ###
###########################
### Square Classes
# 0: Empty
# 1-6: (White) Pawn, Knight, Bishop, Rook, Queen, King
# 7-12: (Black) Pawn, Knight, Bishop, Rook, Queen, King
# 13: Mask

def get_board_tensor_classes_at_move_flat(move_tokens, move_idx, board_mask):
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
        exception = e
    return board_to_tensor_classes_flat(board, board_mask)


def board_to_tensor_classes_flat(board, board_mask):
    board_mask = board_mask.numpy()
    tensor = np.zeros((8, 8))

    # Progress Tensor
    squares_used = np.full((8, 8), False)

    # 1. Add mask, build weights
    weights = [0] * 64
    labels = [0] * 64
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if board_mask[flipped_rank, file]:
                tensor[flipped_rank, file] = 13
                squares_used[flipped_rank, file] = True
                pos = flipped_rank * 8 + file
                weights[pos] = 1

    # 2. Add pieces, build labels
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        flipped_rank = flip_rank(rank)
        piece_class = piece_to_class(piece)
        pos = flipped_rank * 8 + file
        labels[pos] = piece_class
        if squares_used[flipped_rank, file]:
            continue  # means square is masked
        tensor[flipped_rank, file] = piece_class
        squares_used[flipped_rank, file] = True

    # 3. Account for empty squares
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if not squares_used[flipped_rank, file]:
                tensor[flipped_rank, file] = 0

    return tensor, labels, weights







def get_sequence_board_tensor_classes_flat(move_tokens):
    moves = [config.id2token[token_id] for token_id in move_tokens.numpy()]
    board = chess.Board()
    for move in moves:
        if move in ['', '[mask]']:
            break
        try:
            board.push_uci(move)
        except Exception as e:
            break
    return board_to_tensor_classes_no_mask_flat(board)


def board_to_tensor_classes_no_mask_flat(board):
    tensor = np.zeros((8, 8))
    squares_used = np.full((8, 8), False)

    # 1. Add pieces, build labels
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        flipped_rank = flip_rank(rank)
        piece_class = piece_to_class(piece)
        tensor[flipped_rank, file] = piece_class
        squares_used[flipped_rank, file] = True

    # 2. Account for empty squares
    for rank in range(8):
        for file in range(8):
            flipped_rank = flip_rank(rank)
            if not squares_used[flipped_rank, file]:
                tensor[flipped_rank, file] = 0

    return tensor










########################
### Helper Functions ###
########################

def piece_to_class(piece):
    piece_order = ['P', 'N', 'B', 'R', 'Q', 'K']
    index = piece_order.index(piece.symbol().upper()) + 1  # 0 is reserved for empty square
    # If the piece is black, add 6 to the index (to cover the range 0-11)
    if piece.color == chess.BLACK:
        index += 6
    return index


def flip_rank(rank):
    return 7 - rank



