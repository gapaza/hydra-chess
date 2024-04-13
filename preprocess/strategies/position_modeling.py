import tensorflow as tf
import config
from preprocess import py_utils, tf_utils

import chess


@tf.function(input_signature=[tf.TensorSpec(shape=(None, config.seq_length), dtype=tf.int64),
                              tf.TensorSpec(shape=(None, config.seq_length), dtype=tf.int64)])
def get_piece_encoding_tf(encoded_inputs, encoded_labels):
    # print('iiiinputs:', encoded_inputs, encoded_labels)
    results = tf.py_function(get_piece_encoding, [encoded_inputs, encoded_labels],
                         Tout=[tf.int64, tf.int64, tf.int16],
    )

    encoded_inputs, encoded_labels, piece_type_encoding = results
    encoded_inputs.set_shape((config.global_batch_size, config.seq_length))
    encoded_labels.set_shape((config.global_batch_size, config.seq_length))
    piece_type_encoding.set_shape((config.global_batch_size, config.seq_length))
    return encoded_inputs, encoded_labels, piece_type_encoding

def get_piece_encoding(encoded_inputs, encoded_labels):
    # print('inputs:', encoded_inputs, encoded_labels)
    # encoded_inputs, encoded_labels = inputs
    inputs_list = encoded_inputs.numpy().tolist()
    piece_type_encoding = []

    # print('encoded inputs:', inputs_list)
    # print('encoded labels:', encoded_labels)

    for batch_game in inputs_list:
        batch_game_uci = [config.id2token[move] for move in batch_game]
        game = chess.Board()

        # get integer encoding of piece type that moved for each move
        # there are six piece types (1-6): pawn, knight, bishop, rook, queen, king
        game_piece_types = []
        for move_uci in batch_game_uci:
            try:

                if move_uci in ['', '[start]'] or move_uci in config.end_of_game_tokens:
                    game_piece_types.append(0)
                    continue

                move = chess.Move.from_uci(move_uci)
                piece = game.piece_at(move.from_square)
                if piece is not None:
                    # Map the piece type to an integer (1-6)
                    piece_type = piece.piece_type
                    # print piece name
                    # print(piece.symbol())
                else:
                    # If no piece is found, use 0 or another placeholder value
                    piece_type = 0

                game_piece_types.append(piece_type)
                game.push(move)  # Make the move on the board to update the board state
            except:
                game_piece_types.append(0)

        # print('Game piece types:', game_piece_types, min(game_piece_types), max(game_piece_types))

        piece_type_encoding.append(game_piece_types)

    return encoded_inputs, encoded_labels, tf.convert_to_tensor(piece_type_encoding, dtype=tf.int16)





if __name__ == '__main__':

    input_moves = '[start] g1f3 d7d6 c2c4 c7c6 d2d4 g7g6 b1c3 f8g7 g2g3 g8f6 f1g2 e8g8 e1g1 b8d7 b2b3 e7e5 c1b2 f8e8 e2e3 d8e7 a1c1 e5e4 f3d2 d7f8 d1c2 c8f5 f1e1 h7h5 b3b4 f8h7 b4b5 h7g5 b5c6 b7c6 c2d1 a8b8 b2a3 e7d7 c3e2 f5h3 e2f4 h3g2 g1g2 h5h4 g3h4 g5h7 h2h3 g7h6 f2f3 h6f4 e3f4 d6d5 c4d5 c6d5 a3c5 d7f5 f3e4 d5e4 d2f1 b8b2 c1c2 e8b8 e1e2 b2b1 c2c1 b1c1 d1c1 f6d5 e2f2 h7f6 f1g3 f5d7 g3f1 f6h5 f4f5 e4e3 f1e3 d5f4 g2h2 d7c7 h2g1 f4h3 g1f1 h3f2 f1f2 c7g3 f2e2 h5f4 e2d2 g3f2 d2c3 f4e2'
    labels = 'g1f3 d7d6 c2c4 c7c6 d2d4 g7g6 b1c3 f8g7 g2g3 g8f6 f1g2 e8g8 e1g1 b8d7 b2b3 e7e5 c1b2 f8e8 e2e3 d8e7 a1c1 e5e4 f3d2 d7f8 d1c2 c8f5 f1e1 h7h5 b3b4 f8h7 b4b5 h7g5 b5c6 b7c6 c2d1 a8b8 b2a3 e7d7 c3e2 f5h3 e2f4 h3g2 g1g2 h5h4 g3h4 g5h7 h2h3 g7h6 f2f3 h6f4 e3f4 d6d5 c4d5 c6d5 a3c5 d7f5 f3e4 d5e4 d2f1 b8b2 c1c2 e8b8 e1e2 b2b1 c2c1 b1c1 d1c1 f6d5 e2f2 h7f6 f1g3 f5d7 g3f1 f6h5 f4f5 e4e3 f1e3 d5f4 g2h2 d7c7 h2g1 f4h3 g1f1 h3f2 f1f2 c7g3 f2e2 h5f4 e2d2 g3f2 d2c3 f4e2 [black]'



    encoded_inputs = config.encode_tf_batch(input_moves)
    encoded_labels = config.encode_tf_batch(labels)

    encoded_inputs = tf.expand_dims(encoded_inputs, axis=0)
    encoded_labels = tf.expand_dims(encoded_labels, axis=0)

    piece_encoding = get_piece_encoding([encoded_inputs, encoded_labels])
    print(piece_encoding)



def preprocess_decoder_batch(moves):
    encoded_moves = config.encode_tf_batch(moves)

    encoded_shape = tf.shape(encoded_moves)
    batch_size = encoded_shape[0]
    seq_length = encoded_shape[1]

    start_token = tf.fill([batch_size, 1], config.start_token_id)
    encoded_inputs = tf.concat([start_token, encoded_moves], axis=1)
    encoded_inputs = encoded_inputs[:, :seq_length]

    encoded_labels = encoded_moves

    return encoded_inputs, encoded_labels






def preprocess_batch(encoded_moves):
    batch_size = tf.shape(encoded_moves)[0]


    # ---------------------
    # --- MOVE MODALITY ---
    # ---------------------
    # 1. Move sequence masked (None, 128)
    # 2. Move sequence labels (None, 128)
    # 3. Move sequence sample weights (None, 128)

    # Cast endcoded moves to int16
    encoded_moves = tf.cast(encoded_moves, tf.int16)

    # 1. Get move y labels for mask prediction
    move_seq_labels = tf.identity(encoded_moves)

    # 2. Find possible masking positions, constrain positions, and generate random mask
    # Sizes: small (3 window), medium (5 window), large (7 window)
    inp_mask = tf_utils.get_move_masking_positions_batch(encoded_moves)

    # inp_mask, mask_center = generate_random_window_small(inp_mask)   # 3 window
    # inp_mask, mask_center = generate_random_window_medium(inp_mask)  # 5 window
    # inp_mask, mask_center = generate_random_window_large(inp_mask)   # 7 window
    # inp_mask, mask_center = generate_random_window_xlarge(inp_mask)  # 9 window
    # inp_mask, mask_center = generate_random_window_xxlarge(inp_mask)  # 11 window
    inp_mask, mask_center = generate_variable_window_batch(inp_mask)




    move_seq_masked = tf_utils.apply_move_mask(encoded_moves, inp_mask)

    # 3. Create sample weights for loss function
    labels = -1 * tf.ones(tf.shape(encoded_moves), dtype=tf.int16)
    labels = tf.where(inp_mask, encoded_moves, labels)
    move_seq_sample_weights = tf.ones(tf.shape(labels), dtype=tf.int16)
    move_seq_sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), move_seq_sample_weights)


    # ----------------------
    # --- BOARD MODALITY ---
    # ----------------------
    # 1. Board tensor masked (8, 8)
    # 2. Board tensor labels (64)
    # 3. Board tensor sample weights (64)
    mask_tensor = tf.random.uniform((batch_size, 8, 8), minval=0, maxval=1) < 0.30
    masked_board, board_square_labels, board_square_weights = tf.py_function(
        py_utils.get_board_tensor_classes_at_move_flat_batch, [encoded_moves, mask_center, mask_tensor],
        [tf.int16, tf.int16, tf.int16])

    batch_size = encoded_moves.shape[0]  # this provides static shape inference
    masked_board.set_shape(tf.TensorShape([batch_size, 65]))
    board_square_labels.set_shape(tf.TensorShape([batch_size, 65]))
    board_square_weights.set_shape(tf.TensorShape([batch_size, 65]))

    return move_seq_masked, move_seq_labels, move_seq_sample_weights, masked_board, board_square_labels, board_square_weights








# """
#   _       _                             __  __              _     _
#  | |     (_)                           |  \/  |            | |   (_)
#  | |      _  _ __    ___   __ _  _ __  | \  / |  __ _  ___ | | __ _  _ __    __ _
#  | |     | || '_ \  / _ \ / _` || '__| | |\/| | / _` |/ __|| |/ /| || '_ \  / _` |
#  | |____ | || | | ||  __/| (_| || |    | |  | || (_| |\__ \|   < | || | | || (_| |
#  |______||_||_| |_| \___| \__,_||_|    |_|  |_| \__,_||___/|_|\_\|_||_| |_| \__, |
#                                                                              __/ |
#                                                                             |___/
# """

def generate_random_mask_window_linear_xxlarge(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.uniform(shape=(), maxval=maxval, dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 5
    mask_length = 11
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_center

def generate_random_mask_window_linear_xlarge(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 4
    mask_length = 9
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_center

def generate_random_mask_window_linear_large(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 3
    mask_length = 7
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_center

def generate_random_mask_window_linear_medium(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 2
    mask_length = 5
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_center

def generate_random_mask_window_linear_small(inp_mask):
    # inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 1
    mask_length = 3
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)

    return inp_mask, mask_center





# """
#   ____          _         _       __  __              _     _
#  |  _ \        | |       | |     |  \/  |            | |   (_)
#  | |_) |  __ _ | |_  ___ | |__   | \  / |  __ _  ___ | | __ _  _ __    __ _
#  |  _ <  / _` || __|/ __|| '_ \  | |\/| | / _` |/ __|| |/ /| || '_ \  / _` |
#  | |_) || (_| || |_| (__ | | | | | |  | || (_| |\__ \|   < | || | | || (_| |
#  |____/  \__,_| \__|\___||_| |_| |_|  |_| \__,_||___/|_|\_\|_||_| |_| \__, |
#                                                                        __/ |
#                                                                       |___/
# """

def generate_variable_window_batch(inp_mask):
    # Generate random integers and keep track of original indices
    random_ints = tf.random.uniform(shape=(tf.shape(inp_mask)[0],), minval=0, maxval=4, dtype=tf.int32)
    original_indices = tf.range(tf.shape(inp_mask)[0])

    # Split the batch and keep track of original indices
    def split_with_indices(mask, idx):
        split_mask = tf.boolean_mask(mask, tf.equal(random_ints, idx))
        split_indices = tf.boolean_mask(original_indices, tf.equal(random_ints, idx))
        return split_mask, split_indices

    inp_mask_xsmall, indices_xsmall = split_with_indices(inp_mask, 0)      # 1 masking window
    inp_mask_small, indices_small = split_with_indices(inp_mask, 1)      # 3 masking window
    inp_mask_medium, indices_medium = split_with_indices(inp_mask, 2)    # 5 masking window
    inp_mask_large, indices_large = split_with_indices(inp_mask, 3)      # 7 masking window
    # inp_mask_xlarge, indices_xlarge = split_with_indices(inp_mask, 4)    # 9 masking window
    # inp_mask_xxlarge, indices_xxlarge = split_with_indices(inp_mask, 5)  # 11 masking window

    # Apply the functions to each part
    mask_window_xsmall, mask_center_xsmall = generate_random_window_xsmall(inp_mask_xsmall)
    mask_window_small, mask_center_small = generate_random_window_small(inp_mask_small)
    mask_window_medium, mask_center_medium = generate_random_window_medium(inp_mask_medium)
    mask_window_large, mask_center_large = generate_random_window_large(inp_mask_large)
    # mask_window_xlarge, mask_center_xlarge = generate_random_window_xlarge(inp_mask_xlarge)
    # mask_window_xxlarge, mask_center_xxlarge = generate_random_window_xxlarge(inp_mask_xxlarge)

    # Concatenate the results back together along with the original indices
    mask_window_list = [mask_window_xsmall, mask_window_small, mask_window_medium, mask_window_large]
    mask_center_list = [mask_center_xsmall, mask_center_small, mask_center_medium, mask_center_large]
    indices_list = [indices_xsmall, indices_small, indices_medium, indices_large]

    mask_window_concat = tf.concat(mask_window_list, axis=0)
    mask_center_concat = tf.concat(mask_center_list, axis=0)
    indices_concat = tf.concat(indices_list, axis=0)

    # Sort by the original indices
    sort_order = tf.argsort(indices_concat)
    mask_window = tf.gather(mask_window_concat, sort_order)
    mask_center = tf.gather(mask_center_concat, sort_order)

    return mask_window, mask_center





def generate_random_window_xxlarge(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center - 5,
        mask_center - 4,
        mask_center - 3,
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3,
        mask_center + 4,
        mask_center + 5,
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center

def generate_random_window_xlarge(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center - 4,
        mask_center - 3,
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3,
        mask_center + 4
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center

def generate_random_window_large(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center - 3,
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center

def generate_random_window_medium(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center

def generate_random_window_small(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center - 1, mask_center, mask_center + 1
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center



def generate_random_window_xsmall(inp_mask):
    # inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_window_indices = tf.stack([
        mask_center
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center




