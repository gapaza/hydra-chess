import tensorflow as tf
import config
from preprocess.strategies import tf_utils
from preprocess.strategies import py_utils


def preprocess_batch(encoded_moves):
    batch_size = tf.shape(encoded_moves)[0]

    # ---------------------
    # --- MOVE MODALITY ---
    # ---------------------
    # 1. Move sequence masked (None, 128)
    # 2. Move sequence labels (None, 128)
    # 3. Move sequence sample weights (None, 128)

    # Cast encoded moves to int16
    encoded_moves = tf.cast(encoded_moves, tf.int16)

    # 1. Get move y labels for mask prediction
    move_seq_labels = tf.identity(encoded_moves)

    # 2. Find possible masking positions, constrain positions twice for a max length of 7, and generate random mask
    inp_mask = tf_utils.get_move_masking_positions_batch(encoded_moves)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)
    inp_mask, mask_start = generate_random_denoising_prediction(inp_mask)
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
        py_utils.get_board_tensor_classes_at_move_flat_batch, [encoded_moves, mask_start, mask_tensor],
        [tf.int16, tf.int16, tf.int16])

    batch_size = encoded_moves.shape[0]  # this provides static shape inference
    masked_board.set_shape(tf.TensorShape([batch_size, 8, 8]))
    board_square_labels.set_shape(tf.TensorShape([batch_size, 64]))
    board_square_weights.set_shape(tf.TensorShape([batch_size, 64]))

    return move_seq_masked, move_seq_labels, move_seq_sample_weights, masked_board, board_square_labels, board_square_weights








def generate_random_denoising_prediction(inp_mask):
    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    seed = tf.constant([42, 75], dtype=tf.int32)
    rand_idx = tf.map_fn(lambda x: tf.random.stateless_uniform(shape=(), maxval=x, dtype=tf.int32, seed=seed),
                         true_counts)

    # Create a tensor of shape (batch, 7) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_start = mask_center - 3

    # Randomize masking strategy
    mask_strategy = tf.random.uniform(
        shape=(tf.shape(inp_mask)[0],),
        minval=0, maxval=3, dtype=tf.int32
    )  # Generates integers 0, 1, 2
    mask_window_indices = tf.vectorized_map(
        lambda x: tf.case([
            (x[0] == 0, lambda: strategy_2(x[1])),
            (x[0] == 1, lambda: strategy_3(x[1])),
            (x[0] == 2, lambda: strategy_4(x[1]))
        ], exclusive=True),
        (mask_strategy, mask_center)
    )

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(
        tf.equal(tf.expand_dims(mask_range, axis=0),
        mask_window_indices[..., tf.newaxis]),
        axis=-2
    )

    # Randomly select board position to return
    # board_selection = tf.random.uniform(shape=(), minval=0, maxval=7, dtype=tf.int32)
    # board_position = tf.case([
    #     (board_selection[0] == 0, lambda: mask_center - 3),
    #     (board_selection[0] == 1, lambda: mask_center - 2),
    #     (board_selection[0] == 2, lambda: mask_center - 1),
    #     (board_selection[0] == 3, lambda: mask_center),
    #     (board_selection[0] == 4, lambda: mask_center + 1),
    #     (board_selection[0] == 5, lambda: mask_center + 2),
    #     (board_selection[0] == 6, lambda: mask_center + 3)
    # ], exclusive=True)

    return mask_window, mask_start








""" Masking Strategies
    - Board position: always given at first mask token: mask_center - 3
    - Current player: defined as player who's turn it is at the board position

    Strategy 1: mask all tokens
    Strategy 2: fixed start position, random length
    Strategy 3: mask 3 turns of current player
    Strategy 4: mask 4 turns of opponent player
    
    Strategy 5: randomly mask all tokens (not considered)
"""

def strategy_1(mask_center):
    return tf.stack([
        mask_center - 3,  # Board Position
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3
    ], axis=-1)

def strategy_2(mask_center):
    positions = tf.stack([
        mask_center - 3,  # Board Position
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3
    ], axis=-1)

    # Generate a random length up to a maximum of 6 (the total length of positions minus the fixed starting position)
    mask_length = tf.random.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)

    # Mask
    mask = tf.sequence_mask(mask_length, maxlen=7)
    mask = tf.pad(mask, [[0, 7 - tf.shape(mask)[0]]], constant_values=False)

    # Apply the mask, setting unmasked values to -1
    masked_positions = tf.where(mask, positions, -1)
    return masked_positions


def strategy_3(mask_center):
    return tf.stack([
        mask_center - 2,
        mask_center,
        mask_center + 2,
        -1, -1, -1, -1
    ], axis=-1)

def strategy_4(mask_center):
    return tf.stack([
        mask_center - 3,
        mask_center - 1,
        mask_center + 1,
        mask_center + 3,
        -1, -1, -1
    ], axis=-1)

def strategy_5(mask_center):
    window = tf.stack([
        mask_center - 3,  # Board Position
        mask_center - 2,
        mask_center - 1,
        mask_center,
        mask_center + 1,
        mask_center + 2,
        mask_center + 3
    ], axis=-1)
    mask_rate = 0.5
    mask = tf.random.uniform(shape=tf.shape(window), minval=0, maxval=1) > mask_rate  # Adjust the threshold as needed
    masked_window = tf.where(mask, window, -1)
    return masked_window