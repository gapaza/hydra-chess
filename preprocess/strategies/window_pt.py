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

    # Cast endcoded moves to int16
    encoded_moves = tf.cast(encoded_moves, tf.int16)

    # 1. Get move y labels for mask prediction
    move_seq_labels = tf.identity(encoded_moves)

    # 2. Find possible masking positions, constrain positions, and generate random mask
    # Sizes: small (3 window), medium (5 window), large (7 window)
    inp_mask = tf_utils.get_move_masking_positions_batch(encoded_moves)

    # inp_mask, mask_center = generate_random_window_small(inp_mask)   # 3 window
    # inp_mask, mask_center = generate_random_window_medium(inp_mask)  # 5 window
    inp_mask, mask_center = generate_random_window_large(inp_mask)   # 7 window
    # inp_mask, mask_center = generate_random_window_xlarge(inp_mask)  # 9 window
    # inp_mask, mask_center = generate_random_window_xxlarge(inp_mask)  # 11 window
    # inp_mask, mask_center = randomize_window_masking_linear(inp_mask)



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
    masked_board.set_shape(tf.TensorShape([batch_size, 8, 8]))
    board_square_labels.set_shape(tf.TensorShape([batch_size, 64]))
    board_square_weights.set_shape(tf.TensorShape([batch_size, 64]))

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


def randomize_window_masking_linear(inp_mask):
    mask_strategy = tf.random.uniform(
        shape=(tf.shape(inp_mask)[0],),
        minval=0, maxval=4, dtype=tf.int32
    )  # Generates integers for strategies: small, medium, large, xlarge, xxlarge
    inp_mask, mask_center = tf.vectorized_map(
        lambda x: tf.case([
            (x[0] == 0, lambda: generate_random_mask_window_linear_small(x[1])),
            (x[0] == 1, lambda: generate_random_mask_window_linear_medium(x[1])),
            (x[0] == 2, lambda: generate_random_mask_window_linear_large(x[1])),
            (x[0] == 3, lambda: generate_random_mask_window_linear_xlarge(x[1])),
            (x[0] == 4, lambda: generate_random_mask_window_linear_xxlarge(x[1]))
        ], exclusive=True),
        (mask_strategy, inp_mask)
    )
    return inp_mask, mask_center


def generate_random_mask_window_linear_xxlarge(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    seed = tf.constant([42, 42], dtype=tf.int32)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32, seed=seed)
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
    seed = tf.constant([42, 42], dtype=tf.int32)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32, seed=seed)
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
    seed = tf.constant([42, 42], dtype=tf.int32)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32, seed=seed)
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
    seed = tf.constant([42, 42], dtype=tf.int32)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32, seed=seed)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 2
    mask_length = 5
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_center

def generate_random_mask_window_linear_small(inp_mask):
    inp_mask = tf_utils.constrain_move_mask_window_positions(inp_mask)
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    seed = tf.constant([42, 42], dtype=tf.int32)
    maxval = tf.shape(true_indices)[-1]
    rand_idx = tf.random.stateless_uniform(shape=(), maxval=maxval, dtype=tf.int32, seed=seed)
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
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]),
                                axis=-2)

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




