import tensorflow as tf
import config
from preprocess.strategies import tf_utils
from preprocess.strategies import py_utils



def encode_batch(move_sequences, masking_indices, probability_scores):
    encoded_moves = config.encode_tf_batch(move_sequences)
    return encoded_moves, masking_indices, probability_scores



def preprocess_batch(encoded_moves, masking_indices, probability_scores):
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
    inp_mask, mask_center, win_probs = generate_variable_window_batch(inp_mask, masking_indices, probability_scores)
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
    mask_tensor = tf.random.uniform((batch_size, 8, 8), minval=0, maxval=1) < 0.15
    masked_board, board_square_labels, board_square_weights = tf.py_function(
        py_utils.get_board_tensor_classes_at_move_flat_batch, [encoded_moves, mask_center, mask_tensor],
        [tf.int16, tf.int16, tf.int16])

    batch_size = encoded_moves.shape[0]  # this provides static shape inference
    masked_board.set_shape(tf.TensorShape([batch_size, 65]))
    board_square_labels.set_shape(tf.TensorShape([batch_size, 65]))
    board_square_weights.set_shape(tf.TensorShape([batch_size, 65]))

    return move_seq_masked, move_seq_labels, move_seq_sample_weights, masked_board, board_square_labels, board_square_weights, win_probs















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

def generate_variable_window_batch(inp_mask, idx_positions, idx_win_probs):
    # Generate random integers and keep track of original indices
    random_ints = tf.random.uniform(shape=(tf.shape(inp_mask)[0],), minval=0, maxval=2, dtype=tf.int32)
    original_indices = tf.range(tf.shape(inp_mask)[0])

    # Split the batch and keep track of original indices
    def split_with_indices(mask, positions, win_probs_passed, idx):
        split_mask = tf.boolean_mask(mask, tf.equal(random_ints, idx))
        split_positions = tf.boolean_mask(positions, tf.equal(random_ints, idx))
        split_win_probs = tf.boolean_mask(win_probs_passed, tf.equal(random_ints, idx))
        split_indices = tf.boolean_mask(original_indices, tf.equal(random_ints, idx))
        return split_mask, split_positions, split_win_probs, split_indices

    inp_mask_small, positions_small, win_probs_small, indices_small = split_with_indices(inp_mask, idx_positions, idx_win_probs, 0)      # 3 masking window
    inp_mask_xsmall, positions_xsmall, win_probs_xsmall, indices_xsmall = split_with_indices(inp_mask, idx_positions, idx_win_probs, 1)    # 1 masking window
    # inp_mask_medium, indices_medium = split_with_indices(inp_mask, idx_positions, idx_win_probs, 1)    # 5 masking window
    # inp_mask_large, indices_large = split_with_indices(inp_mask, idx_positions, idx_win_probs, 2)      # 7 masking window
    # inp_mask_xlarge, indices_xlarge = split_with_indices(inp_mask, idx_positions, idx_win_probs, 3)    # 9 masking window
    # inp_mask_xxlarge, indices_xxlarge = split_with_indices(inp_mask, idx_positions, idx_win_probs, 4)  # 11 masking window

    # Apply the functions to each part
    mask_window_xsmall, mask_center_xsmall, win_probs_xsmall = generate_random_window_xsmall_eval(inp_mask_xsmall, positions_xsmall, win_probs_xsmall)
    mask_window_small, mask_center_small, win_probs_small = generate_random_window_small_eval(inp_mask_small, positions_small, win_probs_small)
    # mask_window_medium, mask_center_medium = generate_random_window_medium(inp_mask_medium)
    # mask_window_large, mask_center_large = generate_random_window_large(inp_mask_large)
    # mask_window_xlarge, mask_center_xlarge = generate_random_window_xlarge(inp_mask_xlarge)
    # mask_window_xxlarge, mask_center_xxlarge = generate_random_window_xxlarge(inp_mask_xxlarge)

    # Concatenate the results back together along with the original indices
    mask_window_list = [mask_window_xsmall, mask_window_small]
    mask_center_list = [mask_center_xsmall, mask_center_small]
    win_probs_list = [win_probs_xsmall, win_probs_small]
    indices_list = [indices_xsmall, indices_small]

    mask_window_concat = tf.concat(mask_window_list, axis=0)
    mask_center_concat = tf.concat(mask_center_list, axis=0)
    win_probs_concat = tf.concat(win_probs_list, axis=0)
    indices_concat = tf.concat(indices_list, axis=0)

    # Sort by the original indices
    sort_order = tf.argsort(indices_concat)
    mask_window = tf.gather(mask_window_concat, sort_order)
    mask_center = tf.gather(mask_center_concat, sort_order)
    win_probs = tf.gather(win_probs_concat, sort_order)

    return mask_window, mask_center, win_probs











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




def generate_random_window_small_eval(inp_mask, idx_positions, idx_win_probs):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Generate random indices within the counts of True values for each batch element.
    batch_size = tf.shape(inp_mask)[0]
    true_counts = tf.fill([batch_size], 3)
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Select mask center from the idx_options.
    mask_center = tf.gather(idx_positions, rand_idx, batch_dims=1)
    win_probs = tf.gather(idx_win_probs, rand_idx, batch_dims=1)

    mask_window_indices = tf.stack([
        mask_center - 1, mask_center, mask_center + 1
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center, win_probs

def generate_random_window_xsmall_eval(inp_mask, idx_positions, idx_win_probs):
    inp_mask = tf_utils.constrain_move_mask_window_positions_batch(inp_mask)

    # Generate random indices within the counts of True values for each batch element.
    batch_size = tf.shape(inp_mask)[0]
    true_counts = tf.fill([batch_size], 3)
    rand_idx = tf_utils.get_random_mask_position_bias(true_counts)

    # Select mask center from the idx_options.
    mask_center = tf.gather(idx_positions, rand_idx, batch_dims=1)
    win_probs = tf.gather(idx_win_probs, rand_idx, batch_dims=1)

    mask_window_indices = tf.stack([
        mask_center
    ], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center, win_probs



