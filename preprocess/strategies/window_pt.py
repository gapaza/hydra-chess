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

    # inp_mask, mask_center = generate_random_window_small(inp_mask)
    # inp_mask, mask_center = generate_random_window_medium(inp_mask)
    inp_mask, mask_center = generate_random_window_large(inp_mask)

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
        mask_center - 3, mask_center - 2, mask_center - 1, mask_center, mask_center + 1, mask_center + 2, mask_center + 3
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
        mask_center - 2, mask_center - 1, mask_center, mask_center + 1, mask_center + 2
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




