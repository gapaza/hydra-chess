import tensorflow as tf
import config
from preprocess.strategies.tf_utils import get_move_masking_positions, \
        constrain_move_mask_window_positions, generate_random_mask_window, \
        pad_existing_sequence_moves, apply_move_mask, \
        generate_random_mask_window_long, get_move_masking_positions_batch, constrain_move_mask_window_positions_batch
from preprocess.strategies import py_utils



def dual_objective_batch(encoded_texts):
        output_batch = tf.map_fn(
                dual_objective,  # The function to apply to each element in the batch
                encoded_texts,  # The input tensor with shape (None, 128)
                fn_output_signature = (
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # move_seq_masked
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # move_seq_labels
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # move_seq_sample_weights
                    tf.TensorSpec(shape=(8, 8, 14), dtype=tf.int64),  # board_tensor_masked
                    tf.TensorSpec(shape=(64,), dtype=tf.int64),  # board_tensor_labels
                    tf.TensorSpec(shape=(64,), dtype=tf.int64),  # board_tensor_sample_weights
                )
                # The expected output shape and data type
        )
        return output_batch


def dual_objective(encoded_moves):

        # ---------------------
        # --- MOVE MODALITY ---
        # ---------------------
        # 1. Move sequence masked (128,)
        # 2. Move sequence labels (128,)
        # 3. Move sequence sample weights (128,)

        # 1. Get move y labels for mask prediction
        move_seq_labels = tf.identity(encoded_moves)

        # 2. Find possible masking positions, constrain positions, and generate random mask
        inp_mask = get_move_masking_positions(encoded_moves)
        inp_mask = constrain_move_mask_window_positions(inp_mask)
        inp_mask, mask_start, mask_center, mask_end = generate_random_mask_window(inp_mask)
        move_seq_masked = apply_move_mask(encoded_moves, inp_mask)

        # 3. Create sample weights for loss function
        labels = -1 * tf.ones(encoded_moves.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_moves, labels)
        move_seq_sample_weights = tf.ones(labels.shape, dtype=tf.int64)
        move_seq_sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), move_seq_sample_weights)


        # ----------------------
        # --- BOARD MODALITY ---
        # ----------------------
        # 1. Board tensor masked (8, 8, 14)
        # 2. Board tensor labels (64)
        # 3. Board tensor sample weights (64)

        mask_tensor = tf.random.uniform((8, 8), minval=0, maxval=1) < 0.05
        masked_board, board_square_labels, board_square_weights = tf.py_function(py_utils.get_board_tensor_classes_at_move, [encoded_moves, mask_center, mask_tensor], [tf.int64, tf.int64, tf.int64])
        masked_board.set_shape((8, 8, 14))
        board_square_labels.set_shape((64,))
        board_square_weights.set_shape((64,))

        return move_seq_masked, move_seq_labels, move_seq_sample_weights, masked_board, board_square_labels, board_square_weights
















