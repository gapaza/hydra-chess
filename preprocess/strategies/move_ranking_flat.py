import tensorflow as tf
import tensorflow_ranking as tfr
import config
from preprocess.strategies import py_utils
import os







@tf.function
def encode_batch(norm_scores, prev_moves, norm_scores_idx, legal_moves_idx, legal_move_scores):
    prev_moves_encoded = config.tokenizer(prev_moves)
    return norm_scores, prev_moves_encoded, norm_scores_idx, legal_moves_idx, legal_move_scores


@tf.function
def move_ranking_batch_flat(norm_scores, prev_moves, norm_scores_idx, legal_moves_idx, legal_move_scores):
    # (None, 1973) (None, 1973) (None, 128)
    # norm_scores, uci_moves, prev_moves = all_inputs
    output_batch = tf.map_fn(
            move_ranking_flat,  # The function to apply to each element in the batch
            (norm_scores, prev_moves, norm_scores_idx, legal_moves_idx, legal_move_scores),  # The input tensor with shape (None, 128)
            fn_output_signature = (
                tf.TensorSpec(shape=(128,), dtype=tf.int64),                    # current_position
                tf.TensorSpec(shape=(config.vocab_size,), dtype=tf.float32),    # ranked move relevancy scores
                tf.TensorSpec(shape=(8, 8), dtype=tf.int64),                # board_tensor
            )
            # The expected output shape and data type
    )
    return output_batch

@tf.function
def move_ranking_flat(all_inputs):
    candidate_move_scores, previous_moves_encoded, candidate_moves_idx, legal_moves_idx, legal_move_scores = all_inputs  # (3,) (128,) (3,)

    # Create board tensor
    board_tensor = tf.py_function(py_utils.get_sequence_board_tensor_classes_flat, [previous_moves_encoded], tf.int64)
    board_tensor.set_shape((8, 8))

    # Create candidate move labels
    all_move_labels = tf.zeros((config.vocab_size,), dtype=tf.float32)

    # Add legal moves
    legal_moves_idx = tf.reshape(legal_moves_idx, [-1, 1])
    all_move_labels = tf.tensor_scatter_nd_update(all_move_labels, legal_moves_idx, legal_move_scores)

    # Add candidate moves
    candidate_moves_idx = tf.reshape(candidate_moves_idx, [-1, 1])
    all_move_labels = tf.tensor_scatter_nd_update(all_move_labels, candidate_moves_idx, candidate_move_scores)

    return previous_moves_encoded, all_move_labels, board_tensor


























