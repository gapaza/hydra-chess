import tensorflow as tf
import tensorflow_ranking as tfr
import config
from preprocess.strategies.py_utils import get_sequence_board_tensor, get_board_tensor_at_move
import os



@tf.function
def encode_batch(norm_scores, prev_moves):
    prev_moves_encoded = config.tokenizer(prev_moves)
    return norm_scores, prev_moves_encoded


@tf.function
def move_ranking_batch(norm_scores, prev_moves):
    # (None, 1973) (None, 1973) (None, 128)
    # norm_scores, uci_moves, prev_moves = all_inputs
    output_batch = tf.map_fn(
            move_ranking,  # The function to apply to each element in the batch
            (norm_scores, prev_moves),  # The input tensor with shape (None, 128)
            fn_output_signature = (
                tf.TensorSpec(shape=(128,), dtype=tf.int64),                    # current_position
                tf.TensorSpec(shape=(config.vocab_size,), dtype=tf.float32),    # ranked move relevancy scores
                tf.TensorSpec(shape=(8, 8, 12), dtype=tf.int64),                # board_tensor
            )
            # The expected output shape and data type
    )
    return output_batch


@tf.function
def move_ranking(all_inputs):
    candidate_scores, previous_moves_encoded = all_inputs # (1973,) (128,)
    board_tensor = tf.py_function(get_sequence_board_tensor, [previous_moves_encoded], tf.int64)
    board_tensor.set_shape((8, 8, 12))
    return previous_moves_encoded, candidate_scores, board_tensor










#
# @tf.function
# def test_move_ranking(first_element):
#     # your test code here
#     prev_moves_encoded, uci_moves_encoded, norm_scores, board_tensor = move_ranking(first_element)

# @tf.function
# def loss_fn(y_true, y_pred):
#     print('y_true:', y_true)
#     print('y_pred:', y_pred)
#
#     return None
#     # results = tfr.losses._approx_ndcg_loss(y_true, y_pred)
#     # return results




if __name__ == '__main__':
    print('Testing Move Ranking')

    # 1. Test Position
    input_obj = {
        'norm_scores': [1.0, 0.85, 0.84],
        'uci_moves': ['b8c6', 'a7a6', 'g7g6'],
        'prev_moves': ' '.join(['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3'])
    }
    positions = [input_obj] * 128


    # 2. Create Dataset
    import preprocess.FineTuningDatasetGenerator as pp
    dataset = pp.FineTuningDatasetGenerator.create_and_pad_dataset(positions)
    dataset = dataset.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset.save(os.path.join(config.datasets_dir, 'test_dataset'))

    # 3. Parse Dataset
    # first_element = next(iter(dataset.take(1)))

    # 4. Test Move Ranking
    # prev_moves_encoded, uci_moves_encoded, norm_scores, board_tensor = first_element
    #
    #

    ###############
    ### Example ###
    ###############
    # - There are 5 possible moves, where the first three were evaluated by the model
    y_true = [[0., 0.84, 1., 0.85, 0.]]
    y_true = tf.convert_to_tensor(y_true)

    # Prediction
    y_pred = [[0.5, 0.5, 0.5, 0.5, 0.5]]
    y_pred = tf.convert_to_tensor(y_pred)

    # Loss function
    loss = tfr.keras.losses.ApproxNDCGLoss()
    val = loss(y_true, y_pred).numpy()
    print(val)

    # -0.68176705







