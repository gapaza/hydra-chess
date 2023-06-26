import tensorflow as tf
import config
from preprocess.strategies.tf_utils import get_move_masking_positions, \
        constrain_move_mask_window_positions, generate_random_mask_window, \
        pad_existing_sequence_moves, apply_move_mask, \
        generate_random_mask_window_long, get_move_masking_positions_batch, constrain_move_mask_window_positions_batch
from preprocess.strategies.py_utils import get_sequence_board_tensor, get_board_tensor_at_move


def rand_window_batch(encoded_texts):
        output_batch = tf.map_fn(
                rand_window,  # The function to apply to each element in the batch
                encoded_texts,  # The input tensor with shape (None, 128)
                fn_output_signature = (
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # encoded_texts_masked
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # y_labels
                    tf.TensorSpec(shape=(128,), dtype=tf.int64),      # sample_weights
                    tf.TensorSpec(shape=(8, 8, 12), dtype=tf.int64),  # board_tensor
                )
                # The expected output shape and data type
        )
        return output_batch


def rand_window(encoded_texts):

        # 1.1 Get y labels for mask prediction
        y_labels = tf.identity(encoded_texts)

        # 2. Find possible masking positions
        inp_mask = get_move_masking_positions(encoded_texts)

        # 3. Constrain masking positions by disabling first and last move token
        inp_mask = constrain_move_mask_window_positions(inp_mask)

        # 4. Generate random mask
        # inp_mask_1 = generate_random_mask_window(tf.identity(inp_mask))
        # inp_mask_2 = generate_random_mask_window(tf.identity(inp_mask))
        # inp_mask = tf.math.logical_or(inp_mask_1, inp_mask_2)
        inp_mask, mask_start, mask_center, mask_end = generate_random_mask_window(inp_mask)

        # 5 Get board tensor using tf.py_function to call get_board_tensor_from_moves
        # board_tensor = tf.py_function(get_sequence_board_tensor, [encoded_texts], tf.int64)
        board_tensor = tf.py_function(get_board_tensor_at_move, [encoded_texts, mask_center], tf.int64)
        board_tensor.set_shape((8, 8, 12))

        # 6. Create labels for masked tokens
        labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_texts, labels)

        # 7. Create masked input
        encoded_texts_masked = apply_move_mask(encoded_texts, inp_mask)

        # 8. Define loss function weights
        sample_weights = tf.ones(labels.shape, dtype=tf.int64)
        sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), sample_weights)

        return encoded_texts_masked, y_labels, sample_weights, board_tensor


def rand_window_rand_game_token(encoded_texts):

        # 1.1 Get y labels for mask prediction
        y_labels = tf.identity(encoded_texts)

        # 2. Find possible masking positions
        inp_mask = get_move_masking_positions(encoded_texts)

        # 3. Constrain masking positions by disabling first and last move token
        inp_mask, first_true_index, last_true_index = constrain_move_mask_window_positions(inp_mask)
        inp_mask, first_true_index, last_true_index = constrain_move_mask_window_positions(inp_mask)
        minval = tf.squeeze(first_true_index, axis=0)
        maxval = tf.squeeze(last_true_index, axis=0)
        minval = tf.cast(minval, tf.int64)
        maxval = tf.cast(maxval, tf.int64)
        seed = tf.constant([42, 42], dtype=tf.int32)
        random_board_idx = tf.random.stateless_uniform(shape=(), minval=minval, maxval=maxval, seed=seed, dtype=tf.int64)
        # random_board_idx = tf.random.uniform(shape=(), minval=minval, maxval=maxval, dtype=tf.int64)


        # 4. Generate random mask
        # inp_mask_1 = generate_random_mask_window(tf.identity(inp_mask))
        # inp_mask_2 = generate_random_mask_window(tf.identity(inp_mask))
        # inp_mask = tf.math.logical_or(inp_mask_1, inp_mask_2)
        inp_mask, mask_start, mask_center, mask_end = generate_random_mask_window(tf.identity(inp_mask))

        # 5 Get board tensor using tf.py_function to call get_board_tensor_from_moves
        # board_tensor = tf.py_function(get_sequence_board_tensor, [encoded_texts], tf.int64)
        board_tensor = tf.py_function(get_board_tensor_at_move, [encoded_texts, random_board_idx], tf.int64)

        # 6. Create labels for masked tokens
        labels = -1 * tf.ones(encoded_texts.shape, dtype=tf.int64)
        labels = tf.where(inp_mask, encoded_texts, labels)

        # 7. Create masked input
        encoded_texts_masked = apply_move_mask(encoded_texts, inp_mask)

        # 8. Define loss function weights
        sample_weights = tf.ones(labels.shape, dtype=tf.int64)
        sample_weights = tf.where(tf.equal(labels, -1), tf.zeros_like(labels), sample_weights)

        # 10. Add position tokens
        pos_token = random_board_idx

        # print('\n\n\n\n')
        # print('Encoded Texts Masked Before: ', encoded_texts_masked)
        encoded_texts_masked = tf.concat([encoded_texts_masked[:pos_token],
                                          tf.constant([config.pos_token_id], dtype=tf.int64),
                                          encoded_texts_masked[pos_token:]], axis=0)
        encoded_texts_masked = tf.slice(encoded_texts_masked, [0], [128])
        # print('Encoded Texts Masked After: ', encoded_texts_masked)
        # print('\n\n\n\n')


        # Insert the [POS] token and remove the last element from y_labels
        # print('\n\n\n\n')
        # print('Y Labels Before: ', y_labels)
        y_labels = tf.concat([y_labels[:pos_token],
                              tf.constant([config.pos_token_id], dtype=tf.int64),
                              y_labels[pos_token:]], axis=0)
        y_labels = tf.slice(y_labels, [0], [128])
        # print('Y Labels After: ', y_labels)
        # print('\n\n\n\n')

        # Insert the 0 and remove the last element from sample_weights
        # print('\n\n\n\n')
        # print('Sample Weights Before: ', sample_weights)
        sample_weights = tf.concat([sample_weights[:pos_token],
                                    tf.constant([0], dtype=tf.int64),
                                    sample_weights[pos_token:]], axis=0)
        sample_weights = tf.slice(sample_weights, [0], [128])
        # print('Sample Weights After: ', sample_weights)
        # print('\n\n\n\n')


        return encoded_texts_masked, y_labels, sample_weights, board_tensor
