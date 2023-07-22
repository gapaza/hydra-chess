import tensorflow as tf
import config



def get_move_masking_positions(tokenized_text):

    # Set all possible positions to true
    seed = tf.constant([42, 42], dtype=tf.int32)
    inp_mask = tf.random.stateless_uniform(tokenized_text.shape, seed=seed) <= 1.0

    # tokens with id < 3 are special tokens and can't be masked
    # thus, set all tokens with id < 3 to false
    inp_mask = tf.logical_and(inp_mask, tokenized_text > (config.num_special_tokens - 1))

    return inp_mask

def get_move_masking_positions_batch(tokenized_text):
    seed = tf.constant([42, 42], dtype=tf.int32)
    inp_mask = tf.random.stateless_uniform(tf.shape(tokenized_text), seed=seed) <= 1.0
    inp_mask = tf.logical_and(inp_mask, tokenized_text > (config.num_special_tokens - 1))
    return inp_mask








@tf.function
def constrain_move_mask_window_positions(inp_mask):
    true_indices = tf.where(inp_mask)

    # Find first true index
    first_true_index = true_indices[0]

    # Set first true index to False
    inp_mask = tf.concat([
        inp_mask[:first_true_index[0]],
        [False],
        inp_mask[first_true_index[0] + 1:]
    ], axis=0)

    # Set last true index to False
    last_true_index = true_indices[-1]
    inp_mask = tf.concat([
        inp_mask[:last_true_index[0]],
        [False],
        inp_mask[last_true_index[0] + 1:]
    ], axis=0)

    # Set shape and return
    inp_mask.set_shape((config.seq_length,))
    return inp_mask


@tf.function
def constrain_move_mask_window_positions_batch(inp_mask):

    # Find first true index
    mask = tf.math.cumsum(tf.cast(inp_mask, tf.float32), axis=1)
    first_true = tf.equal(mask, 1)
    indices_first = tf.cast(tf.where(first_true), dtype=tf.int32)

    # Find last true index
    reversed_batch = tf.reverse(inp_mask, axis=[1])
    mask_reversed = tf.math.cumsum(tf.cast(reversed_batch, tf.float32), axis=1)
    last_true_reversed = tf.equal(mask_reversed, 1)
    indices_last_reversed = tf.cast(tf.where(last_true_reversed), dtype=tf.int32)
    last_indices_per_batch_reversed = tf.math.segment_min(indices_last_reversed[:, 1], indices_last_reversed[:, 0])
    last_indices_per_batch_reversed = tf.cast(last_indices_per_batch_reversed, dtype=tf.int32)
    last_indices_per_batch = tf.subtract(tf.shape(inp_mask)[1] - 1, last_indices_per_batch_reversed)

    # Create a mask of shape (None, 128) that is True at the first True position and False everywhere else.
    first_true_mask = tf.scatter_nd(indices_first, tf.ones(tf.shape(indices_first)[0], dtype=tf.bool), tf.shape(inp_mask))
    last_true_mask = tf.scatter_nd(tf.stack([indices_last_reversed[:, 0], last_indices_per_batch], axis=1), tf.ones(tf.shape(indices_last_reversed)[0], dtype=tf.bool), tf.shape(inp_mask))

    # Invert the masks and use them to update the original boolean tensor.
    updated_batch = tf.where(first_true_mask | last_true_mask, tf.zeros_like(inp_mask, dtype=tf.bool), inp_mask)
    return updated_batch







def generate_random_mask_window(inp_mask):
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
    return inp_mask, mask_start, mask_center, mask_start + mask_length

def generate_random_mask_window_batch(inp_mask):

    # Get the indices of True values in each batch element.
    indices = tf.broadcast_to(tf.range(tf.shape(inp_mask)[-1]), tf.shape(inp_mask))
    true_indices = tf.ragged.boolean_mask(indices, inp_mask)

    # Get the counts of True values in each batch element.
    true_counts = tf.math.count_nonzero(inp_mask, axis=1, dtype=tf.int32)

    # Generate random indices within the counts of True values for each batch element.
    seed = tf.constant([42, 75], dtype=tf.int32)
    rand_idx = tf.map_fn(lambda x: tf.random.stateless_uniform(shape=(), maxval=x, dtype=tf.int32, seed=seed), true_counts)

    # Create a tensor of shape (batch, 3) that defines the mask window.
    mask_center = tf.gather(true_indices, rand_idx, batch_dims=1)
    mask_start = mask_center - 1
    mask_end = mask_center + 1  # end index is exclusive, so add 1 more
    mask_window_indices = tf.stack([mask_start, mask_center, mask_end], axis=-1)

    # Create a boolean mask tensor of shape (batch, 128) that is True only at the mask window.
    mask_range = tf.range(tf.shape(inp_mask)[-1])
    mask_window = tf.reduce_any(tf.equal(tf.expand_dims(mask_range, axis=0), mask_window_indices[..., tf.newaxis]), axis=-2)

    return mask_window, mask_center










def generate_random_mask_window_long(inp_mask):
    true_indices = tf.squeeze(tf.where(inp_mask), axis=1)
    seed = tf.constant([42, 42], dtype=tf.int32)
    rand_idx = tf.random.stateless_uniform(shape=[], maxval=tf.shape(true_indices)[-1], dtype=tf.int32, seed=seed)
    mask_center = tf.gather(true_indices, rand_idx)
    mask_start = mask_center - 2
    mask_length = 5
    mask_indices = tf.range(mask_start, mask_start + mask_length)
    inp_mask = tf.zeros((config.seq_length,), dtype=tf.bool)
    inp_mask = tf.scatter_nd(tf.expand_dims(mask_indices, 1), tf.ones_like(mask_indices, dtype=tf.bool),
                             inp_mask.shape)
    return inp_mask, mask_start, mask_center, mask_start + mask_length

def pad_existing_sequence_moves(tokenized_text, cutoff):
    encoded_texts = tf.concat([
        tokenized_text[:cutoff],
        config.padding_token_id * tf.ones(tf.shape(tokenized_text[cutoff:]), dtype=tf.int64)
    ], axis=0)
    encoded_texts.set_shape((128,))
    return encoded_texts





def apply_move_mask(tokenized_text, inp_mask):
    mask_token_id = config.mask_token_id
    encoded_texts_masked = tf.where(inp_mask, mask_token_id * tf.ones_like(tokenized_text), tokenized_text)
    return encoded_texts_masked









def get_random_mask_position_bias(true_counts):
    
    def random_index(x):
        return tf.random.uniform(shape=(), maxval=x, dtype=tf.int32)

    def endgame_index(x):
        return x - 1

    rand_idx = tf.map_fn(
        lambda x: tf.cond(tf.random.uniform([]) < config.endgame_bias,
                          lambda: endgame_index(x),
                          lambda: random_index(x)),
        true_counts
    )

    return rand_idx





