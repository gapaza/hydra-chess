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
    # Get the batch size
    batch_size = tf.shape(tokenized_text)[0]

    # Set all possible positions to true
    seed = tf.constant([42, 42], dtype=tf.int32)
    inp_mask = tf.random.stateless_uniform(tf.shape(tokenized_text), seed=seed) <= 1.0

    # Tokens with id < 3 are special tokens and can't be masked
    # Thus, set all tokens with id < 3 to false
    inp_mask = tf.logical_and(inp_mask, tokenized_text > (config.num_special_tokens - 1))

    return inp_mask

def constrain_move_mask_window_positions(inp_mask):
    true_indices = tf.where(inp_mask)
    first_true_index = true_indices[0]
    inp_mask = tf.concat([
        inp_mask[:first_true_index[0]],
        [False],
        inp_mask[first_true_index[0] + 1:]
    ], axis=0)
    last_true_index = true_indices[-1]
    inp_mask = tf.concat([
        inp_mask[:last_true_index[0]],
        [False],
        inp_mask[last_true_index[0] + 1:]
    ], axis=0)
    inp_mask.set_shape((config.seq_length,))
    return inp_mask


def constrain_move_mask_window_positions_batch(inp_mask):
    true_indices = tf.where(inp_mask)
    first_true_index = true_indices[:, 0]
    return true_indices, first_true_index




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

