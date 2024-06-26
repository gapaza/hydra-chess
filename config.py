import os
import pickle
from datetime import datetime
import tensorflow as tf
import platform

# Tensorflow Core
if platform.system() != 'Darwin':
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# Distributed Training
distributed = True
mirrored_strategy = tf.distribute.MirroredStrategy()
global_batch_size = 64  # 128, 256, 512, 1024


#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'hydra-chess')
datasets_dir = os.path.join(root_dir, 'datasets')
pt_datasets_dir = os.path.join(datasets_dir, 'pt')
ft_datasets_dir = os.path.join(datasets_dir, 'ft')
weights_dir = os.path.join(root_dir, 'weights')
tokens_dir = os.path.join(root_dir, 'tokens')
plots_dir = os.path.join(root_dir, 'plots')
models_dir = os.path.join(root_dir, 'models')
encoder_dir = os.path.join(models_dir, 'encoder')
decoder_dir = os.path.join(models_dir, 'decoder')
vision_dir = os.path.join(models_dir, 'vision')



#
#       __  __             _        _
#      |  \/  |           | |      | |
#      | \  / |  ___    __| |  ___ | |
#      | |\/| | / _ \  / _` | / _ \| |
#      | |  | || (_) || (_| ||  __/| |
#      |_|  |_| \___/  \__,_| \___||_|
#

attack_strategy = True
board_modality_classes = 28  # 16 nominal, 28 attack strategy
board_seq_length = 65
seq_length = 128  # 256 max
model_name = 'hydra'
model_type = 'hybrid'  # 'encoder', 'decoder', 'vision', 'hybrid'
model_mode = 'pt-eval'  # 'pt', 'pt-eval' 'ft-classify', 'ft-ndcg'
model_save_name = model_name + '-' + model_mode
model_save_dir = os.path.join(models_dir, model_type, model_save_name)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)



# --> Transfer Learning <-- #
tl_enabled = False
tl_load_checkpoint = os.path.join(models_dir, model_type, 'hydra-pt-eval-240k')
tl_interface_checkpoint = os.path.join(models_dir, model_type, 'hydra-ft-classify-tactics')


###########################
### Transformer Encoder ###
###########################
embed_dim = 256  # 256 nominal
encoder_dense_dim = 2048  # 2048 nominal
encoder_heads = 48  # 48 nominal


###########################
### Transformer Decoder ###
###########################
de_dense_dim = 2048  # 2048 nominal
de_heads = 48
dc_mode = 'single'  # 'dual' or 'single'

##########################
### Transformer Hybrid ###
##########################
hy_dense_dim = 2048
hy_heads = 48
hy_mode = 'single'  # 'dual' or 'single'


##########################
### Vision Transformer ###
##########################
vt_dense_dim = 2048  # 2048 nominal
vt_img_size = 8
vt_patch_size = 1
vt_num_patches = (vt_img_size // vt_patch_size) ** 2
vt_epsilon = 1e-6
vt_heads = 48






#
#       _____          _                     _
#      |  __ \        | |                   | |
#      | |  | |  __ _ | |_  __ _  ___   ___ | |_  ___
#      | |  | | / _` || __|/ _` |/ __| / _ \| __|/ __|
#      | |__| || (_| || |_| (_| |\__ \|  __/| |_ \__ \
#      |_____/  \__,_| \__|\__,_||___/ \___| \__||___/
#
endgame_bias = 0.08


####################
### Pre-Training ###
####################
pt_epochs = 10
pt_batch_size = 256  # 256, 512
pt_batch_val = 120000


# --> Window-3 Datasets <-- #
pt_millionsbase_dataset = os.path.join(pt_datasets_dir, 'millionsbase')
pt_chesscom_dataset = os.path.join(pt_datasets_dir, 'chesscom')
pt_megaset_dataset = os.path.join(pt_datasets_dir, 'megaset')

# 1mil positions
pt_millionsbase_pt3_dataset_med_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-med-64-30p')

# 3.4mil positions
pt_millionsbase_pt3_dataset_large_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-large-64-30p-int16')
pt_millionsbase_pt3_dataset_large_256_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-large-256-30p-int16')

# ~7.7 mil games
pt_megaset_pt3_dataset_64_30p_int16 = os.path.join(pt_datasets_dir, 'megaset-pt3-64-30p-int16')



# --> Window-5 Datasets <-- #
pt_millionsbase_5w_256 = os.path.join(pt_datasets_dir, 'millionsbase-5w-256')

# --> Window-7 Datasets <-- #
pt_millionsbase_7w_256 = os.path.join(pt_datasets_dir, 'millionsbase-7w-256-7peb')

# --> Window-9 Datasets <-- #
pt_millionsbase_9w_256 = os.path.join(pt_datasets_dir, 'millionsbase-7w-256-7peb')

# --> Window-11 Datasets <-- #
pt_millionsbase_11w_256 = os.path.join(pt_datasets_dir, 'millionsbase-11w-256-7peb')

# --> Window-7 Datasets <-- #
pt_millionsbase_7w_256_7peb = os.path.join(pt_datasets_dir, 'millionsbase-7w-256-7peb')


# --> Denoising Datasets <-- #
pt_millionsbase_500k_64 = os.path.join(pt_datasets_dir, 'millionsbase-500k-denoising-64')
pt_millionsbase_500k_256 = os.path.join(pt_datasets_dir, 'millionsbase-500k-denoising-256')

pt_megaset_denoising_64 = os.path.join(pt_datasets_dir, 'megaset-denoising-64')
pt_megaset_denoising_256 = os.path.join(pt_datasets_dir, 'megaset-denoising-256')



# --> Unsupervised Datasets <-- #
pt_megaset_bw = os.path.join(pt_datasets_dir, 'megaset-bw')


# --> Unsupervised Eval Datasets <-- #
pt_mixed_eval_1mil = os.path.join(pt_datasets_dir, 'mixed-eval-1mil')
pt_mixed_eval_4mil = os.path.join(pt_datasets_dir, 'mixed-eval-4mil')




###################
### Fine-Tuning ###
###################
ft_epochs = 3
ft_batch_size = 256
ft_top_n = 3
ft_batch_val = 500

# Datasets
ft_lc0_standard_dir = os.path.join(ft_datasets_dir, 'lc0_standard')
ft_lc0_standard_large_128_dir = os.path.join(ft_datasets_dir, 'lc0_standard_large_128')
ft_lc0_standard_large_128_mask_dir = os.path.join(ft_datasets_dir, 'lc0_standard_large_128_mask')
ft_lc0_standard_large_256_mask_dir = os.path.join(ft_datasets_dir, 'lc0_standard_large_256_mask')


# Tactic Dataset
ft_lichess = os.path.join(ft_datasets_dir, 'lichess_ft')

ft_lichess_mates = os.path.join(ft_datasets_dir, 'lichess_mates')
ft_lichess_tactics = os.path.join(ft_datasets_dir, 'lichess_tactics')






#
#      __      __                 _             _
#      \ \    / /                | |           | |
#       \ \  / /___    ___  __ _ | |__   _   _ | |  __ _  _ __  _   _
#        \ \/ // _ \  / __|/ _` || '_ \ | | | || | / _` || '__|| | | |
#         \  /| (_) || (__| (_| || |_) || |_| || || (_| || |   | |_| |
#          \/  \___/  \___|\__,_||_.__/  \__,_||_| \__,_||_|    \__, |
#                                                                __/ |
#                                                               |___/
#

import tensorflow as tf
from keras.layers import TextVectorization
import re

def custom_standardization(input_data):
    # lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


move_language = 'uci'  # 'uci' | 'san'
if move_language == 'uci':
    vocab_file = os.path.join(tokens_dir, 'tokens_1969_merged.pkl')  # tokens_1966.pkl, tokens_1968_chesscom
elif move_language == 'san':
    vocab_file = os.path.join(tokens_dir, 'tokens_san_9940.pkl')


end_of_game_tokens = ["[white]", "[black]", "[draw]"]
special_tokens = ["[pos]", "[mask]"]
num_special_tokens = len(special_tokens) + 2
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    # remove empty string and [UNK]
    if '' in vocab:
        vocab.remove('')
    vocab.sort()
vocab = special_tokens + vocab + end_of_game_tokens
vocab_size = len(vocab)
tokenizer = TextVectorization(
    max_tokens=vocab_size + 2,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=seq_length,
)
tokenizer_long = TextVectorization(
    max_tokens=vocab_size + 2,
    output_mode="int",
    standardize=custom_standardization,
    output_sequence_length=vocab_size + 2,
)
tokenizer.set_vocabulary(vocab)
tokenizer_long.set_vocabulary(vocab)
vocab = tokenizer.get_vocabulary()
vocab_size = len(vocab)
mask_token_id = tokenizer(["[mask]"]).numpy()[0][0]
padding_token_id = tokenizer(['']).numpy()[0][0]
pos_token_id = tokenizer(["[pos]"]).numpy()[0][0]
white_token_id = tokenizer(["[white]"]).numpy()[0][0]
black_token_id = tokenizer(["[black]"]).numpy()[0][0]
draw_token_id = tokenizer(["[draw]"]).numpy()[0][0]
id2token = dict(enumerate(tokenizer.get_vocabulary()))
token2id = {y: x for x, y in id2token.items()}


def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()


@tf.function
def encode_tf(input):
    encoded_input = tokenizer(input)
    # encoded_input = tf.reshape(encoded_input, (-1,))
    # if tf.rank(encoded_input) > 1:
    #     encoded_input = tf.squeeze(encoded_input, axis=0)
    return encoded_input


@tf.function
def encode_tf_long(input):
    encoded_input = tokenizer_long(input)
    if tf.rank(encoded_input) > 1:
        encoded_input = tf.squeeze(encoded_input, axis=0)
    return encoded_input


@tf.function
def encode_tf_batch(input):
    encoded_input = tokenizer(input)
    return encoded_input


def encode_tf_old(input):
    encoded_input = tokenizer(tf.expand_dims(input, axis=0))
    encoded_input = tf.squeeze(encoded_input, axis=0)
    return encoded_input
