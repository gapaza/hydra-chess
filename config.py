import os
import pickle
from datetime import datetime
import tensorflow as tf
import platform

# tf.config.set_visible_devices([], 'GPU')


# Tensorflow Core
mixed_precision = True
if platform.system() != 'Darwin' and mixed_precision is True:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# Distributed Training
distributed = False
mirrored_strategy = tf.distribute.MirroredStrategy()
global_batch_size = 256  # 64, 128, 256, 512, 1024


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
tokens_dir = os.path.join(root_dir, 'tokens')
plots_dir = os.path.join(root_dir, 'plots')

datasets_dir = os.path.join(root_dir, 'datasets')
pt_datasets_dir = os.path.join(datasets_dir, 'pt')
ft_datasets_dir = os.path.join(datasets_dir, 'ft')

models_dir = os.path.join(root_dir, 'models')
weights_dir = os.path.join(root_dir, 'weights')



#
#       __  __             _        _
#      |  \/  |           | |      | |
#      | \  / |  ___    __| |  ___ | |
#      | |\/| | / _ \  / _` | / _ \| |
#      | |  | || (_) || (_| ||  __/| |
#      |_|  |_| \___/  \__,_| \___||_|
#

model_base = 'encoder'  # 'encoder', 'custom'
model_name = 'hydra'
model_mode = 'move-ranking'  # 'game-modeling', 'position-modeling', 'move-prediction', 'move-ranking'
train_mode = 'pt'               # 'pt', 'ft'

#############################
# --> Transfer Learning <-- #
#############################
# 1. Checkpoint Saving: model / weights
# 2. Model Saving
# - V2: tf version
# - V3: keras version

# Saving Paths
tl_model_class = 'hydra-family'
tl_hydra_base_save = os.path.join(models_dir, tl_model_class, 'hydra-base-decoder-only')
tl_hydra_full_save = os.path.join(models_dir, tl_model_class, 'hydra-decoder-only')
tl_hydra_base_weights_save = os.path.join(weights_dir, tl_model_class, 'hydra-base-v3-ftr-u16.h5')
tl_hydra_full_weights_save = os.path.join(weights_dir, tl_model_class, 'hydra-full-v3-ftr-u16.h5')

tl_decoder_only = os.path.join(models_dir, tl_model_class, 'decoder-only')


# Loading Paths
tl_freeze_base = False
tl_freeze_base_partial = True
tl_hydra_base_load = os.path.join(models_dir, 'hydra-family/hydra-base')
tl_hydra_full_load = os.path.join(models_dir, 'hydra-family/hydra-full-enc-v3-ftr-u16')
tl_hydra_base_weights_load = os.path.join(weights_dir, 'hydra-family/hydra-base.h5')
tl_hydra_full_weights_load = os.path.join(weights_dir, 'hydra-family/hydra-full.h5')


# Production Paths
tl_base_path = None        # tl_hydra_base_load
tl_full_model_path = tl_hydra_full_load  # tl_hydra_full_load
tl_head_path = None        # None for now


###########################
# --> Hyperparameters <-- #
###########################

dense_dim = 2048
heads = 8
attack_strategy = True
board_modality_classes = 28  # 16 nominal, 28 attack strategy
board_seq_length = 65
seq_length = 128  # 256 max
embed_dim = 256  # 256 nominal
num_experts = 8

# --> Dropout
dropout = 0.1





#
#       _____          _                     _
#      |  __ \        | |                   | |
#      | |  | |  __ _ | |_  __ _  ___   ___ | |_  ___
#      | |  | | / _` || __|/ _` |/ __| / _ \| __|/ __|
#      | |__| || (_| || |_| (_| |\__ \|  __/| |_ \__ \
#      |_____/  \__,_| \__|\__,_||___/ \___| \__||___/
#
endgame_bias = 0.05


#######################
# --> Pretraining <-- #
#######################
pt_learning_rate = 0.002
pt_epochs = 200
pt_steps_per_epoch = 26000
pt_val_steps = 500

# Datasets
pt_mixed_eval_4mil = os.path.join(pt_datasets_dir, 'mixed-eval-4mil')
pt_megaset = os.path.join(pt_datasets_dir, 'megaset')
pt_baseline = os.path.join(pt_datasets_dir, 'decoder-only')

# Loaded Dataset
pt_dataset = pt_mixed_eval_4mil
pt_train_buffer = 2048 * 100
pt_val_buffer = 256


#######################
# --> Fine-Tuning <-- #
#######################
ft_learning_rate = 0.0008
ft_epochs = 1
ft_steps_per_epoch = 6000
ft_val_steps = 500

# Datasets
ft_lichess = os.path.join(ft_datasets_dir, 'lichess_ft')
ft_lichess_mates = os.path.join(ft_datasets_dir, 'lichess_mates')
ft_lichess_tactics = os.path.join(ft_datasets_dir, 'lichess_tactics')
ft_evaluations = os.path.join(ft_datasets_dir, 'evaluations')
ft_lichess_puzzles = os.path.join(ft_datasets_dir, 'lichess_puzzles')

# Loaded Dataset
ft_dataset = ft_evaluations
ft_train_buffer = 2048 * 100
ft_val_buffer = 256


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
special_tokens = ["[pos]", "[mask]", '[start]']
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
start_token_id = tokenizer(["[start]"]).numpy()[0][0]
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
