import os
import pickle
from datetime import datetime
import tensorflow as tf
import platform


# Tensorflow Core
if platform.system() != 'Darwin':
    print('--> ENABLING MIXED PRECISION')
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)



#######################
##### Directories #####
#######################
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'hydra-chess')
datasets_dir = os.path.join(root_dir, 'datasets')
pt_datasets_dir = os.path.join(datasets_dir, 'pt')
ft_datasets_dir = os.path.join(datasets_dir, 'ft')
weights_dir = os.path.join(root_dir, 'weights')
models_dir = os.path.join(root_dir, 'models')
tokens_dir = os.path.join(root_dir, 'tokens')
plots_dir = os.path.join(root_dir, 'plots')

##########################
##### Model Settings #####
##########################
mode = 'pt'
model_name = 'hydra-med'
seq_length = 128  # 256 max
embed_dim = 256  # 256 nominal
encoder_dense_dim = 2048  # 2048 nominal
encoder_heads = 48
num_sparse_board = 3
vt_dense_dim = 2048  # 2048 nominal
vt_img_size = 8
vt_patch_size = 1
vt_num_patches = (vt_img_size // vt_patch_size) ** 2
vt_epsilon = 1e-6
vt_heads = 48

#########################
### Transfer Learning ###
#########################
tl_enabled = False
tl_load_checkpoint = os.path.join(models_dir, 'hydra-small-pt')
tl_interface_checkpoint = os.path.join(models_dir, 'hydra-small-ft')



####################
### Pre-Training ###
####################
pt_model_weights = os.path.join(weights_dir, 'hydra-pt')
pt_epochs = 20
pt_batch_size = 64

# Datasets
pt_millionsbase_dataset = os.path.join(pt_datasets_dir, 'millionsbase')
pt_millionsbase_small_dataset = os.path.join(pt_datasets_dir, 'millionsbase-small')
pt_chesscom_dataset = os.path.join(pt_datasets_dir, 'chesscom')
pt_millionsbase_chesscom_dataset = os.path.join(pt_datasets_dir, 'milbase-chesscom')
pt_millionsbase_pt2_dataset = os.path.join(pt_datasets_dir, 'millionsbase-pt2')
pt_chesscom_pt2_dataset = os.path.join(pt_datasets_dir, 'chesscom-pt2')

# 100k positions
pt_millionsbase_pt3_dataset_small = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small')
pt_millionsbase_pt3_dataset_small_64 = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64')
pt_millionsbase_pt3_dataset_small_64_20p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64-20p')
pt_millionsbase_pt3_dataset_small_64_25p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64-25p')
pt_millionsbase_pt3_dataset_small_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64-30p')
pt_millionsbase_pt3_dataset_small_64_40p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64-40p')
pt_millionsbase_pt3_dataset_small_64_50p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-64-50p')
pt_millionsbase_pt3_dataset_small_128 = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small-128')

# 200k positions
pt_millionsbase_pt3_dataset_small2_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-small2-64-30p')

# 1mil positions
pt_millionsbase_pt3_dataset_med_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-med-64-30p')

# 3.4mil positions
pt_millionsbase_pt3_dataset_large_64_30p = os.path.join(pt_datasets_dir, 'millionsbase-pt3-large-64-30p')
pt_millionsbase_pt3_dataset_large_64_30p_int32 = os.path.join(pt_datasets_dir, 'millionsbase-pt3-large-64-30p-int32')
pt_millionsbase_pt3_dataset_large_64_30p_int16 = os.path.join(pt_datasets_dir, 'millionsbase-pt3-large-64-30p-int16')




###################
### Fine-Tuning ###
###################
ft_model_weights = os.path.join(weights_dir, 'hydra-ft')
ft_epochs = 1
ft_batch_size = 64
ft_top_n = 3

# Datasets
ft_lc0_standard_dir = os.path.join(ft_datasets_dir, 'lc0_standard')
ft_lc0_standard_2mil_dir = os.path.join(ft_datasets_dir, 'lc0_standard_2mil')
ft_lc0_standard_2mil_mask_dir = os.path.join(ft_datasets_dir, 'lc0_standard_2mil_mask')
ft_lc0_standard_200k_legal_dir = os.path.join(ft_datasets_dir, 'lc0_standard_200k_legal')
ft_lc0_standard_small_ft2_64 = os.path.join(ft_datasets_dir, 'lc0_standard_small_ft2_64')
ft_lc0_standard_med_ft2_64 = os.path.join(ft_datasets_dir, 'lc0_standard_med_ft2_64')
ft_lc0_standard_large_ft2_64 = os.path.join(ft_datasets_dir, 'lc0_standard_large_ft2_64')







##############################
### Tokenizer + Vocabulary ###
##############################
import tensorflow as tf
from keras.layers import TextVectorization
import re


def custom_standardization(input_data):
    # lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(input_data, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape("!#$%&'()*+,-./:;<=>?@\^_`{|}~"), ""
    )


special_tokens = ["[pos]", "[mask]"]
num_special_tokens = len(special_tokens) + 2
vocab_file = os.path.join(root_dir, 'tokens', 'tokens_1969_merged.pkl')  # tokens_1966.pkl, tokens_1968_chesscom
vocab = []
with open(vocab_file, 'rb') as f:
    vocab = list(pickle.load(f))
    # remove empty string and [UNK]
    if '' in vocab:
        vocab.remove('')
    vocab.sort()
vocab = special_tokens + vocab
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

# Commands
# scp -i ~/keys/gabe-master.pem ./human-training-games-299k.zip ubuntu@3.17.77.24:/home/ubuntu/MultiModalChess/datasets
# scp -i ~/keys/gabe-master.pem ubuntu@18.221.115.53:/home/ubuntu/MultiModalChess/positions/human-training-games-141727.zip .
# scp -i ~/keys/gabe-master.pem ./human-training-games-141727.zip ubuntu@18.221.115.53:/home/ubuntu/MultiModalChess/positions
# scp -i ~/keys/gabe-master.pem /Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/millionsbase.zip ubuntu@3.145.44.57:/home/ubuntu/hydra-chess/datasets/pt/millionsbase
