import tensorflow as tf
import tensorflow_ranking as tfr
import config
import os
import chess



from preprocess.strategies.move_ranking import move_ranking_batch, encode_batch
from preprocess.strategies.move_ranking_flat import move_ranking_batch_flat, move_ranking_flat
from preprocess.strategies.window_masking import rand_window_multi, rand_window_batch_multi
from preprocess.strategies.py_utils import board_to_tensor_classes
from preprocess.strategies.dual_objective import dual_objective_batch, dual_objective
from preprocess.strategies.dual_objective_flat import dual_objective_flat_batch, dual_objective_flat

from preprocess.FT_DatasetGenerator import FT_DatasetGenerator

def test_move_ranking():
    # 1. Test Position
    input_obj = {
        'candidate_scores': [1.0, 0.85, 0.84],
        'candidate_moves': ['b8c6', 'a7a6', 'g7g6'],
        'candidate_moves_idx': [0, 1, 2],
        'prev_moves': ' '.join(['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3'])
    }
    positions = [input_obj]

    # 2. Create Dataset
    dataset = FT_DatasetGenerator.create_and_pad_dataset(positions)
    dataset = dataset.batch(1, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(move_ranking_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 3. Parse Dataset
    first_element = next(iter(dataset.take(1)))

    # 4. Test Move Ranking
    prev_moves_encoded, norm_scores, board_tensor = first_element
    print('prev_moves_encoded:', prev_moves_encoded)
    print('norm_scores:', norm_scores)
    print('board_tensor:', board_tensor)


def test_move_ranking_flat():
    # 1. Test Position
    candidate_moves = ['b8c6', 'a7a6', 'g7g6']
    candidate_moves_idx = [config.vocab.index(move) for move in candidate_moves]
    prev_moves = ['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3']

    board = chess.Board()
    for move in prev_moves:
        board.push_uci(move)
    legal_uci_moves = [move.uci() for move in board.legal_moves]
    legal_uci_moves_idx = [config.vocab.index(move) for move in legal_uci_moves]
    legal_uci_moves_scores = [0.1 for _ in legal_uci_moves_idx]

    input_obj = {
        'candidate_scores': [1.0, 0.85, 0.84],
        'candidate_moves': candidate_moves,
        'candidate_moves_idx': candidate_moves_idx,
        'legal_moves_idx': legal_uci_moves_idx,
        'legal_moves_scores': legal_uci_moves_scores,
        'prev_moves': ' '.join(prev_moves)
    }

    print('input_obj:', input_obj)
    positions = [input_obj]

    # 2. Create Dataset
    dataset = FT_DatasetGenerator.create_and_pad_dataset(positions)
    dataset = dataset.batch(1, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(move_ranking_batch_flat, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # 3. Parse Dataset
    first_element = next(iter(dataset.take(5)))

    # 4. Test Move Ranking
    prev_moves_encoded, norm_scores, board_tensor, norm_scores_sample_weights = first_element
    print('prev_moves_encoded:', prev_moves_encoded)
    print('norm_scores:', norm_scores)
    print('norm_scores_sample_weights:', norm_scores_sample_weights)
    print('board_tensor:', board_tensor)

    flat_predictions = tf.reshape(norm_scores, [-1])  # Flatten the tensor
    values, indices = tf.nn.top_k(flat_predictions, k=3)
    print('values:', values)
    print('indices:', indices)
    top_values = values.numpy().tolist()
    top_indices = indices.numpy().tolist()
    top_uci_moves = [config.id2token[i] for i in top_indices]
    print('Top Values: ', top_values)
    print('Top Indices: ', top_indices)
    print('Top UCI Moves: ', top_uci_moves)




def test_window_masking():
    dataset = tf.data.TextLineDataset('/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/chunks_uci/pgn_chunk_0_100000.txt')
    dataset = dataset.map(config.encode_tf_old, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(rand_window_multi, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    first_element = next(iter(dataset.take(1)))
    move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = first_element

    # print('move_seq_masked:', move_seq_masked)
    # print('move_seq_labels:', move_seq_labels)
    # print('move_seq_sample_weights:', move_seq_sample_weights)
    print('board_tensor_masked:', board_tensor_masked)
    print('board_tensor_labels:', board_tensor_labels)
    print('board_tensor_sample_weights:', board_tensor_sample_weights)



def test_dual_objective():
    dataset = tf.data.TextLineDataset(
        '/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/chunks_uci/pgn_chunk_0_100000.txt')
    dataset = dataset.map(config.encode_tf_old, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dual_objective, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    first_element = next(iter(dataset.take(1)))
    move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = first_element

    # print('move_seq_masked:', move_seq_masked)
    # print('move_seq_labels:', move_seq_labels)
    # print('move_seq_sample_weights:', move_seq_sample_weights)
    for x in range(14):
        if x == 0:
            print('Empty Squares')
        elif x < 7:
            print('White Pieces')
        elif x < 13:
            print('Black Pieces')
        else:
            print('Mask')
        print(board_tensor_masked[:, :, x])
    print('board_tensor_labels:', board_tensor_labels)
    print('board_tensor_sample_weights:', board_tensor_sample_weights)
    return 0

def test_dual_objective_flat():
    dataset = tf.data.TextLineDataset(
        '/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/chunks_uci/pgn_chunk_0_100000.txt')
    dataset = dataset.map(config.encode_tf_old, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dual_objective_flat, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    first_element = next(iter(dataset.take(1)))
    move_seq_masked, move_seq_labels, move_seq_sample_weights, board_tensor_masked, board_tensor_labels, board_tensor_sample_weights = first_element

    # print('move_seq_masked:', move_seq_masked)
    # print('move_seq_labels:', move_seq_labels)
    # print('move_seq_sample_weights:', move_seq_sample_weights)
    print(move_seq_sample_weights)
    print('board_tensor_masked:', board_tensor_masked)
    print('board_tensor_labels:', board_tensor_labels)
    print('board_tensor_sample_weights:', board_tensor_sample_weights)
    return 0



def test_ndcg_loss():
    ###############
    ### Example ###
    ###############
    # - There are 5 possible moves, where the first three were evaluated by the model
    y_true = [[1., 2., 3., 4., 5.]]
    for x in range(1900):
        y_true[0].append(0.)
    y_true = tf.convert_to_tensor(y_true)
    print(y_true)

    # Prediction
    y_pred = [[0.1, 0.2, 0.3, 0.4, 5.]]
    for x in range(1900):
        y_pred[0].append(0.)
    y_pred = tf.convert_to_tensor(y_pred)



    # Loss function
    # loss_key = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
    # loss = tfr.losses.make_loss_fn(loss_key)
    # features = {}
    loss = tfr.keras.losses.ApproxNDCGLoss()
    precision = tfr.keras.metrics.PrecisionMetric(topn=3)
    val = loss(y_true, y_pred).numpy()
    pre = precision(y_true, y_pred).numpy()
    print('NDCG loss:', val)
    print('precision:', pre)





def test_mse_loss():
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # acc = tf.keras.metrics.MeanAbsoluteError
    # loss = tf.keras.losses.MeanSquaredError()

    random_tensor = tf.random.uniform(shape=(8, 8, 12))
    random_tensor2 = tf.random.uniform(shape=(8, 8, 12))

    predictions = tf.zeros(shape=(64, 12))
    labels = tf.ones(shape=(64,))

    # sample_weights = tf.ones(shape=(8, 8, 12))

    val = loss(labels, predictions).numpy()
    # val_acc = acc(rand1, rand2, sample_weight=sample_weights).numpy()

    print('CategoricalCrossentropy loss:', val)
    # print('MSE acc:', val_acc)





def test_board_tensor():
    moves = ['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3']

    # create new board
    board = chess.Board()
    for move in moves:
        board.push_uci(move)


    bool_tensor = tf.random.uniform((8, 8), minval=0, maxval=1) < 0.1
    tensor = board_to_tensor_classes(board, bool_tensor)

    for x in range(14):
        if x == 0:
            print('Empty Squares')
        elif x < 7:
            print('White Pieces')
        elif x < 13:
            print('Black Pieces')
        else:
            print('Mask')
        print(tensor[:, :, x])

    return 0


def test_concat_dataset():
    dataset_1 = tf.data.TextLineDataset(
        '/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/chunks_uci/pgn_chunk_0_100000.txt')
    dataset_2 = tf.data.TextLineDataset(
        '/Users/gapaza/repos/gabe/hydra-chess/datasets/pt/millionsbase/chunks_uci/pgn_chunk_2_100000.txt')
    print('Datasets loaded')
    combined = dataset_1.concatenate(dataset_2)
    print('Datasets concatenated')



if __name__ == '__main__':
    print('Testing Strategy')
    # test_window_masking()
    # test_mse_loss()
    # test_dual_objective()
    # test_dual_objective_flat()
    # test_concat_dataset()
    test_move_ranking_flat()


