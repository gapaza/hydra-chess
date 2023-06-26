import tensorflow as tf
import tensorflow_ranking as tfr
import config
import os

from preprocess.strategies.move_ranking import move_ranking_batch, encode_batch





def test_move_ranking():
    # 1. Test Position
    input_obj = {
        'norm_scores': [1.0, 0.85, 0.84],
        'uci_moves': ['b8c6', 'a7a6', 'g7g6'],
        'prev_moves': ' '.join(['g1f3', 'c7c5', 'e2e4', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3'])
    }
    positions = [input_obj] * 128

    # 2. Create Dataset
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




def test_ndcg_loss():
    ###############
    ### Example ###
    ###############
    # - There are 5 possible moves, where the first three were evaluated by the model
    y_true = [[0., 0.84, 1., 0.85, 0.]]
    y_true = tf.convert_to_tensor(y_true)
    print(y_true)

    # Prediction
    y_pred = [[0.5, 0.5, 0.5, 0.5, 0.5]]
    y_pred = tf.convert_to_tensor(y_pred)

    # Loss function
    # loss_key = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
    # loss = tfr.losses.make_loss_fn(loss_key)
    # features = {}
    loss = tfr.keras.losses.ApproxNDCGLoss()
    val = loss(y_true, y_pred).numpy()
    print(val)



if __name__ == '__main__':
    print('Testing Move Ranking')
    test_ndcg_loss()



