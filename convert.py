import tensorflow as tf
import config
import os

def save_model_weights(model_path, weights_file):
    model = tf.keras.models.load_model(model_path)
    model.save_weights(weights_file)




if __name__ == '__main__':
    model_path = '/home/ubuntu/hydra-chess/models/hydra-family/hydra-base-backup-2'
    weights_path = os.path.join(config.weights_dir, config.tl_model_class, 'hydra-base-backup-2')
    save_model_weights(model_path, weights_path)






























