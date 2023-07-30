import tensorflow as tf
import config



class SaveCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_model(self, epoch, batch, logs):
        self.model.hydra_base.save_weights(config.tl_hydra_base_weights_save, save_format='h5')
        self.model.save_weights(config.tl_hydra_full_weights_save, save_format='h5')
        super()._save_model(epoch, batch, logs)
