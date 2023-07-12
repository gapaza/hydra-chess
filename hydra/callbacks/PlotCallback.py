import os
import config
from keras.utils import plot_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(PlotCallback, self).__init__()
        self.plot_dir = config.plots_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        self.train_accuracies = []
        self.val_accuracies = []
        self.plot_name = name
        self.time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        train_acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')

        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)

        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Model: {self.plot_name} - Epoch {epoch + 1}')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, f'{self.plot_name}-{self.time}.png'))
        plt.show()

