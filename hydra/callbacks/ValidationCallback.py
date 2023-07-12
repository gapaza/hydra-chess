import os
import config
from keras.utils import plot_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time



class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, model_file, print_freq=2000, save=True):
        super(ValidationCallback, self).__init__()
        self.validation_data = validation_data
        self.model_file = model_file
        self.print_freq = print_freq
        self.batch_counter = 1
        self.best_t1 = 0
        self.save = save

    def on_batch_end(self, batch, logs=None):
        if self.batch_counter % self.print_freq == 0:
            print('\n--> VALIDATING MODEL ON BATCH:', batch)
            if config.mode == 'pt':
                loss, accuracy, accuracy_t1, accuracy_t2 = self.model.evaluate(self.validation_data.take(500), verbose=1)
                print(f'Validation move loss after batch {self.batch_counter}: {round(loss, 4)}')
                print(f'Validation move accuracy after batch {self.batch_counter}: {round(accuracy, 4)}')
                print(f'Validation board loss after batch {self.batch_counter}: {round(accuracy_t1, 4)}')
                print(f'Validation board accuracy after batch {self.batch_counter}: {round(accuracy_t2, 4)}')
                accuracy_t1 = accuracy
            elif config.mode == 'ft':
                loss, accuracy, accuracy_t1 = self.model.evaluate(self.validation_data.take(500), verbose=1)
                print(f'Validation loss after batch {self.batch_counter}: {round(loss, 4)}')
                print(f'Validation accuracy after batch {self.batch_counter}: {round(accuracy, 4)}')
                print(f'Validation accuracy_t1 after batch {self.batch_counter}: {round(accuracy_t1, 4)}')

            if accuracy_t1 > self.best_t1:
                self.best_t1 = accuracy_t1
                if self.save:
                    self.model.save_weights(
                        self.model_file, overwrite=True
                    )
                    print('--> Model saved to:', self.model_file)
                # checkpoint = tf.train.Checkpoint(model=self.model)
                # save_path = checkpoint.save(self.model_file)
                # print('--> Model saved to:', save_path)
        self.batch_counter += 1

