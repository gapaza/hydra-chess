import os
import config
from keras.utils import plot_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time



class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, model_file, save=True):
        super(ValidationCallback, self).__init__()
        self.validation_data = validation_data
        self.model_file = model_file
        self.batch_counter = 0
        self.best_accuracy = 0
        self.save = save

        if 'pt' in config.model_mode:
            self.print_freq = config.pt_batch_val
        elif 'ft' in config.model_mode:
            self.print_freq = config.ft_batch_val

    def on_batch_end(self, batch, logs=None):
        compare_accuracy = -1
        self.batch_counter += 1
        if self.batch_counter % self.print_freq == 0:

            # 1. Execute Validation Step
            print('\n--> VALIDATING MODEL ON BATCH:', batch+1)
            if 'pt' in config.model_mode:
                if config.model_type == 'encoder':
                    compare_accuracy = self.validate_encoder_pt()
                elif config.model_type == 'decoder':
                    compare_accuracy = self.validate_decoder_pt()
            elif 'ft' in config.model_mode:
                if 'ndcg' in config.model_mode:
                    compare_accuracy = self.validate_ndcg_ft()
                elif 'classify' in config.model_mode:
                    compare_accuracy = self.validate_classify_ft()

            # 2. Update Save
            self.update_save(compare_accuracy)


    def validate_encoder_pt(self):
        move_loss, move_accuracy, board_loss, board_accuracy = self.model.evaluate(self.validation_data.take(1000), verbose=1)
        print('Move Loss:', round(move_loss, 4))
        print('Move Accuracy:', round(move_accuracy, 4))
        print('Board Loss:', round(board_loss, 4))
        print('Board Accuracy:', round(board_accuracy, 4))
        return move_accuracy

    def validate_decoder_pt(self):
        move_loss, move_accuracy = self.model.evaluate(self.validation_data.take(1000), verbose=1)
        print('Move Loss:', round(move_loss, 4))
        print('Move Accuracy:', round(move_accuracy, 4))
        return move_accuracy

    def validate_ndcg_ft(self):
        ndcg_loss, accuracy_top_3, accuracy_top_1 = self.model.evaluate(self.validation_data.take(1000), verbose=1)
        print('NDCG Loss:', round(ndcg_loss, 4))
        print('Accuracy Top 3:', round(accuracy_top_3, 4))
        print('Accuracy Top 1:', round(accuracy_top_1, 4))
        return accuracy_top_3

    def validate_classify_ft(self):
        classification_loss, top_move_accuracy = self.model.evaluate(self.validation_data.take(500), verbose=1)
        print('Classification Loss:', round(classification_loss, 4))
        print('Top Move Accuracy:', round(top_move_accuracy, 4))
        return top_move_accuracy


    def update_save(self, compare_accuracy):
        if compare_accuracy > self.best_accuracy:
            self.best_accuracy = compare_accuracy
            if self.save:
                checkpoint = tf.train.Checkpoint(model=self.model)
                save_path = checkpoint.save(self.model_file)
                print('--> Model saved to:', save_path)
