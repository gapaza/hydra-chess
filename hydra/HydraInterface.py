import argparse
import time
import config
import platform
import tensorflow as tf
import os
from copy import deepcopy
import random
import chess
import chess.svg
from hydra import HydraEncoderModel
from hydra import HydraDecoderModel
from preprocess.strategies import py_utils


class HydraInterface:

    def __init__(self):
        self.mode = config.model_mode
        self.prediction_mask = True
        self.user_plays_white = True

        # --> Load Model
        self.model = None
        if config.model_type == 'encoder':
            self.model = HydraEncoderModel.build_model(config.model_mode)
        elif config.model_type == 'decoder':
            self.model = HydraDecoderModel.build_model(config.model_mode)
        self.checkpoint = tf.train.Checkpoint(self.model)
        self.checkpoint.restore(config.tl_interface_checkpoint).expect_partial()

        # --> Chess Board
        self.board = chess.Board()
        self.move_history = []

        # --> Unicode Pieces
        self.unicode_pieces = {
            'r': u'♖', 'R': u'♜',
            'n': u'♘', 'N': u'♞',
            'b': u'♗', 'B': u'♝',
            'q': u'♕', 'Q': u'♛',
            'k': u'♔', 'K': u'♚',
            'p': u'♙', 'P': u'♟',
            None: ' '
        }

    def save_svg(self, filename='board.svg'):
        svg = chess.svg.board(board=self.board, flipped=(not self.user_plays_white))
        full_path = os.path.join(config.plots_dir, filename)
        with open(full_path, 'w') as f:
            f.write(svg)

    def new_game(self):
        print('--> STARTING NEW GAME')
        self.board = chess.Board()
        self.move_history = []
        self.save_svg()

    def random_position(self, num_moves=15):
        for _ in range(num_moves):
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            self.board.push(move)
            self.move_history.append(move.uci())

            # If the game is over, reset the board to the initial state
            if self.board.is_game_over():
                self.board.reset()

        # Ensure it's white's turn
        if not self.board.turn:
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                self.board.reset()
            else:
                move = random.choice(legal_moves)
                self.board.push(move)
                self.move_history.append(move.uci())

    def play_interactive_game(self, user_plays_white=True):
        self.user_plays_white = user_plays_white
        self.new_game()
        # self.random_position()
        while not self.board.is_game_over():
            print(self.board)
            print(self.move_history)

            if self.board.turn == chess.WHITE and user_plays_white or \
                    self.board.turn == chess.BLACK and not user_plays_white:
                user_move = input("Your move (in UCI format, e.g. 'e2e4'): ")
                if user_move == 'exit':
                    break
                try:
                    move = chess.Move.from_uci(user_move)
                    if move in self.board.legal_moves:
                        self.move_history.append(user_move)
                        self.board.push(move)
                    else:
                        print("Illegal move. Please enter a valid move:", self.board.legal_moves)
                except ValueError:
                    print("Invalid input. Please enter a valid move in UCI format.")
            else:
                # cpu_move = self.random_move()
                model_move = self.model_move()
                self.move_history.append(model_move.uci())
                print(f"CPU move: {model_move.uci()}")
                self.board.push(model_move)
                self.save_svg()

        print("Game Over.")
        print(self.board.result())

    def random_move(self):
        legal_moves = list(self.board.legal_moves)
        return random.choice(legal_moves)

    def model_move(self):
        legal_moves = list(self.board.legal_moves)

        if self.user_plays_white is False and len(self.move_history) == 0:
            return chess.Move.from_uci('d2d4')


        # 1. Get board tensor
        curr_moves = ' '.join(self.move_history)
        curr_moves_encoded = tf.convert_to_tensor(config.encode(curr_moves))
        curr_board_tensor = py_utils.get_sequence_board_tensor_classes_flat(curr_moves_encoded)
        # print('Curr Moves: ', curr_moves)
        # print('Curr Moves Encoded: ', curr_moves_encoded)
        print('Curr Board Tensor:\n', curr_board_tensor)
        # print(self.board)

        # 2. Get move sequence with mask
        move_input = deepcopy(self.move_history)
        if self.prediction_mask is True:
            move_input.append('[mask]')
        move_input = ' '.join(move_input)
        move_input = tf.convert_to_tensor(config.encode(move_input))
        print('Move input:\n', move_input)

        # 3. Get model prediction
        try:
            curr_board_tensor = tf.expand_dims(curr_board_tensor, axis=0)
            curr_board_tensor = tf.cast(curr_board_tensor, tf.float32)
            move_input = tf.expand_dims(move_input, axis=0)
            predictions = self.model.predict([curr_board_tensor, move_input])
            print('Predictions: ', predictions)

            flat_predictions = tf.reshape(predictions, [-1])  # Flatten the tensor
            values, indices = tf.nn.top_k(flat_predictions, k=3)
            top_values = values.numpy().tolist()
            top_indices = indices.numpy().tolist()
            top_uci_moves = [config.id2token[i] for i in top_indices]
            print('Top Values: ', top_values)
            print('Top Indices: ', top_indices)
            print('Top UCI Moves: ', top_uci_moves)
            for top_move in top_uci_moves:
                move = chess.Move.from_uci(top_move)
                if move in legal_moves:
                    return move
            print('--> ERROR, NO LEGAL MOVE SELECTED BY MODEL')
            exit(0)

        except Exception as e:
            print("Error in model prediction: ", e)
            print("Selecting random move")
            return random.choice(legal_moves)
