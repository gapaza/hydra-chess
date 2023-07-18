from hydra.HydraInterface import HydraInterface







if __name__ == '__main__':
    interface = HydraInterface()

    white_pieces = True
    interface.play_interactive_game(user_plays_white=white_pieces)
