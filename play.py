import hydra

from hydra.DecoderInterface import DecoderInterface


if __name__ == '__main__':
    # interface = hydra.HydraInterface()
    # interface.play_interactive_game()

    interface = DecoderInterface()
    interface.play_interactive_game()
