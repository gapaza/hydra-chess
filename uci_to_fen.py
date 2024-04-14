import chess
import chess.pgn


def create_pgn_from_uci(uci_moves, filename="game.pgn"):
    game = chess.pgn.Game()
    node = game

    # Setup a chess board
    board = chess.Board()

    # Split the UCI moves string and apply each move to the board
    for move in uci_moves.split():
        move_obj = chess.Move.from_uci(move)
        if move_obj in board.legal_moves:
            board.push(move_obj)
            node = node.add_variation(move_obj)
        else:
            print(f"Invalid move: {move}")
            break

    # Set headers (optional, but useful for metadata)
    game.headers["Event"] = "Example Game"
    game.headers["Site"] = "Internet"
    game.headers["Date"] = "2024.04.13"
    game.headers["Round"] = "1"
    game.headers["White"] = "Player1"
    game.headers["Black"] = "Player2"
    game.headers["Result"] = "*"  # Or use board.result() if the game is over

    # Write the PGN file
    with open(filename, "w+") as pgn_file:
        print(game, file=pgn_file)


# Example UCI move sequence
uci_moves = "g1f3 g8f6 c2c4 e7e6 b1c3 c7c5 g2g3 b8c6 f1g2 d7d5 c4d5 e6d5 d2d4 c5d4 f3d4 f8c5 d4b3 c5b6 e1g1"
create_pgn_from_uci(uci_moves)







