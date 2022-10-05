import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import chess


def create_ai():
    model = tf.keras.models.load_model('model_weights')
    return model


# return index of move to be played from board.legal_moves
def play_next_move(model, board):
    board_scores = []
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(move)
        board.push(move)
        converted_board = np.transpose(np.array(utils.convert_board_to_input(board)))
        converted_board = converted_board[np.newaxis, :]
        board_scores.append(model.predict(converted_board, verbose=0))
        board.pop()
    print(board_scores)
    index_max = np.argmin(np.array(board_scores))
    print(index_max)
    return legal_moves[index_max]


if __name__ == '__main__':
    board = chess.Board()
    model = create_ai()
    print("Game beginning, you are white, input move and all future moves in algebraic chess notation")
    print(board)

    while True:
        next_move_valid = False
        while not next_move_valid:
            next_move_string = input("Input move")
            next_move = chess.Move.from_uci(next_move_string)
            if next_move in board.legal_moves:
                board.push(next_move)
                print(board)
                next_move_valid = True
            else:
                print("move not legal or notation not valid, please input again")
        board.push(play_next_move(model, board))
        print("AI has played, printing updated board")
        print(board)
