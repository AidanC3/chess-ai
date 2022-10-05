from PyPDF2 import PdfReader
import numpy as np
import os
import chess
import chess.pgn


def import_data():
    file = "chess_puzzles.pdf"

    reader = PdfReader(file)
    number_of_pages = len(reader.pages)

    vec_list = []
    for i in range(50, 600):
        page = reader.pages[i]
        text = page.extract_text()
        text = text.split()
        # print(text)
        game = ""
        counter = 0
        for line in text:
            if len(line) == 9 and line[0].isnumeric():
                if counter < 8:
                    for char in line:
                        game += char
                    counter += 1
                else:
                    vec_list.append(game)
                    game = ""
                    for char in line:
                        game += char
                    counter = 1
    # print(vec_list)
    # print(len(vec_list))

    # check each vector is correct length

    sum = 0
    for vec in vec_list:
        sum += len(vec)
    print(str(sum / len(vec_list)))

    return vec_list


def convert_string_to_nums_list(string):
    converted_list = []
    for char in string:
        if char == 'Z':
            converted_list.append(0)
        elif not char.isnumeric():
            converted_list.append(ord(char))
        else:
            converted_list.append(int(char))
    return converted_list


def convert_to_black(string):
    return string.swapcase()


def clean_and_convert(game_list):
    del game_list[0:63:8]
    game_vector = np.array(game_list)
    return game_vector


def rotate_board(game_vector):
    game_vector = np.resize(game_vector, (8, 8))
    game_vector = np.rot90(game_vector, k=2)
    game_vector = np.resize(game_vector, (64,))
    return game_vector


def generate_boards(moves_from_win, dir_path):

    file_list = []
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            file_list.append(os.path.join(dir_path, path))

    board_list = []

    for file in file_list:
        good_game = True
        #print(file)
        game = chess.pgn.read_game(open(file))
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        try:
            for i in range(moves_from_win):
                board.pop()
        except IndexError:
            good_game = False
        if game.headers['Result'] != "1/2-1/2" and good_game:
            #print("board")
            #print(board)
            board_list.append((board, game.headers['Result']))
    cleaned_board_list = []
    for board in board_list:
        cleaned_board_list.append(convert_algebraic_to_puzzle(board))
    return cleaned_board_list


#also tacks on classification of 0 or 1 on end of list to signify white or black wins
def convert_algebraic_to_puzzle(board):
    converted_board = []
    cleaned_board_string = "".join(str(board[0]).split())

    for char in cleaned_board_string:
        if char == '.':
            converted_board.append(0)
        else:
            converted_board.append(ord(char))
    if board[1] == "0-1":
        converted_board.append(0)
    else:
        converted_board.append(1)
    return converted_board

def convert_board_to_input(board):
    converted_board = []
    cleaned_board_string = "".join(str(board).split())
    #print(cleaned_board_string)
    for char in cleaned_board_string:
        if char == '.':
            converted_board.append(0)
        else:
            converted_board.append(ord(char))
    #print(converted_board)
    return converted_board


if __name__ == '__main__':
    boards = generate_boards()
    for board in boards:
        print(len(board))


