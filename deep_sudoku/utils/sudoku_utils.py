from typing import Tuple, List

import numpy as np
from sudoku import Sudoku
from deep_sudoku import config as cfg
import os
import pickle


def load_seed() -> List[str]:
    with open(cfg.SEEDS_PATH, 'r') as f:
        seed_list = f.read().split('\n')
    return seed_list


def load_latest_sudoku_list() -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    os.makedirs(cfg.SUDOKU_LISTS_DIR, exist_ok=True)
    files = os.listdir(cfg.SUDOKU_LISTS_DIR)

    if len(files) == 0:
        start_line = 0
        sudokus = list()
    else:
        start_line = 0
        for file in files:
            current_line = file.split('_')[0]
            if int(current_line) > start_line:
                start_line = int(current_line)

        with open('data/sudoku_lists/%d_sudokus.pil' % start_line, 'rb') as handle:
            sudokus = pickle.load(handle)

    return sudokus, start_line


def solve_sudoku(board: np.ndarray) -> np.ndarray:
    board_none = board.copy()
    board_none[board_none == 0] = None
    puzzle = Sudoku(3)
    puzzle.board = board_none
    return np.array(puzzle.solve().board)


def load_string(string) -> np.ndarray:
    board = list(map(int, list(string)))
    return np.reshape(board, (9, 9)).astype('O')


def make_random_move(board: np.ndarray, solved: np.ndarray, valid_move: str = True) -> np.ndarray:
    temp_board = board.copy()
    possible_moves = np.argwhere(temp_board == 0)
    move_index = np.random.choice(range(len(possible_moves)))
    pos_to_play = possible_moves[move_index]

    correct_number = int(solved[pos_to_play[0], pos_to_play[1]])

    if valid_move:
        new_number = correct_number
    else:
        new_number = np.random.choice([x for x in range(1, 9) if x != correct_number])

    temp_board[pos_to_play[0], pos_to_play[1]] = new_number

    return temp_board


def generate_sudokus():
    seed_list = load_seed()

    sudokus, start_line = load_latest_sudoku_list()

    for i, line in enumerate(seed_list[start_line:], start_line):
        sudoku_board = load_string(line)
        solved_board = solve_sudoku(sudoku_board)
        print("Line %d/%d" % (i, len(seed_list)))
        sudokus.append((sudoku_board, solved_board))

        if (i % 100) == 0:
            with open('data/sudoku_lists/%d_sudokus.pil' % (i + 1), 'wb') as handle:
                pickle.dump(sudokus, handle)

    with open('%d_sudokus.pil' % len(seed_list), 'wb') as handle:
        pickle.dump(sudokus, handle)
