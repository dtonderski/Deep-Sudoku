from typing import Tuple, List, Generator

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
            if file.split('.')[-1] == 'pil':
                current_line = file.split('_')[0]
                if int(current_line) > start_line:
                    start_line = int(current_line)

        with open('data/sudoku_lists/%d_sudokus.pil' % start_line, 'rb') as handle:
            sudokus = pickle.load(handle)

    return sudokus, start_line


def solve_sudoku(board: np.ndarray) -> np.ndarray:
    board_none = board.copy().astype('O')
    if np.any(board == 0):
        board_none[board_none == 0] = None
    puzzle = Sudoku(3)
    puzzle.board = board_none
    return np.array(puzzle.solve().board).astype('uint8')


def validate_sudoku(board: np.ndarray) -> bool:
    puzzle = Sudoku(3)
    puzzle.board = board
    return puzzle.validate()


def load_string(string) -> np.ndarray:
    board = list(map(int, list(string)))
    return np.reshape(board, (9, 9)).astype('uint8')


def make_random_moves(board: np.ndarray, solved: np.ndarray, n_valid_moves: int, n_invalid_moves: int) -> np.ndarray:
    """
    Function that takes in an unsolved and a solved board and makes a number of valid and invalid moves.
    """

    temp_board = board.copy()

    if n_valid_moves > 0:
        possible_moves = np.argwhere(temp_board == 0)
        valid_move_indices = np.random.choice(range(len(possible_moves)), n_valid_moves, replace=False)
        valid_moves = possible_moves[valid_move_indices]
        temp_board[valid_moves[:, 0], valid_moves[:, 1]] = solved[valid_moves[:, 0], valid_moves[:, 1]]

    if n_invalid_moves > 0:
        possible_moves = np.argwhere(temp_board == 0)
        invalid_move_indices = np.random.choice(range(len(possible_moves)), n_invalid_moves, replace=False)
        invalid_moves = possible_moves[invalid_move_indices]
        correct_values = solved[invalid_moves[:, 0], invalid_moves[:, 1]]

        shifting = np.random.choice(range(1, 9), len(correct_values))
        incorrect_values = np.mod(correct_values + shifting, 9)
        incorrect_values[incorrect_values == 0] = 9
        temp_board[invalid_moves[:, 0], invalid_moves[:, 1]] = incorrect_values

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

    with open('data/sudoku_lists/%d_sudokus.pil' % len(seed_list), 'wb') as handle:
        pickle.dump(sudokus, handle)


def permute_sudokus(boards: List[np.ndarray], rng: np.random.Generator) -> List[np.ndarray]:
    permutation = rng.permutation(list(range(1, 10)))
    permuted_boards = boards.copy()

    for i in range(9):
        permuted_boards[boards == i+1] = permutation[i]
    return permuted_boards


def transpose_sudokus(boards: np.ndarray) -> np.ndarray:
    permuted_boards = []
    for board in boards:
        permuted_boards.append(board.transpose())

    return np.array(permuted_boards)


def permute_rows(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted_boards = boards.copy()

    for block in range(3):
        permutation = rng.permutation(list(range(3)))
        for j in range(3):
            permuted_boards[:, block * 3 + j, :] = boards[:, block * 3 + permutation[j], :]
    return permuted_boards


def permute_cols(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted_boards = boards.copy()

    for block in range(3):
        permutation = rng.permutation(list(range(3)))
        for j in range(3):
            permuted_boards[:, :, block * 3 + j] = boards[:, :, block * 3 + permutation[j]]
    return permuted_boards


def permute_row_blocks(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(list(range(3)))
    permuted_boards = boards.copy()
    for i in range(3):
        j = permutation[i]
        permuted_boards[:, i * 3:(i + 1) * 3, :] = boards[:, j * 3:(j + 1) * 3, :]
    return np.array(permuted_boards)


def permute_row_cols(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(list(range(3)))
    permuted_boards = boards.copy()
    for i in range(3):
        j = permutation[i]
        permuted_boards[:, i * 3:(i + 1) * 3, :] = boards[:, j * 3:(j + 1) * 3, :]
    return np.array(permuted_boards)


def augment_sudokus(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        boards = transpose_sudokus(boards)

    augmentation_functions = [permute_sudokus, permute_rows, permute_cols, permute_row_blocks, permute_row_cols]
    for augmentation in augmentation_functions:
        boards = augmentation(boards, rng)

    return boards
