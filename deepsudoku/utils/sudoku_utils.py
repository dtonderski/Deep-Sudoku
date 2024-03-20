from typing import Tuple, List

import numpy as np
from sudoku import Sudoku
from deepsudoku.config import Data
import os
import pickle


def load_seed() -> List[str]:
    """
    :return: list of sudoku seeds, where each seed is a string with 81
             characters, 17 of which are non-zero. Each one represents a
             solvable sudoku.
    """
    with open(Data.config("seeds_path"), "r") as f:
        seed_list = f.read().split("\n")
    return seed_list


def generate_sudokus():
    """
    Function that loads the sudoku seeds and the latest sudoku list, solves
    sudokus starting from the last unsolved one, appends a (unsolved, solved)
    tuple to the sudoku list, and saves it to the appropriate directory.
    """
    seed_list = load_seed()

    sudokus, start_line = load_latest_sudoku_list()

    for i, line in enumerate(seed_list[start_line:], start_line):
        sudoku_board = load_string(line)
        solved_board = solve_sudoku(sudoku_board)
        print("Line %d/%d" % (i, len(seed_list)))
        sudokus.append((sudoku_board, solved_board))

        if (i % 100) == 0:
            with open(
                f'{Data.config("sudoku_lists_dir")}/%d_sudokus.pil' % (i + 1),
                "wb",
            ) as handle:
                pickle.dump(sudokus, handle)

    with open(
        f'{Data.config("sudoku_lists_dir")}/%d_sudokus.pil' % len(seed_list),
        "wb",
    ) as handle:
        pickle.dump(sudokus, handle)


def load_latest_sudoku_list() -> (
    Tuple[List[Tuple[np.ndarray, np.ndarray]], int]
):
    """
    Function that loads the latest sudoku list and the number of solved sudokus
    :return: tuple of (list of (tuple of solved and unsolved boards)) and
             the number of solved sudokus
    """
    if not os.path.exists(Data.config("sudoku_lists_dir")):
        os.makedirs(Data.config("sudoku_lists_dir"))

    files = os.listdir(Data.config("sudoku_lists_dir"))

    if len(files) == 0:
        start_line = 0
        sudokus = list()
    else:
        start_line = 0
        for file in files:
            if file.split(".")[-1] == "pil":
                current_line = file.split("_")[0]
                if int(current_line) > start_line:
                    start_line = int(current_line)

        with open(
            f'{Data.config("sudoku_lists_dir")}/%d_sudokus.pil' % start_line,
            "rb",
        ) as handle:
            sudokus = pickle.load(handle)

    return sudokus, start_line


def solve_sudoku(board: np.ndarray) -> np.ndarray:
    """
    :param board: (9,9) board, where unfilled squares are represented by zeros
    :return: (9,9) solved board with no zero elements
    """
    board_none = board.copy().astype("O")
    if np.any(board == 0):
        board_none[board_none == 0] = None
    puzzle = Sudoku(3)
    puzzle.board = board_none
    return np.array(puzzle.solve().board).astype("uint8")


def validate_sudoku(board: np.ndarray) -> bool:
    """
    Return false if there are duplicate entries in any row/block/col.
    :param board: (9,9) board, where unfilled squares are represented by zeros
    :return: true if sudoku is valid, false otherwise
    """

    row_numbers, col_numbers, box_numbers = (
        np.zeros((9, 9)),
        np.zeros((9, 9)),
        np.zeros((3, 3, 9)),
    )

    row_i, col_i = np.where(board)

    for i, j in zip(row_i, col_i):
        row_numbers[i, board[i, j] - 1] += 1
        col_numbers[j, board[i, j] - 1] += 1
        box_numbers[i // 3, j // 3, board[i, j] - 1] += 1

    if (
        np.any(row_numbers > 1)
        or np.any(col_numbers > 1)
        or np.any(box_numbers > 1)
    ):
        return False

    return True


def load_string(string) -> np.ndarray:
    """
    Function that converts a sudoku string to a (9,9) board.
    :param string: string of length 81 representing a sudoku board, with
                   unfilled squares being represented by zeros.
    :return: (9,9) board array.
    """
    board = list(map(int, list(string)))
    return np.reshape(board, (9, 9)).astype("uint8")


def make_random_moves(
    board: np.ndarray,
    solved: np.ndarray,
    n_valid_moves: int,
    n_invalid_moves: int = 0,
    rng_seed: int = None,
) -> np.ndarray:
    """
    Function that takes in an unsolved and a solved board and makes a number of
    valid and invalid random moves.
    :param board: (9,9) potentially unsolved sudoku board
    :param solved: (9,9) solved sudoku board
    :param n_valid_moves: number of valid moves to make
    :param n_invalid_moves: number of invalid, randomly selected moves to make
    :param rng_seed:
    :return: (9,9) board
    """

    new_board = board.copy()
    rng = np.random.default_rng(rng_seed)

    if n_invalid_moves > 0:
        possible_moves = np.argwhere(new_board == 0)
        invalid_move_indices = rng.choice(
            range(len(possible_moves)), n_invalid_moves, replace=False
        )
        invalid_moves = possible_moves[invalid_move_indices]
        correct_values = solved[invalid_moves[:, 0], invalid_moves[:, 1]]

        shifting = rng.choice(range(1, 9), len(correct_values))
        incorrect_values = np.mod(correct_values + shifting, 9)
        incorrect_values[incorrect_values == 0] = 9
        new_board[invalid_moves[:, 0], invalid_moves[:, 1]] = incorrect_values

    if n_valid_moves > 0:
        possible_moves = np.argwhere(new_board == 0)
        n_valid_moves = min(n_valid_moves, len(possible_moves))
        valid_move_indices = rng.choice(
            range(len(possible_moves)), n_valid_moves, replace=False
        )
        valid_moves = possible_moves[valid_move_indices]
        new_board[valid_moves[:, 0], valid_moves[:, 1]] = solved[
            valid_moves[:, 0], valid_moves[:, 1]
        ]

    return new_board


def permute_sudokus(
    boards: np.ndarray, rng: np.random.Generator
) -> List[np.ndarray]:
    """
    Function that permutes sudoku entries (for example, could change all 1s to
    9s and 9s to 1s) while keeping sudoku validity. Used for augmentation
    :param boards: (batch_size, 9, 9) array with sudokus
    :param rng: generator for reproducible randomness
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permutation = rng.permutation(list(range(1, 10)))
    permuted_boards = boards.copy()

    for i in range(9):
        permuted_boards[boards == i + 1] = permutation[i]
    return permuted_boards


def transpose_sudokus(boards: np.ndarray) -> np.ndarray:
    """
    Function transposing sudokus (switching rows with cols)
    :param boards: (batch_size, 9, 9) array with sudokus
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permuted_boards = []
    for board in boards:
        permuted_boards.append(board.transpose())

    return np.array(permuted_boards)


def permute_rows(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Function that permutes the rows in a sudoku. For example, row 0 could be
    switched with row 8
    :param boards: (batch_size, 9, 9) array with sudokus
    :param rng: generator for reproducible randomness
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permuted_boards = boards.copy()
    for block in range(3):
        permutation = rng.permutation(list(range(3)))
        for j in range(3):
            permuted_boards[:, block * 3 + j, :] = boards[
                :, block * 3 + permutation[j], :
            ]
    return permuted_boards


def permute_cols(boards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Function that permutes the cols in a sudoku. For example, col 0 could be
    switched with col 8
    :param boards: (batch_size, 9, 9) array with sudokus
    :param rng: generator for reproducible randomness
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permuted_boards = boards.copy()
    for block in range(3):
        permutation = rng.permutation(list(range(3)))
        for j in range(3):
            permuted_boards[:, :, block * 3 + j] = boards[
                :, :, block * 3 + permutation[j]
            ]
    return permuted_boards


def permute_row_blocks(
    boards: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Function that permutes the row blocks in a sudoku. For example, rows 0-2
    could be switched with rows 3-5
    :param boards: (batch_size, 9, 9) array with sudokus
    :param rng: generator for reproducible randomness
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permutation = rng.permutation(list(range(3)))
    permuted_boards = boards.copy()
    for i in range(3):
        j = permutation[i]
        permuted_boards[:, i * 3 : (i + 1) * 3, :] = boards[
            :, j * 3 : (j + 1) * 3, :
        ]
    return np.array(permuted_boards)


def permute_col_blocks(
    boards: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Function that permutes the col blocks in a sudoku. For example, cols 0-2
    could be switched with cols 3-5
    :param boards: (batch_size, 9, 9) array with sudokus
    :param rng: generator for reproducible randomness
    :return: (batch_size, 9, 9) array with augmented sudokus
    """
    permutation = rng.permutation(list(range(3)))
    permuted_boards = boards.copy()
    for i in range(3):
        j = permutation[i]
        permuted_boards[:, :, i * 3 : (i + 1) * 3] = boards[
            :, :, j * 3 : (j + 1) * 3
        ]
    return np.array(permuted_boards)


def augment_sudokus(
    boards: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Function that applies augmentation functions. Since the transpose function
    is deterministic, it is only applied 50% of the time.
    :param boards: (batch_size, 9, 9) array containing sudokus
    :param rng: numpy random generator used for reproducible randomness
    :return: (batch_size, 9, 9) array containing augmented sudokus
    """
    if rng.random() < 0.5:
        boards = transpose_sudokus(boards)

    augmentation_functions = [
        permute_sudokus,
        permute_rows,
        permute_cols,
        permute_row_blocks,
        permute_col_blocks,
    ]
    for augmentation in augmentation_functions:
        boards = augmentation(boards, rng)

    return boards
