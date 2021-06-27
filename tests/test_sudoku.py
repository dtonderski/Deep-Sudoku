from deep_sudoku.utils import sudoku_utils
import numpy as np


def count_invalid_moves(new_board, solved):
    return np.sum(np.logical_and(new_board != 0,
                                 np.not_equal(new_board, solved)))


def count_valid_moves(board, new_board, solved):
    return np.sum(np.logical_and.reduce((board == 0,
                                         new_board != 0,
                                         np.equal(new_board, solved))))


def test_random_moves():
    sudokus, _ = sudoku_utils.load_latest_sudoku_list()

    rng = np.random.default_rng(1)

    for n_invalid_moves in range(20):
        for n_valid_moves in range(20):
            board_index = rng.choice(range(len(sudokus)))
            board, solved = sudokus[board_index]

            new_board = sudoku_utils.make_random_moves(board, solved, n_valid_moves, n_invalid_moves)
            assert (count_invalid_moves(new_board, solved) == n_invalid_moves)
            assert (count_valid_moves(board, new_board, solved) == n_valid_moves)


def test_augmentation_functions():
    sudokus, _ = sudoku_utils.load_latest_sudoku_list()
    rng = np.random.default_rng(1)

    augmentation_functions = [sudoku_utils.permute_sudoku, sudoku_utils.transpose_sudoku, sudoku_utils.permute_rows,
                              sudoku_utils.permute_cols, sudoku_utils.permute_row_blocks, sudoku_utils.permute_row_cols]

    for i in range(20):
        board_index = rng.choice(range(len(sudokus)))
        _, solved = sudokus[board_index]

        for augmentation in augmentation_functions:
            new_solved = augmentation(solved, i)
            assert sudoku_utils.validate_sudoku(new_solved)


def test_augmentation():
    sudokus, _ = sudoku_utils.load_latest_sudoku_list()
    rng = np.random.default_rng(1)

    for i in range(20):
        board_index = rng.choice(range(len(sudokus)))
        _, solved = sudokus[board_index]

        new_solved = sudoku_utils.augment_sudoku(solved, i)
        assert sudoku_utils.validate_sudoku(new_solved)
