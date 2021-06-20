from deep_sudoku.utils import sudoku_utils
import numpy as np


def random_move_testing(check_valid_moves):
    sudokus, _ = sudoku_utils.load_latest_sudoku_list()

    for i in range(100):
        board_index = np.random.choice(range(len(sudokus)))
        board = sudokus[board_index][0]
        solved = sudokus[board_index][1]
        new_board = board.copy()

        for j in range(20):
            new_board = sudoku_utils.make_random_move(new_board, solved, valid_move=check_valid_moves)
            assert (np.all(np.logical_or(np.equal(new_board, board), np.equal(new_board, solved))) == check_valid_moves)


def test_random_invalid_moves():
    check_valid_moves = False
    random_move_testing(check_valid_moves)


def test_random_valid_moves():
    check_valid_moves = True
    random_move_testing(check_valid_moves)


def test_random_move():
    sudokus, _ = sudoku_utils.load_latest_sudoku_list()
    for i in range(100):
        board_index = np.random.choice(range(len(sudokus)))
        board = sudokus[board_index][0]
        solved = sudokus[board_index][1]
        new_board = sudoku_utils.make_random_move(board, solved)
        assert (np.sum(~np.equal(board, new_board)) == 1)
