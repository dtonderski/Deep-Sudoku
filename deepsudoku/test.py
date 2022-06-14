from deepsudoku.utils import sudoku_utils, data_utils
import numpy as np

sudokus, _ = sudoku_utils.load_latest_sudoku_list()
rng = np.random.default_rng(1)
board_index = rng.choice(range(len(sudokus)))
board, solved = sudokus[board_index]

data_utils.to_categorical(board)
