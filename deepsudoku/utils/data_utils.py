import numpy as np
from typing import Tuple, List, Callable
from deepsudoku.utils import sudoku_utils
from deepsudoku.config import Data
import pickle


def to_categorical(board: np.ndarray) -> np.ndarray:
    flat = board.flatten().astype('uint8')
    categorical_flat = np.eye(10)[flat]
    categorical_channels_last = categorical_flat.reshape(9, 9, 10)[:, :, 1:]
    categorical_channels_first = np.moveaxis(categorical_channels_last, -1, 0)
    return categorical_channels_first


def to_numerical(board_cat: np.ndarray) -> np.ndarray:
    board_num = np.zeros((9, 9), dtype='uint8')
    for i in np.argwhere(board_cat):
        board_num[i[1], i[2]] = i[0] + 1
    return board_num


def split_data(train_fraction: float = 0.7, val_fraction: float = 0.2,
               test_fraction: float = 0.1, rng_seed: int = 0):
    sudokus, start_line = sudoku_utils.load_latest_sudoku_list()
    assert (train_fraction + val_fraction + test_fraction <= 1)
    indices = list(range(len(sudokus)))
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(indices)
    val_start_index = int(len(sudokus) * train_fraction)
    test_start_index = int(len(sudokus) * (train_fraction + val_fraction))

    with open(Data.config("train_path"), 'wb') as handle:
        pickle.dump(sudokus[:val_start_index], handle)

    with open(Data.config("val_path"), 'wb') as handle:
        pickle.dump(sudokus[val_start_index:test_start_index], handle)

    with open(Data.config("test_path"), 'wb') as handle:
        pickle.dump(sudokus[test_start_index:], handle)


def load_data() \
        -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
                 List[Tuple[np.ndarray, np.ndarray]],
                 List[Tuple[np.ndarray, np.ndarray]]]:
    with open(Data.config("train_path"), 'rb') as handle:
        train_sudokus = pickle.load(handle)
    with open(Data.config("val_path"), 'rb') as handle:
        val_sudokus = pickle.load(handle)
    with open(Data.config("test_path"), 'rb') as handle:
        test_sudokus = pickle.load(handle)
    return train_sudokus, val_sudokus, test_sudokus


def uniform_possible_moves_distribution(max_possible_moves: int = 64) \
        -> Tuple[List[int], List[float]]:

    possible_numbers_of_moves_to_make = list(range(0, max_possible_moves))
    probabilities = [1 / max_possible_moves] * max_possible_moves
    return possible_numbers_of_moves_to_make, probabilities


def zero_moves_distribution(max_possible_moves: int = 64) \
        -> Tuple[List[int], List[int]]:
    possible_numbers_of_moves_to_make = list(range(0, max_possible_moves))
    probabilities = [0] * max_possible_moves
    probabilities[0] = 1
    return possible_numbers_of_moves_to_make, probabilities


def make_moves(sudokus: List[Tuple[np.ndarray, np.ndarray]],
               distribution_function: Callable =
               uniform_possible_moves_distribution,
               rng_seed: int = None) \
        -> List[Tuple[np.ndarray, np.ndarray]]:
    possible_numbers_of_moves_to_make, probabilities = distribution_function()
    n_sudokus = len(sudokus)

    rng = np.random.default_rng(rng_seed)
    numbers_of_moves_to_make = rng.choice(possible_numbers_of_moves_to_make,
                                          size=n_sudokus, p=probabilities)

    new_sudokus = []
    for i, (board, solved) in enumerate(sudokus):
        new_board = sudoku_utils.make_random_moves(board, solved,
                                                   numbers_of_moves_to_make[i])
        new_sudokus.append((new_board, solved))
    return new_sudokus


def generate_batch(sudokus: List[Tuple[np.ndarray, np.ndarray]],
                   augment: bool = True, rng_seed: int = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to generate batches. The main problem is this - we want
    reproducible randomness. If we are to generate batches for 100 epochs,
    we want the augmentation to be different for each epoch...
    TODO: finish this
    :param sudokus:
    :param augment:
    :param rng_seed:
    :return:
    """

    x, y = [], []
    rng = np.random.default_rng(rng_seed)

    for i in range(len(sudokus)):
        board, solved = sudokus[i]
        if augment:
            x_aug, y_aug = sudoku_utils.augment_sudokus(
                np.array([board, solved]), rng)
            x.append(x_aug)
            y.append(y_aug)
        else:
            x.append(board)
            y.append(solved)

    return np.array(x), np.array(y)


def fast_generate_batch(sudokus: List[Tuple[np.ndarray, np.ndarray]],
                        augment: bool = True, rng_seed: int = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Function that does the same as generate_batch, but each sudoku in the batch
    is augmented in the same way. This way is about 25 times faster.
    :param sudokus:
    :param augment:
    :param rng_seed:
    :return:
    """
    x, y = [], []
    rng = np.random.default_rng(rng_seed)

    for sudoku in sudokus:
        x.append(sudoku[0])
        y.append(sudoku[1])

    if augment:
        sudokus = x + y
        augmented_sudokus = sudoku_utils.augment_sudokus(np.array(sudokus),
                                                         rng)
        x = augmented_sudokus[:len(augmented_sudokus) // 2]
        y = augmented_sudokus[len(augmented_sudokus) // 2:]

    return x, y
