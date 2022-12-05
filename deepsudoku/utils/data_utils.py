import numpy as np
from typing import Tuple, List, Callable
from deepsudoku.utils import sudoku_utils
from deepsudoku.config import Data
import pickle


def to_categorical(board: np.ndarray) -> np.ndarray:
    """
    Function converting a (9,9) board array to a (9,9,9) categorical array,
    with categorical[i,j,k] == 1 iff board[j,k] == i+1
    :param board: (9,9) board array
    :return: (9,9,9) categorical array
    """
    flat = board.flatten().astype('uint8')
    categorical_flat = np.eye(10)[flat]
    categorical_channels_last = categorical_flat.reshape((9, 9, 10))[:, :, 1:]
    categorical_channels_first = np.moveaxis(categorical_channels_last, -1, 0)
    return categorical_channels_first


def to_numerical(board_cat: np.ndarray) -> np.ndarray:
    """
    Function converting a (9,9,9) categorical array with channels first to a
    (9,9) board array, where board[x,y] = categorical[c,x,y] + 1 if
    categorical[c,x,y] != 0. Inverse of the to_categorical function.
    :param board_cat: (9,9,9) categorical array with channels first
    :return: (9,9) board array
    """
    board_num = np.zeros((9, 9), dtype='uint8')
    for i in np.argwhere(board_cat):
        board_num[i[1], i[2]] = i[0] + 1
    return board_num


def split_data(train_fraction: float = 0.7, val_fraction: float = 0.2,
               test_fraction: float = 0.1, rng_seed: int = 0):
    """
    Function that loads in the latest sudoku list, randomly (using rng_seed)
    splits the sudokus into train, validation and test sets, and dumps each set
    to the location defined in the config file.
    :param train_fraction: fraction of sudokus to use as train data
    :param val_fraction: fraction of sudokus to use as validation data
    :param test_fraction: fraction of sudokus to use as test data
    :param rng_seed: seed used for reproducible randomness.
    :return:
    """
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


def load_data() -> Tuple[List[Tuple[np.ndarray, np.ndarray]],
                         List[Tuple[np.ndarray, np.ndarray]],
                         List[Tuple[np.ndarray, np.ndarray]]]:
    """
    :return: tuple of lists of training, validation, and test sudokus. Each
             list consists of tuples of unsolved and solved sudokus.

    """
    with open(Data.config("train_path"), 'rb') as handle:
        train_sudokus = pickle.load(handle)
    with open(Data.config("val_path"), 'rb') as handle:
        val_sudokus = pickle.load(handle)
    with open(Data.config("test_path"), 'rb') as handle:
        test_sudokus = pickle.load(handle)
    return train_sudokus, val_sudokus, test_sudokus


def uniform_possible_moves_distribution(max_possible_moves: int = 64) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Function returning a uniform distribution over possible moves
    :param max_possible_moves: highest number of moves to make
    :return: tuple of list of moves to make and list of probabilities for them
    """

    possible_numbers_of_moves_to_make = np.array(range(max_possible_moves))
    probabilities = np.ones(max_possible_moves) / max_possible_moves
    return possible_numbers_of_moves_to_make, probabilities


def difficulty_distribution():
    """
    Loads the difficulty, reverses it, discards last element, and returns as a
    distribution. note that len(difficulty) = 65, not 64. We only care about
    elements [1,64] here, as we will not be making 64 moves (which would solve
    the sudoku).
    :return:
    """
    difficulty = load_difficulty()
    possible_numbers_of_moves_to_make = np.array(range(0, 64))
    probabilities = np.flip(difficulty)[0:64]
    probabilities = probabilities/probabilities.sum()
    return possible_numbers_of_moves_to_make, probabilities


def difficulty_uniform_combo_distribution(uniform_scale=0.5) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a distribution that is the normalized sum of the difficulty
    distribution and a uniform distribution scaled by factor uniform_scale
    :param uniform_scale:
    :return:
    """
    difficulty_moves, difficulty_probabilities = difficulty_distribution()
    uniform_moves, uniform_probabilities = (
        uniform_possible_moves_distribution())

    new_probabilities = (difficulty_probabilities + uniform_probabilities *
                         uniform_scale)
    new_probabilities = new_probabilities / new_probabilities.sum()
    return difficulty_moves, new_probabilities


def zero_moves_distribution(max_possible_moves: int = 64) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Function returning a distribution over possible moves with the zero element
    having probability 1. Was used for testing, currently unused.
    :param max_possible_moves: highest number of moves to make
    :return: tuple of list of moves to make and list of probabilities for them
    """
    possible_numbers_of_moves_to_make = np.array(range(0, max_possible_moves))
    probabilities = np.zeros(max_possible_moves)
    probabilities[0] = 1
    return possible_numbers_of_moves_to_make, probabilities


def uniform_invalid_moves_fraction_distribution(linspace_elements=30) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Currently unused
    :param linspace_elements:
    :return:
    """
    possible_invalid_move_fractions = np.linspace(0, 1, linspace_elements)
    probabilities = [1 / linspace_elements] * linspace_elements
    return possible_invalid_move_fractions, np.array(probabilities)


def make_moves(sudokus: List[Tuple[np.ndarray, np.ndarray]],
               n_moves_distribution: Callable =
               uniform_possible_moves_distribution,
               invalid_sudoku_probability: float = 0,
               invalid_moves_fraction_distribution: Callable =
               uniform_invalid_moves_fraction_distribution,
               rng_seed: int = None) \
        -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """
    Take in a distribution of number of moves to make, the probability of
    making ANY invalid moves, and the distribution of fraction of moves
    that are to be invalid, and make the moves.


    :param sudokus: list of tuple of unsolved and solved sudokus in the form of
                    (9,9) numpy arrays
    :param n_moves_distribution: function returning distribution from which
                                  to sample number of moves
    :param invalid_sudoku_probability: probability of making any invalid moves
    :param invalid_moves_fraction_distribution: function returning distribution
                                from which to sample number of invalid moves
    :param rng_seed: seed to generator used for reproducible randomness
    :return: list of (tuple of (unsolved and solved sudokus with moves made and
             bool saying if sudoku is valid))
    """
    n_sudokus = len(sudokus)
    rng = np.random.default_rng(rng_seed)

    possible_numbers_of_moves_to_make, probabilities = n_moves_distribution()
    n_moves = rng.choice(possible_numbers_of_moves_to_make, size=n_sudokus,
                         p=probabilities)

    possible_invalid_move_fractions, probabilities = \
        invalid_moves_fraction_distribution()

    invalid_move_fractions = rng.choice(possible_invalid_move_fractions,
                                        size=n_sudokus, p=probabilities)

    sudoku_invalid = rng.binomial([1] * n_sudokus, invalid_sudoku_probability)
    # make 0 invalid moves if sudoku is supposed to be valid
    n_invalid_moves_to_make = (np.floor(invalid_move_fractions * n_moves)
                               * sudoku_invalid).astype(int)
    n_valid_moves_to_make = n_moves - n_invalid_moves_to_make

    new_sudokus = []
    for i, (board, solved) in enumerate(sudokus):
        new_board = sudoku_utils.make_random_moves(board, solved,
                                                   n_valid_moves_to_make[i],
                                                   n_invalid_moves_to_make[i],
                                                   rng_seed)

        new_sudokus.append((new_board, solved, not sudoku_invalid[i]))
    return new_sudokus


def generate_numpy_batch(sudokus:
List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                         augment: bool = True, rng_seed: int = None) \
        -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

    x, y, valid = [], [], []
    rng = np.random.default_rng(rng_seed)

    for i in range(len(sudokus)):
        board, solved, valid_i = sudokus[i]
        if augment:
            x_aug, y_aug = sudoku_utils.augment_sudokus(
                np.array([board, solved]), rng)
            x.append(x_aug)
            y.append(y_aug)
            valid.append(valid_i)
        else:
            x.append(board)
            y.append(solved)
            valid.append(valid_i)

    return np.array(x), (np.array(y), np.array(valid)[..., np.newaxis])


def fast_generate_numpy_batch(sudokus:
List[Tuple[np.ndarray, np.ndarray, bool]],
                              augment: bool = True, rng_seed: int = None) \
        -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Function that does the same as generate_batch, but each sudoku in the batch
    is augmented in the same way. This way is about 25 times faster than
    augmenting each sudoku individually, and since we do not experience
    overfitting, it is not a problem
    :param sudokus: list of (tuples of (unsolved and solved board pairs in the
                    form of (9,9) arrays and a boolean indicating sudoku
                    validity))
    :param augment: boolean determining whether to use augmentation in batch
                    generation
    :param rng_seed: seed passed to rng for reproducible randomness
    :return: tuple of a (batch_size, 9, 9) array containing unsolved sudokus
             and a tuple of a (batch_size, 9, 9) array with solved sudokus
             and a (batch_size,1) array specifying whether a sudoku is valid.
    """
    rng = np.random.default_rng(rng_seed)

    x, y_board, y_valid = zip(*sudokus)

    if augment:
        sudokus = x + y_board
        augmented_sudokus = sudoku_utils.augment_sudokus(np.array(sudokus),
                                                         rng)
        x = augmented_sudokus[:len(augmented_sudokus) // 2]
        y_board = augmented_sudokus[len(augmented_sudokus) // 2:]
    else:
        x = np.stack(x)
        y_board = np.stack(y_board)

    y_valid = np.array(y_valid)[..., np.newaxis]

    return x, (y_board, y_valid)


def save_difficulty(difficulty: np.ndarray):
    with open(Data.config('difficulty_path'), 'wb') as f:
        pickle.dump(np.array(difficulty), f)


def load_difficulty() -> np.ndarray:
    with open(Data.config('difficulty_path'), 'rb') as f:
        return pickle.load(f)
