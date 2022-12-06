import random
from datetime import datetime

import torch

from deepsudoku.montecarlo.sudoku_state import SudokuState
from deepsudoku.montecarlo.simulation import play_sudoku_until_failure

from typing import List, Tuple
import numpy as np


def generate_training_data(train_sudokus:
                           List[Tuple[np.ndarray, np.ndarray, bool]],
                           network: torch.nn.Module,
                           n_simulations_function: callable,
                           use_PUCTS: bool = False,
                           min_data_size: int = 4096,
                           verbose: int = 0) \
        -> List[Tuple[np.ndarray, np.ndarray, bool]]:
    """

    :param train_sudokus: list of tuples (sudoku, solution, valid)
    :param network:
    :param n_simulations_function: 
    :param use_PUCTS: 
    :param min_data_size: minimum data size to save
    :param verbose: 0,1, or 2
    :return: list of tuples of (sudoku, solution, valid) containing states 
             encountered during solving sudoku
    """
    # Shuffle not in place
    sudokus = random.sample(train_sudokus, len(train_sudokus))

    saved_states = []
    valids = []
    sudokus_sampled_from = 0

    if verbose:
        print(f"Sampled {len(saved_states)}/{min_data_size} sudokus, "
              f"time = {datetime.now()}")

    print_j = 1

    for i, sudoku_package in enumerate(sudokus):
        sudoku_board, solution, _ = sudoku_package

        root = SudokuState(sudoku_board, network,
                           simulations_function=n_simulations_function,
                           use_PUCTS=use_PUCTS)

        node, successful_game = play_sudoku_until_failure(
            root, solution, n_simulations_function)

        if not successful_game:
            sudokus_sampled_from += 1
            if verbose > 1:
                print(f"{node.n_zeros} zeros, "
                      f"{len(node.encountered_states)} encountered states")

            current_valids = []
            for encountered_state in node.encountered_states:
                valid = np.all(np.logical_or(encountered_state == 0,
                                             encountered_state == solution))

                valids.append(valid)
                current_valids.append(valid)
                saved_states.append((encountered_state, solution, valid))

            if verbose:
                if len(saved_states) // (min_data_size // 8) >= print_j:
                    print_j += 1
                    print(f"Sampled {len(saved_states)}/{min_data_size} "
                          f"sudokus, time = {datetime.now()}")

            if verbose > 1:
                print(f"Current valids fraction: "
                      f"{sum(current_valids) / len(current_valids):.2f}")

        if len(saved_states) >= min_data_size:
            break

    if not saved_states:
        print("============= ALL SUDOKUS SOLVED SUCCESSFULLY! =============")
        print("This will cause code to break.")
        return saved_states

    if verbose:
        print(f"Valids fraction: {sum(valids) / len(valids):.2f}, "
              f"sampled from {sudokus_sampled_from} sudokus")

    return saved_states
