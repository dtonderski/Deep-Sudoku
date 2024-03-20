import torch
from collections import defaultdict
from datetime import datetime
from deepsudoku.montecarlo.sudoku_state import SudokuState
from deepsudoku.montecarlo.simulation import play_sudoku_until_failure
from operator import itemgetter
from typing import List, Tuple, Dict
import numpy as np


def categorical_accuracy(
    x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, eps=1e-5
) -> torch.Tensor:
    mask = y[1][:, :, None, None] * (x == 0)
    return torch.logical_and(
        torch.eq(y_pred[0].argmax(dim=1), y[0]), mask[:, 0]
    ).sum() / (mask.sum() + eps)


def binary_accuracy(y: torch.Tensor, y_pred: torch.Tensor):
    return ((y_pred[1] > 0) == y[1]).sum() / len(y[1])


def evaluate(
    val_sudokus: List[Tuple[np.ndarray, np.ndarray, bool]],
    network: torch.nn.Module,
    n_simulations_function: callable,
    use_PUCTS: bool = False,
    n_played_sudokus: int = 128,
    print_every_n: int = 16,
    verbose: bool = True,
):

    sudoku_packages = val_sudokus[:n_played_sudokus]
    moves_before_ending_dict = defaultdict(list)
    percentage_completed_dict = defaultdict(list)

    for i, (sudoku_board, solution, _) in enumerate(sudoku_packages):
        if verbose and i % print_every_n == 0:
            print(f"{i + 1}/{n_played_sudokus}, time = {datetime.now()}")

        root = SudokuState(
            sudoku_board,
            network,
            simulations_function=n_simulations_function,
            use_PUCTS=use_PUCTS,
        )

        node, successful_game = play_sudoku_until_failure(
            root, solution, n_simulations_function
        )

        starting_blanks = root.n_zeros
        moves_before_ending = (
            starting_blanks if successful_game else len(node.action_set) - 1
        )

        moves_before_ending_dict[starting_blanks // 10].append(
            moves_before_ending
        )
        percentage_completed_dict[starting_blanks // 10].append(
            moves_before_ending / starting_blanks
        )

    return (
        {
            x: y
            for x, y in sorted(
                moves_before_ending_dict.items(), key=itemgetter(0)
            )
        },
        {
            x: y
            for x, y in sorted(
                percentage_completed_dict.items(), key=itemgetter(0)
            )
        },
    )


def get_averages(
    moves_before_ending_dict: Dict[int, List[int]],
    percentage_completed_dict: Dict[int, List[float]],
) -> Tuple[Dict[int, float], Dict[int, float]]:

    avg_moves_dict = {
        x: sum(y) / len(y) for x, y in moves_before_ending_dict.items()
    }
    avg_percentage_dict = {
        x: sum(y) / len(y) for x, y in percentage_completed_dict.items()
    }
    return avg_moves_dict, avg_percentage_dict


def print_evaluation(
    avg_moves_dict: Dict[int, float], avg_percentage_dict: Dict[int, float]
) -> None:
    for (zeros, avg_moves), (_, avg_percentage) in zip(
        avg_moves_dict.items(), avg_percentage_dict.items()
    ):
        print(
            f"{zeros * 10} to {zeros * 10 + 9} zeros: "
            f"average moves before ending: {avg_moves:.1f}, "
            f"avg percentage completed: {avg_percentage * 100:.1f}"
        )


def evaluate_and_print(
    val_sudokus: List[Tuple[np.ndarray, np.ndarray, bool]],
    network: torch.nn.Module,
    n_simulations_function: callable,
    use_PUCTS: bool = False,
    n_played_sudokus: int = 128,
    print_every_n: int = 16,
    verbose: bool = True,
) -> None:
    moves_before_failure_dict, percentage_completed_dict = evaluate(
        val_sudokus,
        network,
        n_simulations_function,
        use_PUCTS,
        n_played_sudokus,
        print_every_n,
        verbose,
    )

    avg_moves_dict, avg_percentage_dict = get_averages(
        moves_before_failure_dict, percentage_completed_dict
    )

    print_evaluation(avg_moves_dict, avg_percentage_dict)
