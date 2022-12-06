import numpy as np
from deepsudoku.montecarlo.sudoku_state import SudokuState
from typing import Tuple

def get_n_simulations_function(min_simulations: int, max_simulations: int,
                               difficulty: np.ndarray) -> callable:
    def n_simulations_function(n_zeros: int) -> int:
        n_simulations = int(max(difficulty[int(n_zeros)] * max_simulations,
                                min_simulations))
        return n_simulations

    return n_simulations_function


def run_simulations(node: SudokuState, n_simulations_function: callable,
                    use_steps_taken=True):
    root = node
    start = node.N.sum() if use_steps_taken else 0

    for i in range(start, n_simulations_function(node.n_zeros)):
        while True:
            if node.leaf or node.n_zeros == 0:
                node.leaf = False
                leaf_v = node.V

                # Update all parents of nodes on the way to the root
                while node.last_parent and node.n_zeros < root.n_zeros:
                    action_set = node.action_set
                    for parent in node.parents:
                        parent.update_state(leaf_v, action_set)
                    node = node.last_parent

                # We have reached a leaf and updated all relevant values. Next!
                node = root
                break
            else:
                node = node.get_best_child_simulation()


def play_sudoku_until_failure(node: SudokuState, solution: np.ndarray,
                              n_simulations_function: callable) \
        -> Tuple[SudokuState, bool]:
    successful_game = True

    while node.n_zeros > 0:
        run_simulations(node, n_simulations_function)
        node, move = node.get_best_child_evaluation()
        if not node.is_valid(solution):
            successful_game = False
            break

    return node, successful_game
