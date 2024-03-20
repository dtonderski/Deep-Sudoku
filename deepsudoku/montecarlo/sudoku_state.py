from __future__ import annotations

import numpy as np
import torch
from typing import Set, Tuple, List, Dict


class SudokuState:
    sudoku_board: np.ndarray
    network: torch.nn.Module
    use_PUCTS: bool
    c_PUCTS: float
    N: np.ndarray
    W: np.ndarray
    Q: np.ndarray
    N_sum: float
    last_parent: SudokuState
    parents: List[SudokuState]
    leaf: bool
    action_set: Set[Tuple[int, int, int]]
    children: Dict[Tuple[int, int, int], SudokuState]
    encountered_states: List[np.ndarray]
    transposition_table: Dict[int, SudokuState]
    simulations_function: callable
    hash: int

    def __init__(
        self,
        sudoku_board: np.ndarray,
        network: torch.nn.Module,
        simulations_function: callable,
        parent: SudokuState = None,
        action_set: Set[Tuple[int, int, int]] = None,
        encountered_states: List[np.ndarray] = None,
        transposition_table: Dict[int, SudokuState] = None,
        use_PUCTS: bool = False,
        c_PUCTS: float = 1,
    ):
        """

        :param sudoku_board: (..., 9, 9) numpy array, where ... can be any
                             number of preceding ones. Values in [0, 10],
                             where 0s represent blanks
        :param network: pytorch module to use for predictions
        :param simulations_function: function taking current number of blanks
                                     and returning number of simulations to run
        :param parent:
        :param action_set: set of actions ((entry-1, row, col) tuples) taken
                           to get to current state from root state
        :param encountered_states: list of numpy arrays representing sudokus
                                   encountered during simulation
        :param transposition_table: dictionary mapping hash values to
                                    SudokuStates
        """
        # Sudoku board should be numpy array of size (9,9), but can be
        # (1,9,9) or (1,1,9,9), with values in [0, 9]. 0s represent blanks
        self.sudoku_board = sudoku_board
        self.n_zeros = (sudoku_board == 0).sum()
        self.use_PUCTS = use_PUCTS
        self.c_PUCTS = c_PUCTS

        self.N = np.zeros((9, 9, 9), dtype="uint16")
        self.W = np.zeros((9, 9, 9), dtype="float16")
        self.Q = np.zeros((9, 9, 9), dtype="float16") + 0.5

        self.N_sum = 0.001

        self.network = network
        with torch.no_grad():
            sudoku_board_tensor = (
                torch.tensor(sudoku_board)
                .float()
                .reshape((-1, 1, 9, 9))
                .cuda()
            )
            p_raw, v_raw = self.network(sudoku_board_tensor)
            self.P = torch.softmax(p_raw, 1)[0].cpu().numpy()
            self.V = torch.sigmoid(v_raw)[0][0].cpu().numpy()

        self.last_parent = parent
        self.parents = [parent] if parent is not None else []
        self.leaf = True

        self.simulations_function = simulations_function

        # We need this because given a state and a parent we need to quickly
        # compute the action required to go from parent to state, so we can
        # update all actions leading to this state. We can't just store this,
        # because a state can have many parents.
        self.action_set = action_set if action_set is not None else set()
        self.children = {}

        self.encountered_states = (
            encountered_states if encountered_states is not None else []
        )

        self.encountered_states.append(sudoku_board)

        self.transposition_table = (
            transposition_table if transposition_table is not None else dict()
        )
        self.hash = self.calculate_hash()
        self.transposition_table[self.hash] = self

    def get_child_from_action(self, action: Tuple[int, int, int]):
        if action in self.children.keys():
            # Child state is in children of this state
            child = self.children[action]
        else:
            # We have seen the new child state before, but this
            # state does not know that the new state is its child.
            # We must add this relationship to the child and the parent
            child_action_set = self.action_set.union({action})
            child_hash = hash(tuple(child_action_set))
            if child_hash in self.transposition_table.keys():
                child = self.transposition_table[child_hash]
                child.parents.append(self)
                self.children[action] = child
            else:
                # We have never seen the new child state before. Create it
                child_sudoku_board = self.sudoku_board.copy()
                row, col = action[1], action[2]
                entry = action[0] + 1
                child_sudoku_board[row, col] = entry

                self.leaf = False

                child = SudokuState(
                    child_sudoku_board,
                    self.network,
                    self.simulations_function,
                    self,
                    self.action_set.union({action}),
                    self.encountered_states,
                    self.transposition_table,
                    use_PUCTS=self.use_PUCTS,
                    c_PUCTS=self.c_PUCTS,
                )
                self.children[action] = child
        child.last_parent = self
        return child

    def get_best_child_simulation(self):
        if self.use_PUCTS:
            p_scaled = (
                self.c_PUCTS * self.P * np.sqrt(self.N_sum) / (1 + self.N)
            )
        else:
            p_scaled = self.P * (
                1 - self.N / self.simulations_function(self.n_zeros)
            )
        productivity = self.Q + p_scaled

        # We are only interested in nodes where temp_sudoku is 0
        productivity = productivity * (self.sudoku_board == 0)

        # Find where productivity is maximized. Need unravel index
        # because argmax gives a flattened index
        maximum = np.unravel_index(np.argmax(productivity), (9, 9, 9))
        best_action = (int(maximum[0]), int(maximum[1]), int(maximum[2]))

        return self.get_child_from_action(best_action)

    def get_best_child_evaluation(self):
        # When we actually make the move, don't look at the N dependence
        probability = self.Q + self.P

        # We are only interested in nodes where temp_sudoku is 0
        probability = probability * (self.sudoku_board == 0)

        # Find where productivity is maximized. Need unravel index
        # because argmax gives a flattened index for some reason
        maximum = np.unravel_index(np.argmax(probability), (9, 9, 9))
        best_action = (int(maximum[0]), int(maximum[1]), int(maximum[2]))

        return self.get_child_from_action(best_action), maximum

    def update_state(self, leaf_v: float, leaf_action_set):
        # Update all moves
        coords = np.array(list(leaf_action_set - self.action_set))
        indices, rows, cols = coords[:, 0], coords[:, 1], coords[:, 2]

        self.N[indices, rows, cols] += 1
        self.N_sum += 1
        self.W[indices, rows, cols] += leaf_v

        self.Q[indices, rows, cols] = (
            self.W[indices, rows, cols] / self.N[indices, rows, cols]
        )

    def is_valid(self, solution: np.array) -> bool:
        # Solution should be 9 by 9 numpy array with values in [1,9]
        return np.all(
            np.logical_or(
                self.sudoku_board == solution, self.sudoku_board == 0
            )
        )

    def calculate_hash(self) -> int:
        return hash(tuple(self.action_set))

    def __hash__(self) -> int:
        return self.hash
