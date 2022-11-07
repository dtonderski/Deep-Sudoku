from collections import defaultdict
from typing import List, Tuple
import torch
import numpy as np
import random
from datetime import datetime
from operator import itemgetter


def tensor_action_to_dict_key(tensor_action_tuple: Tuple[torch.Tensor, Tuple]):
    return tensor_to_dict_key(tensor_action_tuple[0]) + tensor_action_tuple[1]


def tensor_to_dict_key(tensor):
    return tuple(tensor.flatten().tolist())


class TensorDict(defaultdict):
    def __getitem__(self, key):
        return defaultdict.__getitem__(self, tensor_to_dict_key(key))

    def __setitem__(self, key, value):
        # Need this because otherwise we call tensor_to_dict_key twice when
        # getting a key that doesn't exist
        if type(key) is not tuple:
            key = tensor_to_dict_key(key)
        defaultdict.__setitem__(self, key, value)


def get_best_move(sudoku, Q_dict, PV_dict, solution=None, verbose=False):
    probability = Q_dict[sudoku] + PV_dict[sudoku][0]
    probability_masked = probability * (sudoku[0] == 0)

    eval_max = np.unravel_index(np.argmax(probability_masked.cpu()), (9, 9, 9))
    if verbose:
        print(eval_max)
        if solution is not None:
            print(solution[eval_max[1], eval_max[2]])
            print(solution[eval_max[1], eval_max[2]] == eval_max[0])
    return eval_max


def make_best_move(sudoku, Q_dict, PV_dict, N_dict, solution=None,
                   verbose=False):
    eval_max = get_best_move(sudoku, Q_dict, PV_dict, solution, verbose)
    N = N_dict[sudoku][eval_max]
    sudoku[0, 0, eval_max[1], eval_max[2]] = eval_max[0] + 1
    return N


def run_simulations(sudoku, network, simulations_function, N_dict=None,
                    Q_dict=None,
                    W_dict=None, PV_dict=None, steps_already_made=0,
                    warm_start=False, verbose=1, debug=False):
    if ((N_dict is None or Q_dict is None or W_dict is None or PV_dict is None)
            or not warm_start):
        if verbose >= 1:
            print("Resetting dictionaries")
        N_dict = TensorDict(lambda: torch.zeros((9, 9, 9)).long().cuda())
        Q_dict = TensorDict(lambda: (torch.zeros((9, 9, 9)) + 0.5).cuda())
        W_dict = defaultdict(float)
        PV_dict = TensorDict()
    n_zeros = (sudoku == 0).sum()
    steps = simulations_function(n_zeros)
    for i in range(steps_already_made, steps):
        if verbose >= 1:
            print(f"Iteration {i + 1}/{steps}")
        j = -1

        # Empty traversed edges, clone original sudoku tensor
        edges: List[Tuple[torch.Tensor, Tuple]] = []
        temp_sudoku = sudoku.clone()

        # Run one iteration of the simulation
        while True:
            j += 1
            if verbose >= 2 and j > 1:
                print(f"j = {j}")

            # If node has been seen before, load p and v values from network.
            # Else, run network on sudoku
            if tensor_to_dict_key(temp_sudoku) not in PV_dict.keys():
                p_raw, v_raw = network(temp_sudoku)

                v = torch.sigmoid(v_raw)
                p = torch.softmax(p_raw, 1)[0]

                PV_dict[temp_sudoku] = (p, v)
                leaf = True

                if verbose >= 3:
                    print("Unseen node reached!")
            else:
                p, v = PV_dict[temp_sudoku]
                leaf = False

                if verbose >= 3:
                    print("Previously seen node reached!")

            # End of game reached. This is needed because the end of the game
            # might have been reached before, in which case leaf would be False
            # above
            if (temp_sudoku == 0).sum() == 0:
                if verbose >= 3:
                    print("Sudoku completed!")
                # If last node appended to edges has been played more than
                # once, we can cancel the simulation, as it will not go
                # anywhere
                state, action = edges[-1]
                if N_dict[state][action] > 1:
                    if debug:
                        print("Reached end node more than once, breaking sim!")
                    return N_dict, Q_dict, W_dict, PV_dict
                leaf = True

            # If we have reached a leaf, update each (state,action) pair for
            # every traversed edge and end simulation. Otherwise, continue.
            if leaf:
                if verbose >= 3:
                    print("Leaf reached!")

                for state, action in edges:
                    N_current = N_dict[state][action]
                    N_dict[state][action] += 1
                    W_dict[tensor_action_to_dict_key((state, action))] += (
                        v[0][0])

                    Q_dict[state][action] = (
                            W_dict[tensor_action_to_dict_key((state, action))]
                            / N_current)

                break
            else:
                if verbose >= 3:
                    print("Non-leaf reached!")

                current_zeros = n_zeros - j
                # print(f"{n_zeros=},{j=},{current_zeros=}")
                Q = Q_dict[temp_sudoku]
                p_scaled = p * (1 - N_dict[temp_sudoku] / simulations_function(
                    current_zeros))
                productivity = Q + p_scaled

                # We are only interested in nodes where temp_sudoku is 0
                productivity = productivity * (temp_sudoku[0] == 0)

                # Find where productivity is maximized. Need unravel index
                # because argmax gives a flattened index for some reason
                maximum = np.unravel_index(torch.argmax(productivity).cpu(),
                                           (9, 9, 9))
                max_row, max_col = maximum[1], maximum[2]
                max_entry = maximum[0] + 1

                # Append last traversed state action pair to the edges list
                # and update temp_sudoku. We need to clone temp_sudoku as
                # otherwise modifying temp_sudoku will also modify the tensor
                # in our edges list

                edges.append((temp_sudoku.clone(), maximum))
                temp_sudoku[0, 0, max_row, max_col] = max_entry

                if verbose >= 3:
                    print(f"Maximum productivity: {productivity[maximum]}")
                    print(f"Sudoku updated with {maximum}")

    return N_dict, Q_dict, W_dict, PV_dict


def evaluate_on_validation(val_sudokus, network,
                           n_played_sudokus=64,
                           min_simulations=4,
                           max_simulations=256,
                           verbose=0):
    sudokus = val_sudokus[:n_played_sudokus]
    moves_before_ending_dict = defaultdict(list)
    percentage_completed_dict = defaultdict(list)
    for i, (sudoku, solution, valid) in enumerate(sudokus):
        zeros = (sudoku == 0).sum()
        print_every_n = min(16, n_played_sudokus)
        if i % (n_played_sudokus // print_every_n) == 0:
            print(f"{i + 1}/{n_played_sudokus}, time = {datetime.now()}")
        sudoku = torch.Tensor(sudoku).reshape((1, 1, 9, 9)).cuda()
        solution = torch.Tensor(solution).reshape((1, 1, 9, 9)).cuda()

        moves_before_ending, _, _ = (
            play_sudoku_until_failure(sudoku, solution,
                                      network,
                                      get_simulations_function(
                                          min_simulations,
                                          max_simulations),
                                      verbose))
        moves_before_ending_dict[zeros // 10].append(moves_before_ending)
        percentage_completed_dict[zeros // 10].append(
            moves_before_ending / zeros)

    return ({x: y for x, y in sorted(moves_before_ending_dict.items(),
                                     key=itemgetter(0))},
            {x: y for x, y in sorted(percentage_completed_dict.items(),
                                     key=itemgetter(0))})


def get_averages(moves_before_ending_dict, percentage_completed_dict):
    avg_moves_dict = {x: sum(y) / len(y) for x, y in
                      moves_before_ending_dict.items()}
    avg_percentage_dict = {x: sum(y) / len(y) for x, y in
                           percentage_completed_dict.items()}
    return avg_moves_dict, avg_percentage_dict


def print_evaluation(avg_moves_dict, avg_percentage_dict):
    for (zeros, avg_moves), (_, avg_percentage) \
            in zip(avg_moves_dict.items(), avg_percentage_dict.items()):
        print(
            f"{zeros * 10} to {zeros * 10 + 9} zeros: "
            f"average moves before ending: {avg_moves:.1f}, "
            f"avg percentage completed: {avg_percentage * 100:.1f}")


def generate_training_data(train_sudokus, network,
                           min_simulations=4,
                           max_simulations=256,
                           min_data_size=4096,
                           verbose=0):
    sampled_sudokus = random.sample(train_sudokus, len(train_sudokus))

    valids = []
    saved_states = []

    print(
        f"Sampled {len(saved_states)}/{min_data_size} sudokus, "
        f"time = {datetime.now()}")

    print_j = 0

    for i, sudoku_package in enumerate(sampled_sudokus):
        sudoku, solution, _ = sudoku_package
        sudoku = torch.Tensor(sudoku).reshape((1, 1, 9, 9)).cuda()

        _, PV_dict, successful_game = (
            play_sudoku_until_failure(sudoku,
                                      solution,
                                      network,
                                      get_simulations_function(
                                          min_simulations,
                                          max_simulations),
                                      verbose=0))

        if not successful_game:
            encountered_states = PV_dict.keys()
            if verbose:
                print(
                    f"{(sudoku == 0).sum()} zeros, {len(encountered_states)} "
                    f"encountered states")
            saved_states_iteration = [np.reshape(x, (9, 9)) for x in
                                      list(encountered_states)]

            current_valids = []
            for sudoku in saved_states_iteration:
                valid = np.all(np.logical_or(sudoku == 0, sudoku == solution))
                valids.append(valid)
                current_valids.append(valid)
                saved_states.append((sudoku, solution, valid))

            if len(saved_states) // (min_data_size // 8) >= print_j:
                print_j += 1
                print(
                    f"Sampled {len(saved_states)}/{min_data_size} sudokus, "
                    f"time = {datetime.now()}")

            if verbose:
                print(
                    f"Current valids fraction: "
                    f"{sum(current_valids) / len(current_valids):.2f}")

        if len(saved_states) >= min_data_size:
            break

    # All sudokus were successful, wtf
    if not saved_states:
        print(
            "=============== ALL SUDOKUS SOLVED SUCCESSFULLY! ===============")
        return

    print(f"Valids fraction: {sum(valids) / len(valids):.2f}")

    return saved_states


def get_simulations_function(min_simulations=4,
                             max_simulations=64,
                             difficulty=None):
    def simulations_function(n_zeros):
        if difficulty:
            n_simulations = int(
                max(difficulty[int(n_zeros)] * max_simulations,
                    min_simulations))
        else:
            n_simulations = max_simulations
        return n_simulations

    return simulations_function


def play_sudoku_until_failure(sudoku, solution, network,
                              simulations_function, verbose=0):
    n_zeros = (sudoku == 0).sum()
    N_dict, Q_dict, W_dict, PV_dict = None, None, None, None
    steps_taken = 0
    moves_before_ending = 0
    successful_game = True

    if type(solution) is not torch.Tensor:
        solution = torch.Tensor(solution).reshape((1, 1, 9, 9)).cuda()
    with torch.no_grad():
        while n_zeros > 0:
            N_dict, Q_dict, W_dict, PV_dict = (
                run_simulations(sudoku, network,
                                simulations_function=simulations_function,
                                verbose=0,
                                steps_already_made=steps_taken,
                                N_dict=N_dict, Q_dict=Q_dict,
                                W_dict=W_dict, PV_dict=PV_dict,
                                warm_start=True))

            steps_taken = make_best_move(sudoku, Q_dict, PV_dict,
                                         N_dict)

            valid = torch.all(
                torch.logical_or(torch.eq(sudoku, solution), sudoku == 0))

            if not valid:
                if verbose:
                    print(f"Sudoku failed at moves = {moves_before_ending}!")
                successful_game = False
                break
            else:
                moves_before_ending += 1
                sudoku.cuda()

            n_zeros = (sudoku == 0).sum()
            if verbose:
                print(f"Zeros left: {n_zeros}")
    return moves_before_ending, PV_dict, successful_game
