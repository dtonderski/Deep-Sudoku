import time
from collections import defaultdict
from typing import List, Tuple, Iterable
import torch
import numpy as np


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


def run_simulations(sudoku, network, steps, N_dict=None, Q_dict=None,
                    W_dict=None, PV_dict=None, c=1, cutoff=0, verbose=1,
                    warm_start=False):
    if ((N_dict is None or Q_dict is None or W_dict is None or PV_dict is None)
            or not warm_start):
        if verbose >= 1:
            print("Resetting dictionaries")
        N_dict = TensorDict(lambda: torch.zeros((9, 9, 9)).cuda())
        Q_dict = TensorDict(lambda: torch.zeros((9, 9, 9)).cuda())
        W_dict = TensorDict(lambda: torch.zeros((9, 9, 9)).cuda())
        PV_dict = TensorDict()

    times = defaultdict(list)
    for i in range(steps):
        if verbose >= 1:
            print(f"Iteration {i + 1}/{steps}")
        j = -1

        # Empty traversed edges, clone original sudoku tensor
        edges: List[Tuple[torch.tensor, Iterable]] = []
        temp_sudoku = sudoku.clone()

        # Run one iteration of the simulation
        while True:

            j += 1
            if verbose >= 2 and j > 1:
                print(f"j = {j}")

            # If node has been seen before, load p and v values from network.
            # Else, run network on sudoku
            if tensor_to_dict_key(temp_sudoku) not in PV_dict.keys():

                start = time.time()

                p_raw, v_raw = network(temp_sudoku)
                times["network"].append(time.time() - start)

                start = time.time()
                v = torch.sigmoid(v_raw)
                p = torch.softmax(p_raw, 1)[0]
                times["activation"].append(time.time() - start)

                start = time.time()

                PV_dict[temp_sudoku] = (p, v)
                leaf = True

                times["PV_dict_set"].append(time.time() - start)
                if verbose >= 3:
                    print("Unseen node reached!")
            else:

                start = time.time()

                p, v = PV_dict[temp_sudoku]
                leaf = False

                times["PV_dict"].append(time.time() - start)
                if verbose >= 3:
                    print("Previously seen node reached!")

            # End of game reached. This is needed because the end of the game
            # might have been reached before, in which case leaf would be False
            # above
            if (temp_sudoku == 0).sum() == 0:
                if verbose >= 3:
                    print("Sudoku completed!")
                leaf = True

            # If we have reached a leaf, update each (state,action) pair for
            # every traversed edge and end simulation. Otherwise, continue.
            if leaf:
                if verbose >= 3:
                    print("Leaf reached!")
                start = time.time()
                for state, action in edges:
                    N_dict[state][action] += 1
                    W_dict[state][action] += v[0][0]
                    Q_dict[state][action] = (W_dict[state][action] /
                                             N_dict[state][action])

                times["Dicts_update"].append(time.time() - start)

                break
            else:
                if verbose >= 3:
                    print("Non-leaf reached!")
                start = time.time()
                Q = Q_dict[temp_sudoku]
                N = N_dict[temp_sudoku]
                times["Dicts_access"].append(time.time() - start)

                start = time.time()
                # Cut off small values as otherwise we do a lot of useless
                # exploration. Remember that we only care about finding ANY
                # valid move, not the "best" valid move, whatever that could
                # mean.
                p_cutoff = p * (p > cutoff)
                # 1e-9 is so that we choose a reasonable move in the first
                # iteration where N.sum() is 0. Kind of ugly.
                U = c * p_cutoff * (torch.sqrt(N.sum() + 1e-9) / (1 + N))
                productivity = Q + U

                # We are only interested in nodes where temp_sudoku is 0
                productivity = productivity * (temp_sudoku[0] == 0)

                # When productivity is negative for all searched nodes, argmax
                # would choose an already full node, as it has productivity
                # 0 after the above line. Productivity can never go below -1.
                # Therefore, need below to fix:
                productivity[productivity == 0] = -1

                # # Just a sanity check to convince me
                # for i in range(9):
                #     if temp_sudoku[0,0,0,i] != 0:
                #         assert sum(productivity[:,0,i] == 0)

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
                times["rest"].append(time.time() - start)

                # t2 = time.time()
                if verbose >= 3:
                    print(f"Maximum productivity: {productivity[maximum]}")
                    print(f"Sudoku updated with {maximum}")

                # Figured out below
                # if productivity[maximum] < 1e-12:
                #     # For debug purposes, should never happen, but it does.
                #     # Trying to figure out why.
                #     print(temp_sudoku)
                #     print(productivity)
    return N_dict, Q_dict, W_dict, PV_dict, times


def get_best_move(sudoku, N_dict, solution=None, verbose=False):
    eval_max = np.unravel_index(np.argmax(N_dict[sudoku].cpu()), (9, 9, 9))
    if verbose:
        print(eval_max)
        if solution is not None:
            print(solution[eval_max[1], eval_max[2]])
            print(solution[eval_max[1], eval_max[2]] == eval_max[0])
    return eval_max


def make_best_move(sudoku, N_dict, solution=None, verbose=False):
    eval_max = get_best_move(sudoku, N_dict, solution, verbose)
    sudoku[0, 0, eval_max[1], eval_max[2]] = eval_max[0] + 1
