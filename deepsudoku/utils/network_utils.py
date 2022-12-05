import torch
import numpy as np
from typing import Tuple, Literal


def to_categorical(tensor):
    tensor = tensor.long()
    x = tensor.ravel()
    n = x.shape[0]
    cat = torch.zeros((n, 10), device=x.device)
    cat[torch.arange(n), x] = 1
    cat = cat.reshape(tensor.shape[0], 9, 9, 10)[:, :, :, 1:]
    return cat.permute(0, 3, 1, 2)


def numpy_batch_to_pytorch(x_train: np.ndarray,
                           y_train: Tuple[np.ndarray, np.ndarray],
                           device: Literal['cuda', 'cpu'] = 'cuda'):
    """

    :param x_train: np uint8 array of size (n,9,9) with unsolved sudokus
                    in [0, 9], with 0 being empty cells
    :param y_train: tuple of ((np uint8 array of size (n,9,9) in [1, 9]
                    with solved sudokus) and (np bool array of size (n, 1) with
                    sudoku validity))
    :param device: 'cuda' or 'cpu'
    :return: x_train: torch float tensor of size (n,1,9,9) on specified device
             y_train: tuple of ((torch long tensor of size (n,9,9) in [0, 8]),
                      (torch float tensor of size (n, 1) specifying validity))

    """
    x_train = torch.tensor(x_train.reshape((-1, 1, 9, 9))).float().to(device)
    y_train = (torch.tensor(y_train[0] - 1).long().to(device),
               torch.tensor(y_train[1]).float().to(device))
    return x_train, y_train
