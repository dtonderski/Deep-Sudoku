from typing import Tuple

import torch


def getFractionOfEmptyCellsFilledCorrectly(
    output: Tuple[torch.Tensor, torch.Tensor],
    target: Tuple[torch.Tensor, torch.Tensor],
    input: torch.Tensor,
):
    """
    Function taking in network input, output, and target, and returning a
    tensor containing the fractions of empty cells filled correctly for each
    sudoku in the batch.

    :param output: tuple (p,v), where:
        p: (batch_size,9,9,9) tensor representing unnormalized scores,
            converted to a probability distribution of moves for each
            cell through a softmax.
        v: (batch_size, 1) tensor representing the predicted value of
            the sudoku, the unnormalized 'probability' that the sudoku
            is valid. Note that these are unnormalized logits.
    :param target: tuple (p,v), where:
        p: (batch_size, 9, 9) tensor containing the completed sudoku, with
            p[i,j,k] in {0;8}, NOT {1,9} because this is used for
            classification.
        v: (batch_size, 1) tensor, where v[i] = [1] if sudoku is valid and [0]
           otherwise.
    :param input: (batch_size, 1, 9, 9), where input[i,j,k,l] in {0;9}. In
        other words, this is the unedited sudokus, with empty cells being
        represented by zeros.
    :return fractions: (batch_size) tensor containing the fractions of empty
        cells filled correctly for each sudoku.
    """

    emptyCellsFilledCorrectly = torch.sum(
        torch.logical_and(
            torch.max(output[0], 1)[1] == target[0], input[:, 0] == 0
        ),
        (1, 2),
    )
    emptyCells = torch.sum(input == 0, (1, 2, 3))
    return emptyCellsFilledCorrectly / emptyCells
