import torch
import torch.nn.functional as functional
from typing import Tuple


def get_binary_cross_entropy_weights(v_target, eps=1e-5):
    n_valid = v_target.sum()
    n_invalid = v_target.shape[0] - n_valid
    weights = v_target * (n_valid + n_invalid) / (2 * n_valid + eps) + (
        1 - v_target
    ) * (n_valid + n_invalid) / (2 * n_invalid + eps)
    return weights


def loss(
    input_tensor: torch.Tensor,
    output_tuple: Tuple[torch.Tensor, torch.Tensor],
    target_tuple: Tuple[torch.Tensor, torch.Tensor],
    mask: torch.Tensor = None,
    weight=1,
    binary_cross_entropy_weights=None,
    eps=1e-5,
):
    """

    :param input_tensor: the input to the network. Using to mask loss for
                         non-blanks
    :param output_tuple: see output of Network.forward. Same shape as
                          target_tuple.
    :param target_tuple: tuple (p,v), where:
                          p: (batch_size, 9, 9, 9) tensor containing the
                             estimated probability distribution over sudoku
                             moves
                          v: (batch_size, 1) tensor, where v[i] = [1] if sudoku
                             i is valid and [0]
                             otherwise
    :param mask: used to mask v_loss. If None, no v_losses are masked.
    :param weight: v_loss is scaled by weight.
    :param binary_cross_entropy_weights:
    :param eps: add this to mask to handle sudokus with no empty cells -
                p will be 0, but we still want to learn v.
    :return: sum of masked p crossentropy and masked, weighted v
             binary_crossentropy.
    """

    p_entropy = functional.cross_entropy(
        output_tuple[0], target_tuple[0], reduction="none"
    )
    p_zero_mask = input_tensor[:, 0] == 0
    p_invalid_mask = (target_tuple[1] != 0).unsqueeze(-1)
    p_mask = p_zero_mask * p_invalid_mask

    p_entropy_masked = p_mask * p_entropy

    p_loss_sudoku_wise = p_entropy_masked.sum(dim=(-1, -2)) / (
        p_mask.sum(dim=(-1, -2)) + eps
    )

    p_loss = p_loss_sudoku_wise.sum() / p_invalid_mask.sum()

    v_entropy = weight * functional.binary_cross_entropy_with_logits(
        output_tuple[1],
        target_tuple[1],
        binary_cross_entropy_weights,
        reduction="none",
    )
    if mask is None:
        mask = torch.ones(v_entropy.shape).to(v_entropy.device)

    v_entropy_masked = v_entropy * mask
    v_loss = v_entropy_masked.sum() / (mask.sum() + eps)
    return p_loss, v_loss
