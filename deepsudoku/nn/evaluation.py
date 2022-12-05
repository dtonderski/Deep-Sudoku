import torch


def categorical_accuracy(x: torch.Tensor, y: torch.Tensor,
                         y_pred: torch.Tensor) -> torch.Tensor:
    mask = y[1][:, :, None, None] * (x == 0)
    return torch.logical_and(
        torch.eq(y_pred[0].argmax(dim=1), y[0]), mask[:, 0]
    ).sum() / mask.sum()


def binary_accuracy(y: torch.Tensor, y_pred: torch.Tensor):
    return ((y_pred[1] > 0) == y[1]).sum() / len(y[1])
