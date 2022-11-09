import torch
from deepsudoku.utils import network_utils
import pytest

test_shapes = [(4096, 1, 9, 9),
               (1, 1, 9, 9)]


@pytest.mark.parametrize("shape", test_shapes)
def test_to_categorical_float(shape):
    a = (torch.rand(shape) * 10).long().float()
    b = network_utils.to_categorical(a)
    n, val, i, j = torch.where(b == 1)
    torch.all(a[n, 0, i, j] == val + 1)


@pytest.mark.parametrize("shape", test_shapes)
def test_to_categorical_long(shape):
    a = (torch.rand(shape) * 10).long()
    b = network_utils.to_categorical(a)
    n, val, i, j = torch.where(b == 1)
    torch.all(a[n, 0, i, j] == val + 1)
