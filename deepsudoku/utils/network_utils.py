import torch


def to_categorical(tensor):
    tensor = tensor.long()
    x = tensor.ravel()
    n = x.shape[0]
    cat = torch.zeros((n, 10), device=x.device)
    cat[torch.arange(n), x] = 1
    cat = cat.reshape(tensor.shape[0], 9, 9, 10)[:, :, :, 1:]
    return cat.permute(0, 3, 1, 2)
