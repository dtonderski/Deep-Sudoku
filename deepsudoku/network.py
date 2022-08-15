import torch
from torch import nn
from torch.nn import functional
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, s):
        return torch.relu(self.bn(self.conv(s)))


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        skip = x
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x += skip
        return torch.relu(x)


class ValueBlock(nn.Module):
    def __init__(self):
        super(ValueBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=3,
                              kernel_size=1)
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 9 * 9, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = x.view(-1, 3 * 9 * 9)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PolicyBlock(nn.Module):
    def __init__(self):
        super(PolicyBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=9,
                              kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, x):
        return self.conv(x)


class Network(nn.Module):
    def __init__(self, n_res_blocks=2):
        super(Network, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.convBlock = ConvBlock()
        self.resBlockList = []
        for i in range(self.n_res_blocks):
            setattr(self, f"res_{i}", ResBlock())
        self.valueBlock = ValueBlock()
        self.policyBlock = PolicyBlock()

    def forward(self, x: torch.Tensor):
        """
        
        :param x: (batch_size,1,9,9) tensor representing the sudokus, where 0s
                  represent empty cells
        :return v: (batch_size, 1) tensor representing the predicted value of
                   the sudoku, the unnormalized 'probability' that the sudoku
                   is valid. Note that these are unnormalized logits.
        :return p: (batch_size,9,9,9) tensor representing unnormalized scores,
                   converted to a probability distribution of moves for each
                   cell through a softmax.
        """
        x = self.convBlock(x)
        for i in range(self.n_res_blocks):
            x = getattr(self, f"res_{i}")(x)
        p = self.policyBlock(x)
        v = self.valueBlock(x)
        return p, v


def my_loss(output: Tuple[torch.Tensor, torch.Tensor],
            target: Tuple[torch.Tensor, torch.Tensor],
            weight=1):
    """

    :param output: see output of Network.forward
    :param target: tuple (p,v), where:
        p: (batch_size, 9, 9) tensor containing the completed sudoku
        v: (batch_size, 1) tensor, where v[i] = [1] if sudoku is valid and [0]
           otherwise
    :param weight: v_loss is scaled by weight
    :return: numerical value of the loss
    """
    p_loss = functional.cross_entropy(output[0], target[0])
    v_loss = weight*functional.binary_cross_entropy_with_logits(output[1],
                                                                target[1])
    return p_loss + v_loss
