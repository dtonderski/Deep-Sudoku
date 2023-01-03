import torch
from torch import nn
from deepsudoku.utils.network_utils import to_categorical
from torchvision.ops import SqueezeExcitation


class ConvBlock(nn.Module):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=9, out_channels=filters,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class SeResBlock(nn.Module):
    def __init__(self, filters, se_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters,
                               kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.se = SqueezeExcitation(filters, se_channels)

    def forward(self, x):
        skip = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += skip
        return torch.relu(x)


class PolicyBlock(nn.Module):
    def __init__(self, filters):
        super(PolicyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=9,
                               kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


class ValueBlock(nn.Module):
    def __init__(self, filters, value_channels, dropout):
        super(ValueBlock, self).__init__()
        self.value_channels = value_channels
        self.conv = nn.Conv2d(in_channels=filters,
                              out_channels=value_channels,
                              kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(value_channels)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(value_channels * 9 * 9, 128)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.dropout1(x)
        x = x.reshape(-1, self.value_channels * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class SeResNet(nn.Module):
    def __init__(self, blocks, filters, se_channels, dropout=0.2,
                 value_channels=32):
        super().__init__()
        self.blocks = blocks
        self.convBlock = ConvBlock(filters)

        for i in range(self.blocks):
            setattr(self, f"block_{i}", SeResBlock(filters, se_channels))
        self.valueBlock = ValueBlock(filters, value_channels=value_channels,
                                     dropout=dropout)
        self.policyBlock = PolicyBlock(filters)

    def forward(self, x: torch.Tensor):
        """

        :param x: (batch_size,9,9,9) tensor one-hot encoded tensor OR
                  (batch_size,1,9,9) tensor where blanks represent 0s
        :return v: (batch_size, 1) tensor representing the predicted value of
                   the sudoku, the unnormalized 'probability' that the sudoku
                   is valid. Note that these are unnormalized logits.
        :return p: (batch_size,9,9,9) tensor representing unnormalized scores,
                   converted to a probability distribution of moves for each
                   cell through a softmax.
        """
        if x.shape[1] == 1:
            # Must one-hot encode!
            x = to_categorical(x)

        x = self.convBlock(x)
        for i in range(self.blocks):
            x = getattr(self, f"block_{i}")(x)
        p = self.policyBlock(x)
        v = self.valueBlock(x)
        return p, v


def main():
    x = torch.zeros((5120, 1, 9, 9))
    print(x.shape)
    model = SeResNet(10, 128, 32)
    x = model(x)
    print(x[0].shape, x[1].shape)


if __name__ == '__main__':
    main()
