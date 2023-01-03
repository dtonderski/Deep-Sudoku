import torch
from torch import nn
from torch.nn import functional


class ResidualBlock(nn.Module):
    pass


class ConvolutionTriplet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 resolution: int):
        super().__init__()

        # Not currently used as below experimentation did not work
        _ = resolution
        # assert output_channels % 4 == 0

        # Kernels should always be odd: decrease by one if image res is even
        # vh_kernel_size = resolution - (1 - resolution % 2)

        # self.conv_normal = dsnn.Conv2d(input_channels, output_channels // 2,
        #                              (3, 3), padding='same')
        # self.conv_horizontal = dsnn.Conv2d(input_channels,
        #                                  output_channels // 4,
        #                                  (1, vh_kernel_size), padding='same')
        # self.conv_vertical = dsnn.Conv2d(input_channels, output_channels // 4,
        #                                (vh_kernel_size, 1), padding='same')
        self.conv = nn.Conv2d(input_channels, output_channels, (3, 3),
                              padding='same')

    def forward(self, x):
        # x1 = self.conv_normal(x)
        # x2 = self.conv_horizontal(x)
        # x3 = self.conv_vertical(x)
        #
        # return torch.concat([x1, x2, x3], dim=1)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, input_resolution, input_channels, output_channels,
                 mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = output_channels
        self.block1 = nn.Sequential(
            ConvolutionTriplet(input_channels, mid_channels, input_resolution),
            nn.BatchNorm2d(mid_channels)
        )

        self.block2 = nn.Sequential(
            ConvolutionTriplet(mid_channels, output_channels,
                               input_resolution),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        x = self.block1(x)
        x = functional.relu(x)
        skip = x
        x = self.block2(x)
        x += skip
        return functional.relu(x)


class ValueBlock(nn.Module):
    def __init__(self, in_channels, n_channels=3, n_hidden=64):
        super().__init__()
        self.n_channels = n_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=n_channels,
                              kernel_size=1)
        self.bn = nn.BatchNorm2d(n_channels)
        self.fc1 = nn.Linear(n_channels * 9 * 9, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.n_channels * 9 * 9)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class PolicyBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=9,
                              kernel_size=3, padding='same')

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, value_channels=16, value_hidden=64):
        super().__init__()
        # 1, 9, 9
        self.conv1 = ConvolutionTriplet(1, 64, 9)
        # 128, 9, 9
        self.down1 = DownBlock(64, 128, 8, 1)
        # 256, 8, 8
        self.down2 = DownBlock(128, 256, 4, 2)
        # 512, 4, 4
        self.down3 = DownBlock(256, 512, 2, 2)
        # 1024, 2, 2
        self.up1 = UpBlock(512, 256, 4, 2)
        # 512, 4, 4
        self.up2 = UpBlock(256, 128, 8, 2)
        # 256, 8, 8
        self.up3 = UpBlock(128, 64, 9, 1)
        # 128, 9, 9
        self.policy = PolicyBlock(64)
        self.value = ValueBlock(64, n_channels=value_channels,
                                n_hidden=value_hidden)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.policy(x), self.value(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2, stride),
            ConvBlock(resolution, in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution, stride):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, 2,
                                            stride)
        self.up = ConvBlock(resolution, in_channels, out_channels)

    def forward(self, x, x_skip):
        """ Pad x so that x and x_skip are the same size """
        x = self.up_sample(x)

        diff_x = x_skip.shape[2] - x.shape[2]
        diff_y = x_skip.shape[3] - x.shape[3]

        x = functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                               diff_y // 2, diff_y - diff_y // 2])

        return self.up(torch.concat([x, x_skip], 1))


if __name__ == '__main__':
    sudoku = torch.zeros(1, 1, 9, 9)
    model = UNet()
    p, v = model(sudoku)
    print(f"{p.shape=}")
    print(f"{v.shape=}")
