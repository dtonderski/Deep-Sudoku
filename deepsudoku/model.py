import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.convs = []
        for i in range(8):
            self.convs.append(nn.Conv2d(in_channels=64, out_channels=64,
                                        kernel_size=(3, 3), stride=(1, 1),
                                        padding=1))
        self.convs = nn.ModuleList(self.convs)

        self.convlast = nn.Conv2d(in_channels=64, out_channels=9,
                                  kernel_size=(3, 3), stride=(1, 1),
                                  padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = nn.ReLU()(x)
        for conv in self.convs:
            x = conv(x)
            x = nn.ReLU()(x)
        x = self.convlast(x)
        return x
