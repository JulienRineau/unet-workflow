import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class Block(nn.Module):
    """
    Two convolutional layers with ReLU activation and batch normalization, followed by a max pooling layer.

    Parameters:
    - inputs (int): Number of input channels.
    - middles (int): Number of middle layer channels.
    - outs (int): Number of output channels.
    """

    def __init__(self, inputs: int = 3, middles: int = 64, outs: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, middles, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(middles, outs, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(outs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn(self.conv2(x)))
        return self.pool(x), x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.en1 = Block(3, 64, 64)
        self.en2 = Block(64, 128, 128)
        self.en3 = Block(128, 256, 256)
        self.en4 = Block(256, 512, 512)
        self.en5 = Block(512, 1024, 512)

        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.de4 = Block(1024, 512, 256)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.de3 = Block(512, 256, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.de2 = Block(256, 128, 64)

        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.de1 = Block(128, 64, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, e1 = self.en1(x)
        x, e2 = self.en2(x)
        x, e3 = self.en3(x)
        x, e4 = self.en4(x)
        _, x = self.en5(x)

        x = self.upsample4(x)
        x = torch.cat([x, e4], dim=1)
        _, x = self.de4(x)

        x = self.upsample3(x)
        x = torch.cat([x, e3], dim=1)
        _, x = self.de3(x)

        x = self.upsample2(x)
        x = torch.cat([x, e2], dim=1)
        _, x = self.de2(x)

        x = self.upsample1(x)
        x = torch.cat([x, e1], dim=1)
        _, x = self.de1(x)

        x = self.conv_last(x)
        return x
