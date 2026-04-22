from torchinfo import summary
from torch import nn
from abc import ABC
from load_utils import NUM_CLASSES

class CustomSpectrogramModel(nn.Module, ABC):
    """
        Abstract class designed to ease the implementation
        of other models. It is designed to be used as a
        template only for models taking as input a 1x128x501 spectrogram.
    """

    def __init__(self):
        super(CustomSpectrogramModel, self).__init__()
        self._net = None

    def forward(self, x):
        return self._net(x)

    def __str__(self):
        return str(summary(self._net, input_size=(1, 1, 128, 501), verbose=0))

class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, in_channels, channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        if strides != 1 or in_channels != channels:
            self.conv3 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=strides)
            self.bn3 = nn.BatchNorm2d(channels)
        else:
            self.conv3 = None
            self.bn3 = None

    def forward(self, x):
        y = nn.ReLU()(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.bn3(self.conv3(x))
        y += x
        return nn.ReLU()(y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk

class ResNet18(CustomSpectrogramModel):
    def __init__(self):
        super(ResNet18, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # 64x64x251
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 64x64x126
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # 64x32x126
        b3 = nn.Sequential(*resnet_block(64, 128, 2)) # 128x16x63
        b4 = nn.Sequential(*resnet_block(128, 256, 2)) # 256x8x32
        b5 = nn.Sequential(*resnet_block(256, 512, 2)) # 512x4x16

        self._net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, NUM_CLASSES))

if __name__ == '__main__':
    net = ResNet18()
    print(net)