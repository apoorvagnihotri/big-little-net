from layerdef import *
from blmodule import BLModule
from resnet import 


def conv(in_channels=3, out_channels=64, kernel_size=3, stride=1):
    """Base Convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=1,
                     bias=False)


class BitLittleNet(nn.Module):
    '''The collection of `K` branches
    `forward()` is supposed to return the combination `K` branches.
    And we aim to join multiple `BigLittleModule`s to form
    the whole network
    '''

    def __init__(self, arch):
        super().__init__()
        self.conv_base = conv(3, 64, 7, 2)
        for layout in arch:
            self.modules = BLModule(layout)

    # def forward(self, x):
    #     identity = x

    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     # normalization is happening before the activations
    #     out = self.relu(out)

    #     out = self.conv2(out)
    #     out = self.bn2(out)

    #     if self.downsample is not None:
    #         identity = self.downsample(x)

    #     out += identity
    #     out = self.relu(out)

    #     return out
