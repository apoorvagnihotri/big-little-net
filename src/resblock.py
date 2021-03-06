'''This file contains different blocks that are used in
the Big-Little Net Architecture.'''
import torch.nn as nn
import torch.nn.functional as F


# Custom Conv2D layers
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):
    r"""A class for basic BasicBlock, only used in the starting of the network.
    This block doesn't have shortcut"""

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.expansion = expansion
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        return out


class ResBlock(nn.Module):
    r'''The main class of Residual Blocks.

    :attr:`inplanes` the number of activations in last layer
    :attr:`planes` the number of activations inside the block layers
    '''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.expansion = expansion
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = self._find_downsampler()

    def _find_downsampler(self):
        '''used to downsample identity for adding to output'''
        downsample = None
        if self.stride != 1 or self.inplanes != self.planes * self.expansion:
            downsample = nn.Sequential(
                    conv1x1(
                      self.inplanes,
                      self.planes * self.expansion,
                      self.stride),
                    nn.BatchNorm2d(self.planes * self.expansion),
                )
        return downsample

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu3(out)

        return out


class ResBlockB(nn.Module):
    r'''ResBlock for the Big branch.'''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']
        self.last = kwargs['last']

        self.rb = ResBlock(inplanes = inplanes, planes = planes,
                      stride = stride, expansion = expansion)

    def forward(self, x):
        out = self.rb(x)

        # increasing image size if last layer
        if self.last:
            out = F.interpolate(out, scale_factor=2, mode='bilinear')

        return out


class ResBlockL(nn.Module):
    r'''ResBlock for the Little branch.

    :attr:`alpha` the scalar with wich we need to reduce the number of
    layers in the convoltions in the Little Branch.
    '''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        alpha = kwargs['alpha']
        stride = kwargs['stride']
        expansion = kwargs['expansion']
        self.last = kwargs['last']

        self.rb = ResBlock(inplanes = inplanes, planes = planes,
                      stride = stride, expansion = expansion)

        # We only define upsampling if block before merge
        if self.last:
            self.conv4 = conv1x1(planes * expansion, planes * expansion * alpha)
            self.bn4 = nn.BatchNorm2d(planes * expansion * alpha)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.rb(x)

        # increasing layers if last layer
        if self.last:
            out = self.conv4(out)
            out = self.bn4(out)
            out = self.relu(out)

        return out


class TransitionLayer(nn.Module):
    r'''Block used to merge two branches'''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        self.rb = ResBlock(inplanes = inplanes, planes = planes,
                      expansion = expansion, stride = stride)

    def forward(self, xs):
        assert(xs[0].shape == xs[1].shape)
        out = xs[0] + xs[1] # merge via add

        out = self.rb(out)

        return out
