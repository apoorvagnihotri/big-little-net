'''This file contains different blocks that are used in
the Big-Little Net Architecture.'''
import torch.nn as nn


# Custom Conv2D layers
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class ResBlock(nn.Module):
    r'''The main class of Residual Blocks.

    :attr:`inplanes` the number of activations in last layer
    :attr:`planes` the number of activations inside the block layers
    :attr:`downsample` a `nn.Sequential` object, used to match the size
    and layers of input image to perform point-wise addition to the
    output of the block.
    '''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
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
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        assert(identity.shape == out.shape)
        out += identity
        out = self.relu(out)

        return out


class ResBlockB(nn.Module):
    r'''ResBlock for the Big branch.'''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        # calling the immediate parent's init.
        self.rb = ResBlock(inplanes = inplanes, planes = planes,
                      stride = stride, expansion = expansion)
        # upsample
        self.upsample = nn.Upsample(scale_factor = 2, # fixed for K = 2
                                    mode='bilinear')

    def forward(self, x):
        out = self.rb(x)

        # increasing image size
        out = self.upsample(out)

        return out


class ResBlockL(nn.Module):
    r'''ResBlock for the Little branch.

    :attr:`alpha` the scalar with wich we need to reduce the number of
    layers in the ResBlock
    '''

    def __init__(self, **kwargs):
        super().__init__()
        inplanes = kwargs['inplanes']
        planes = kwargs['planes']
        alpha = kwargs['alpha']
        stride = kwargs['stride']
        expansion = kwargs['expansion']

        self.rb = ResBlock(inplanes = inplanes, planes = planes,
                      stride = stride, expansion = expansion)
        # for increasing the # of layers
        self.conv4 = conv1x1(planes * expansion, planes * expansion * alpha)
        self.bn4 = nn.BatchNorm2d(planes * expansion * alpha)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.rb(x)

        # increasing layers
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out


class TransitionLayer(nn.Module):
    r'''A Specialization of ResBlock with support for merging branches'''

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
