'''This file contains different blocks that are used in
the Big-Little Net Architecture.'''

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

    def __init__(self, inplanes, planes, stride, expansion, downsample):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.expansion = expansion
        self.stride = stride

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

        out += identity
        out = self.relu(out)

        return out


class ResBlockB(ResBlock):
    r'''Specialization of the ResBlock for the Big branch.'''

    def __init__(self, inplanes,
                 planes,
                 img_up = 2,
                 stride = 2,
                 expansion = 2,
                 downsample = None):
        # calling the immediate parent's init.
        ResBlock.__init__(self, inplanes = inplanes, planes = planes,
                          stride = stride, expansion = expansion,
                          downsample = downsample)
        # upsample
        self.upsample = nn.Upsample(scale_factor=img_up,
                                    mode='bilinear')

    def forward(self, x):
        out = ResBlock(x)

        # increasing image size
        out = self.upsample(out)

        return out


class ResBlockL(ResBlock):
    r'''Specialization of the ResBlock for the Little branch.

    :attr:`beta` the scalar with wich we need to reduce the number of
    layers in the ResBlock
    '''

    def __init__(self,
                 inplanes,
                 planes,
                 beta = 2,
                 stride = 1,
                 expansion = 2,
                 downsample = None):
        planes_r = int(planes / beta)
        ResBlock.__init__(self, inplanes = inplanes, planes = planes_r,
                          stride = stride, expansion = expansion,
                          downsample = downsample)
        # for increasing the # of layers
        self.conv4 = conv1x1(planes_r * expansion, planes * expansion)
        self.bn4 = nn.BatchNorm2d(planes * expansion)

    def forward(self, x):
        out = ResBlock(x)

        # increasing layers
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out


class TransitionLayer(ResBlock):
    r'''A Specialization of ResBlock with support for merging branches'''

    def __init__(self,
                 inplanes,
                 planes,
                 stride = 2,
                 expansion = 4,
                 downsample = None):
        ResBlock.__init__(self, inplanes = inplanes, planes = planes,
                          expansion = expansion, stride = stride)

    def forward(self, x1, x2): # fixed K = 2
        out = x1 + x2 # merge via add

        out = ResBlock(out)

        return out
