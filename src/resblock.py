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
    r'''The main class of Residual Blocks.'''

    def __init__(self, inplanes, planes, expansion, stride):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * expansion)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
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

        out += identity
        out = self.relu(out)

        return out


class ResBlockB(ResBlock):
    r'''Specialization of the ResBlock for the Big branch.'''

    def __init__(self, inplanes, planes, expansion, stride = 2):
        # calling the immediate parent's init.
        ResBlock.__init__(self, inplanes = inplanes, planes = planes,
                          expansion = expansion, stride = stride)
        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') # fixed K = 2

    def forward(self, x):
        out = ResBlock.forward(x)

        # increasing image size
        out = self.upsample(out)

        return out


class ResBlockL(ResBlock):
    r'''Specialization of the ResBlock for the Little branch.'''

    def __init__(self, inplanes, planes, expansion, stride = 1):
        ResBlock.__init__(self, inplanes = inplanes, planes = planes,
                          expansion = expansion, stride = stride)
        self.conv4 = conv1x1(planes * expansion, inplanes)
        self.bn4 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        out = ResBlock.forward(x)

        # increasing layers
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out


class TransitionLayer(ResBlock):
    r'''A Specialization of ResBlock with support for merging branches'''

    def __init__(self, inplanes, planes, expansion, stride):
        ResBlock.__init__(self, inplanes = inplanes, planes = planes,
                          expansion = expansion, stride = stride)

    def forward(self, x1, x2): # fixed K = 2
        out = x1 + x2 # merge via add

        out = ResBlock.forward(out)

        return out