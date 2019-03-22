import torch.nn as nn
from .resblock import *
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['bL_ResNet', 'bl_resnet50', 'bl_resnet101', 'bl_resnet152']


model_urls = {
    # 'bl_resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    # 'bl_resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    # 'bl_resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    # 'bl_resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # 'bl_resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class bL_ResNet(nn.Module):

    def __init__(self,
                 layers,
                 alpha = 2,
                 beta = 4,
                 num_classes=1000,
                 zero_init_residual=False):
        super().__init__()
        # pass 1 | Convolution
        self.inplanesB = 64
        self.inplanesL = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # pass 2 | bL-module
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.littleblock = BasicBlock(
            inplanes=64,
            planes=32, 
            stride=2, 
            expansion=2)

        # pass 3 | `ResBlockB`s & `ResBlockL`s
        arg_d = {
            'planes': 64,
            'beta': beta,
            'alpha': alpha,
            'reps': layers[0],
            'stride': 2,
            'expansion': 4,
            'last': False
        }

        self.big_layer1 = self._make_layer(ResBlockB, arg_d)
        self.little_layer1 = self._make_layer(ResBlockL, arg_d)
        self.transition1 = self._make_layer(TransitionLayer, arg_d)

        arg_d['planes'] = 128; arg_d['reps'] = layers[1];
        self.big_layer2 = self._make_layer(ResBlockB, arg_d)
        self.little_layer2 = self._make_layer(ResBlockL, arg_d)
        self.transition2 = self._make_layer(TransitionLayer, arg_d)

        arg_d['planes'] = 256; arg_d['reps'] = layers[2];
        self.big_layer3 = self._make_layer(ResBlockB, arg_d)
        self.little_layer3 = self._make_layer(ResBlockL, arg_d)
        arg_d['stride'] = 1
        self.transition3 = self._make_layer(TransitionLayer, arg_d)

        arg_d['planes'] = 512; arg_d['reps'] = layers[3];
        arg_d['stride'] = 2
        self.res_layer1 = self._make_layer(ResBlock, arg_d)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * arg_d['expansion'], num_classes)
        # training code takes care of taking the softmax vai logsofmax error

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, Block, arg_d_c):
        '''Instantiates a sequence of `Block`s.

        :attr:`Block` is the Big-Little Net `Block` we have chosen
        :attr:`arg_d_c` are the arguments required to create objects for `Block`s
        '''
        # according to the Block, setting some defaults.
        arg_d = arg_d_c.copy() # making a local copy, avoiding mutability

        if Block == ResBlockB:
            inplanes = self.inplanesB
        elif Block == ResBlockL:
            inplanes = self.inplanesL
            arg_d['planes'] = int(arg_d['planes'] / arg_d['alpha'])
            arg_d['reps'] = math.ceil(arg_d['reps'] / arg_d['beta'])
            arg_d['stride'] = 1 # stride is always 1 for ResBlockL
        elif Block == ResBlock:
            inplanes = self.inplanesB
            assert (inplanes == self.inplanesL) # debugging
        elif Block == TransitionLayer:
            inplanes = self.inplanesB
            assert (inplanes == self.inplanesL) # debugging
            arg_d['reps'] = 1 # reps is always one for TransitionLayer

        expansion = arg_d['expansion']
        stride = arg_d['stride']
        planes = arg_d['planes']
        reps = arg_d['reps']
        alpha = arg_d['alpha']
        beta = arg_d['beta']

        # if last layer, we would do upsampling
        i = 0
        if i == reps - 1:
            arg_d['last'] = True

        layers = []
        layers.append(Block(inplanes = inplanes, **arg_d))
        inplanes = self._new_inplanes(Block, **arg_d)

        # after first Block, stride is set to 1
        if Block in [ResBlock, ResBlockB, ResBlockL]:
            arg_d['stride'] = 1
        
        for i in range(1, reps):
            # if last layer
            if i == reps - 1:
                arg_d['last'] = True
            layers.append(Block(inplanes = inplanes, **arg_d))
            inplanes = self._new_inplanes(Block, **arg_d)

        # updating the current branch's inplanes
        if Block == ResBlockB:
            self.inplanesB = inplanes
        elif Block == ResBlockL:
            self.inplanesL = inplanes
            assert(self.inplanesB == inplanes) # should be equal
        elif Block in [ResBlock, TransitionLayer]:
            self.inplanesB = inplanes
            self.inplanesL = inplanes

        return nn.Sequential(*layers)

    def _new_inplanes(self, Block, **arg_d):
        planes = arg_d['planes']
        expansion = arg_d['expansion']
        alpha = arg_d['alpha']
        last = arg_d['last']
        assert (Block in [ResBlockB, ResBlock, TransitionLayer, ResBlockL])
        new_inplanes = int(planes * expansion)
        if last and Block == ResBlockL:
            new_inplanes = int(planes * expansion * alpha)
        return new_inplanes


    def forward(self, x):
        # Conv
        base1 = self.conv1(x)
        base1 = self.bn1(base1)
        base1 = self.relu1(base1)

        # pass 2 | bL-module
        little1 = base1; big1 = base1;
        big1 = self.conv2(big1)
        big1 = self.bn2(big1)
        big1 = self.relu2(big1)
        little1 = self.littleblock(little1)
        assert (big1.shape == little1.shape)
        base2 = little1 + big1

        # pass 3 | `ResBlockB`s & `ResBlockL`s  planes = 64
        little2 = base2; big2 = base2;
        big2 = self.big_layer1(big2)
        little2 = self.little_layer1(little2)
        # print ('1st layer passed')
        base3 = self.transition1([big2, little2])

        # pass 4 | planes = 128
        little3 = base3; big3 = base3;
        big3 = self.big_layer2(big3)
        little3 = self.little_layer2(little3)
        # print ('2nd layer passed')
        base4 = self.transition2([big3, little3])

        # pass 5 | planes = 256
        little4 = base4; big4 = base4;
        big4 = self.big_layer3(big4)
        little4 = self.little_layer3(little4)
        # print ('3rd layer passed')
        out = self.transition3([big4, little4])

        # pass 6 | Res_Block | planes = 512
        out = self.res_layer1(out)

        # avg pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def bl_resnet50(pretrained=False, **kwargs):
    """Constructs a bL-ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = bL_ResNet([2, 3, 5, 3], **kwargs)
    # print ('model created')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def bl_resnet101(pretrained=False, **kwargs):
    """Constructs a bL-ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = bL_ResNet([3, 7, 17, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def bl_resnet152(pretrained=False, **kwargs):
    """Constructs a bL-ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = bL_ResNet([4, 11, 29, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model