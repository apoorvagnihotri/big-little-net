"""This file is to build structures that help to store the definitions of the
`Layer`s to be used in Big-Little Net Achitecture.

`Layer` object is same as `torch.nn.modules` object.
"""

import torch.nn as nn
import resnet
import utils

class LayerDef:
    r"""Used to store the layer definitions and the attributes of the
    layer corresponding to the biggest branch.

    **Note**: Always pass `Layer` definition as positional arg
    and the arguments to that `Layer` as keyword arguments

    * :attr:`defn` the function definition for the layer
    to be called when compiling.

    * :attr:`kwargs` are the relevant arguments to the function
    definition passed in `defn`. When the object is called,
    these arguments would be passed to the `defn`.
    """
    def __init__(self, defn, **kwargs):
        self.defn = defn
        self.kwargs = kwargs

    def get_layer_obj(self, **ckwargs):
        r"""Return a `Layer` with defination `defn` modified custom arguments"""
        ckwargs = utils.modify_args(self.kwargs, ckwargs)
        return self.defn(**ckwargs)

    def __call__(self, **ckwargs):
        return self.get_layer_obj(**ckwargs)

    def __repr__(self):
        r"""representation of stored stuff useful for debugging"""
        layer_doc = inspect.getdoc(self)
        hr = "\n\n" + "-"*30 + "\nFunction doc string:\n"
        func_doc = inspect.getdoc(self.defn)
        return layer_doc + hr + func_doc


# Modules below are for ease of use during implementation of Big-Little Net
class Conv3x3Def(LayerDef):
    r"""Object for storing a 3x3 `nn.Conv2d` function defination.
    Uses the convolution definitions from `renset.py`


    * :attr:`in_planes` number of input planes

    * :attr:`out_planes` number of output planes

    * :attr:`stride` stride to be chosen for this convolution layer
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__(resnet.conv3x3,
                         in_planes = in_planes,
                         out_planes = out_planes,
                         stride = stride)

class Conv1x1Def(LayerDef):
    r"""Object for storing a 1x1 `nn.Conv2d` function defination.
    Uses the convolution definitions from `renset.py`


    * :attr:`in_planes` number of input planes

    * :attr:`out_planes` number of output planes

    * :attr:`stride` stride to be chosen for this convolution layer
    """
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__(resnet.conv1x1,
                         in_planes = in_planes,
                         out_planes = out_planes,
                         stride = stride)

class FC(LayerDef):
    pass

class Transition(LayerDef):
    pass
