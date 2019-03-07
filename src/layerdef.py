"""This file is to build structures that help to store the definitions of the
`Layer`s to be used in Big-Little Net Achitecture.

`Layer` object is same as `torch.nn.modules` object.
"""

import torch.nn as nn
import inspect
import resnet
import utils
import math

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

    def get_Kbranch_layer(self, K, **addtional_kwargs):
        r"""Return a `Layer` with defination `defn` modified custom arguments
        corresponding to `K`th branch of Big-Little Net

        * :attr:`K` `Branch` number.

        * :attr:`addtional_kwargs` Additional Kwargs that would overwrite
        the defaults given in Big-Little Net Architecture.
        """
        # getting the kwargs for `Layer` corresp. to `K`th branch
        ckwargs = self._custom_kwargs(K)
        ckwargs = utils.modify_args(self.kwargs, ckwargs)
        # overwrite with user provided additional_kwargs
        final_kwargs = utils.modify_args(ckwargs, addtional_kwargs)
        return self.defn(**final_kwargs)

    def __repr__(self):
        r"""representation of stored stuff useful for debugging"""
        layer_doc = inspect.getdoc(self)
        hr = "\n\n" + "-"*30 + "\nFunction doc string:\n"
        func_doc = inspect.getdoc(self.defn)
        return layer_doc + hr + func_doc

    def __call__(self, K):
        return self.get_Kbranch_layer(K)

    def _custom_kwargs(self, K):
        r"""Pass no custom kwarg, when no specialization"""
        return {}


# Modules below are for ease of use during implementation of Big-Little Net

# @ todo, ConvDef can be made a parent of Conv classes, if required
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

    def _custom_kwargs(self, K):
        r"""Returns custom attributes for `Conv3x3` in `K`th `Branch`"""
        ckwargs = {"in_planes": math.ceil(self.kwargs["in_planes"]/(2**K)),
                   "out_planes": math.ceil(self.kwargs["out_planes"]/(2**K))}
        return ckwargs

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

    def _custom_kwargs(self, K):
        r"""Returns custom attributes for `Conv1x1` in `K`th `Branch`"""
        ckwargs = {"in_planes": math.ceil(self.kwargs["in_planes"]/(2**K)),
                   "out_planes": math.ceil(self.kwargs["out_planes"]/(2**K))}
        return ckwargs

class FC(LayerDef):
    pass

class Transition(LayerDef):
    pass
