"""This file is to build structures that help to store the definitions of the
`Layer`s to be used in Big-Little Net Achitecture.

`Layer` object is same as `torch.nn.modules` object.
"""

import torch.nn as nn
import inspect
import utils
import math


class LayerDef:
    r"""Used to store the layer definitions and the attributes of the
    layer corresponding to the biggest branch.

    **Note**: Always pass `Layer` definition `defn` as positional arg
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

    def get_Kbranch_layer(self, **kwargs):
        r"""This function is called by `Block`, when it is trying to
        instantitate the `Layer`s corresponding to the correct branch
        with attributes correspoing to the Big-Little Net.
        """
        # getting the kwargs for `Layer` corresp. to `K`th branch
        ckwargs = self._custom_kwargs(**kwargs)
        ckwargs = utils.modify_args(self.kwargs, ckwargs)
        return self.defn(**ckwargs)

    def __repr__(self):
        r"""representation of stored stuff useful for debugging"""
        layer_doc = inspect.getdoc(self)
        hr = "\n\n" + "-"*30 + "\nFunction doc string:\n"
        func_doc = inspect.getdoc(self.defn)
        return layer_doc + hr + func_doc

    def __call__(self, **kwargs):
        return self.get_Kbranch_layer(**kwargs)

    def _custom_kwargs(self, **kwargs):
        r"""No specialization, ignore this case"""
        return kwargs


class Conv2dDef(LayerDef):
    r"""Object for storing a `nn.Conv2d` function defination.
    
    * :attr:`in_channels` number of input planes
    * :attr:`out_channels` number of output planes
    * :attr:`stride` stride to be chosen for this convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, ds=False):
        super().__init__(nn.Conv2d,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=1,
                         padding=1,
                         bias=False)
        self.ds = ds

    def _custom_kwargs(self, **kwargs):
        r"""Returns custom attributes for `Conv` in `K`th `Branch`."""
        K = kwargs['K']
        alpha = kwargs['alpha']
        RB_it = kwargs['RB_it'] # the iteration in ResBlock reps 
        if K: # Little Branch
            ckwargs = {"in_channels": math.ceil(self.kwargs["in_channels"]/(alpha)),
                       "out_channels": math.ceil(self.kwargs["out_channels"]/(alpha))}
        else: # Big Branch
            # the first 3x3 convolution (depicted with ds=True,
            # and 1st ResBlock) is with stride 2
            if (self.ds) and (RB_it == 0):
                ckwargs = {"stride" = 2}
            if ()
        return ckwargs

# class FC(LayerDef): ### Need to get to the end of the Network
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super().__init__(nn.Linear,
#                          in_channels=in_channels,
#                          out_channels=out_channels,
#                          kernel_size=kernel_size,
#                          stride=1,
#                          padding=1,
#                          bias=False)

#     def _custom_kwargs(self, K):
#         r"""Returns custom attributes for `Conv` in `K`th `Branch`"""
#         ckwargs = {"in_channels": math.ceil(self.kwargs["in_channels"]/(alpha)),
#                    "out_channels": math.ceil(self.kwargs["out_channels"]/(alpha))}
#         return ckwargs
#     pass

# We can use a flag passed (that would be stored in the ResBlock) to
# get_Kbranch_layer for understanding where we would like to upsample
# or something