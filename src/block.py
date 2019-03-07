"""A `Block` is a collection of `Layer`s corresponding to `K`th `Branch`.
"""
import torch.nn as nn
import inspect

class Block(nn.Module):
    r"""Part of `BLModule`, used to store a multiple `Layer` objects
    given the branch number `K`.

    * :attr:`layer_defs` A list of `LayerDef`s, used to create
    `Layer`s according to the branch number.

    * :attr:`K` is the branch number we are trying to build the
    `Block` of `Layer`s for.
    """
    def __init__(self, layer_defs, K):
        super().__init__()
        self.layer_defs = layer_defs
        self.layers = []
        self.K = K
        # init all `Layer`s from the corresp. `LayerDef`s.
        for layer_def in layer_defs:
            self.layers.append(layer_def.get_Kbranch_layer(K))

    def forward(self, x):
        identity = x
        for layer in self.layers:
            x = layer(x) # allowed as type(layer) == `nn.Module`
        return x

    def __repr__(self):
        r"""for debugging purposes"""
        branchstring = "This is a `Block` of `Layer`s in " + str(self.K) + "th `Branch`"
        layerdef_string = "\n\n`Layer`s:\n["
        for layerdef in self.layer_defs:
            layerdef_string += inspect.getdoc(layerdef.defn) + ", "
        layerdef_string = layerdef_string[:len(layerdef_string)-2] + "]"
        return branchstring + layerdef_string