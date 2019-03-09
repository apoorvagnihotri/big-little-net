"""A `Block` is a collection of `Layer`s corresponding to `K`th `Branch`.
"""
import torch.nn as nn
import inspect
import math

class Block(nn.Module):
    r"""Part of `BLModule`, used to store multiple `Layer` objects
    given the branch number `K`.

    * :attr:`residual_block` An object that contains a list of
    `LayerDef`s, for a single `ResBlock` in the Architecture

    * :attr:`reps` The number of time `residual_block` pass is to
    be repeated in the big branch. 

    * :attr:`K` Denotes the Little Branch if `1`.

    * :attr:`alpha` The scaler which is used to reduce the number of
    kernels in the Little-Branch.

    * :attr:`beta` the complexity controller in Big-Little Nets
    that controls the number of `ResBlock`s in `K`th branch.
    """
    def __init__(self, residual_block, reps, K, alpha, beta):
        super().__init__()
        self.residual_block = residual_block
        self.layers = []
        self.K = K
        if K: # Little Branch
            self.reps = max(math.ceil(reps / beta) - 1, 1)
        else: # Big Branch
            self.reps = reps
        # repeat for the number of reps of ResBlock for `K`th branch.
        for RB_it in range(self.reps):
            for layerdef in residual_block:
                self.layers.append(layerdef.get_Kbranch_layer(K=K,
                                                              alpha=alpha,
                                                              RB_it=RB_it))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        r"""for debugging purposes"""
        branchstring = "This is a `Block` of `Layer`s in " + str(self.K) + "th `Branch`"
        layerdef_string = "\n\n`Layer`s:\n\n"
        for layer in self.layers:
            layerdef_string += str(layer) + "\n"
        return branchstring + layerdef_string

# class UpSample(nn.Module):
    

# class Transition(nn.Module):
#     r"""Used to combine K branches"""
#     def __init__(self, method='add'):
#         super().__init__()

#     def forward(self, xs):
#         if method == 'add':
#             pass
