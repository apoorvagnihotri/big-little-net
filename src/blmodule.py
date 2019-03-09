"""This file holds the `BLModule`, a couple of
them together make the Big-Little Net.
"""
from block import Block

class BLModule(nn.Module):
    r"""Holds the `K` `Block`s corresponding to `K` branches
    After computing all the `Branch`es, all `Branch`es merge

    * :attr:`resdiual_blocks` A list of `ResBlock`s, used to create
    `Layer`s according to the branch number.

    * :attr:`num_branch` is the branches.

    **Note**: We can only accomodate 2 Branches, as the paper doesn't
    follow any specific formulas for higher order branches.
    """
    def __init__(self, resdiual_block, reps, num_branch=2):
        r"""Takes the resdiual_block and the number of branches
        to make in the Net.
        """
        super().__init__()
        self.blocks = []
        for i in range(num_branch):
            self.blocks.append(Block(resdiual_block, i))

        self.upsample = UpSample() ### Work on how to pass the size info to upsampler or It can be done mannualy by the network creator...
        # Can't be done as we have to repeat is multiple times.
        self.merge = Transition() #################

    def forward(self, x):
        xs = [] # for storing the results of parallel branches

        #
        for i, block in enumerate(self.blocks):
            temp = block(x)
            xs.append(upsample(temp, i)) # upsampling ith branch

        out = self.merge(*xs)
        return out
