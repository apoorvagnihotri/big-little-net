"""This file contains helper Classes to abstract away the details
and avoid *difficult to debug* nested lists.
"""

from layerdef import *


class ArchBlock():
    r""" This class is used to store the middle part
    of the architecture of the ResNet.

    `layout` is a list of tuples of (`ResBlock`s, `reps`) defining
    the middle part of the architecture.
    """
    def __init__(self, layout):
        self.layout = layout
        self.size = len(layout)

    def __getitem__(self, key):
        return (layout[key])

    def __len__(self):
        return self.size

    def __repr__(self):
        return "ArchBlock storing " + str([lay for lay in self.layout])


class ResBlock():
    r"""Used to store the Layer definitions to be used in
    the BL Net.

    `layerdefs` is a list of `Layers`s definitions defining
    a `ResBlock`.
    """
    def __init__(self, layerdefs):
        self.layerdefs = layerdefs
        self.size = layerdefs

    def __getitem__(self, key):
        return self.layerdefs[key]

    def __len__(self):
        return self.size

    def __repr__(self):
        return "ResBlock storing " + str([ld.defn for ld in self.layerdefs])

