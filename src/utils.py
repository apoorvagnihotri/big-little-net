""" This file contain utilities"""

def modify_args(o_kwargs, c_kwargs):
    r"""Returns `c_kwargs` updated with arguments
    from `o_kwargs` that were not present in `c_kwargs`.

    **Note** returns overwritten c_kwargs.

    * :attr:`o_kwargs` the original keyword arguments.

    * :attr:`c_kwargs` the current keyword arguments.
    """

    for k in [*o_kwargs]:
        if k not in [*c_kwargs]:
            # add original keyword argument that's missing
            c_kwargs[k] = o_kwargs[k]
    return c_kwargs
