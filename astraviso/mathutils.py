"""
Math utilities for Astra Viso.
"""
from __future__ import division
import numpy as np
from functools import reduce

def dot_sequence(*args):
    """
    Perform a sequential dot product between any number of inputs.

    Parameters
    ----------
    *args :
        Any number of inputs. Inputs must be compatible in shape with a
        sequential dot product, i.e. np.dot(arg1, arg2, arg3, ...)

    Returns
    -------
    product :
        Output result. Will be either a scalar or array depending on the input
        types and dimensions.
    """
    return reduce(np.dot, args)
