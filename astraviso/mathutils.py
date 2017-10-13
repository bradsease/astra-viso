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

def unit(vector):
    """
    Normalize input vector.

    Parameters
    ----------
    vector : ndarray
        Input vector as either a row or column array.

    Returns
    -------
    unit_vector : float
        Input vector normalized to magnitude 1.

    Examples
    --------
    >>> import numpy as np
    >>> vector = np.ones(3)
    >>> unit(vector)
    array([ 0.57735027,  0.57735027,  0.57735027])
    """
    return vector / np.linalg.norm(vector)

def angle(vector1, vector2):
    """
    Compute angle between two vectors.

    Parameters
    ----------
    vector1 : ndarray
        First input vector as either a row or column array.
    vector2 : ndarray
        Second input vector as either a row or column array.

    Returns
    -------
    angle : float
        Angle between vector1 and vector two, in radians.
        
    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([0, 1, 0])
    >>> angle(v1, v1)
    0.0
    >>> angle(v1, v2)
    1.5707963267948966
    """
    return np.arccos(np.clip(np.dot(unit(vector1), unit(vector2)), -1, 1))
