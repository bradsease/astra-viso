"""
Math utilities for Astra Viso.
"""
from __future__ import division
import numpy as np
from functools import reduce
import bisect

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

def lagrange_interpolating_polynomial(x_in, y_in, x_out):
    """
    Interpolate input data with a lagrange interpolating polynomial.

    Parameters
    ----------
    x_in : ndarray
        Input x coordinate data.
    y_in : ndarray
        Input y coordinate data.
    x_out : float
        Desired x coordinate for interpolated value.

    Returns
    -------
    y_out : float
        Interpolated y coordinate.
    """

    interpolant_val = np.zeros(len(x_in))
    for j in range(len(x_in)):
        interpolant_val[j] = y_in[j]
        for k in range(len(x_in)):
            if k != j:
                interpolant_val[j] *= (x_out - x_in[k])/(x_in[j] - x_in[k])

    return np.sum(interpolant_val)

def build_lagrange_interpolator(x_in, y_in):
    """
    Build a lagrange interpolation function.

    Parameters
    ----------
    x_in : ndarray
        Input x coordinate data.
    y_in : ndarray
        Input y coordinate data.

    Returns
    -------
    lagrange_fcn : function
        Lagrange interpolator of the form lagrange_fcn(x_out) where x_out is the
        x value of the desired interpolation point.
    """
    return lambda x_out: lagrange_interpolating_polynomial(x_in, y_in, x_out)

class MovingWindowInterpolator:
    """
    Sliding window interpolation class.
    """

    def __init__(self, x_in, y_in, interpolator, window_size=5):
        """
        Build a sliding window interpolator.

        Parameters
        ----------
        x_in : ndarray
            Input x coordinate data.
        y_in : ndarray
            Input y coordinate data.
        interpolator : function
            Desired interpolation method provided as a function of the form
            interp_fcn = interpolator(x_in, y_in).

        Returns
        -------
        mwi : MovingWindowInterpolator
            Moving window interpolator object.
        """

        self._x_in = x_in
        self._y_in = y_in
        self._window_size = window_size

        self._index_cache = None
        self._interpolator = interpolator
        self._interpolant_cache = None

    def __call__(self, x_out):
        """
        Interpolate the internal data for a particular x value.

        Parameters
        ----------
        x_out : float
            Desired x coordinate for interpolated value.

        Returns
        -------
        y_out : float
            Interpolated y coordinate.
        """

        if self._index_cache is None or abs(self._index_cache-x_out) > 0.5:
            self._build_interpolant(x_out)

        return self._interpolant_cache(x_out)

    def _get_interval(self, x_out):
        """
        Extract coordinates for moving window.

        Parameters
        ----------
        x_out : float
            Desired x coordinate for interpolated value.

        Returns
        -------
        interval : ndarray
            Coordinates of the desired window size (N) corresponding to N
            closest x coordinates to x_out.

        Notes
        -----
        This method also caches the input x_out value to determine if
        construction of a new window is necessary.
        """

        #self._index_cache = bisect.bisect(self._x_in, x_out) - 1
        self._index_cache = x_out

        return sorted(abs(self._x_in-x_out).argsort()[:self._window_size])

    def _build_interpolant(self, x_out):
        """
        Build an interpolating function for a given data window.

        Parameters
        ----------
        x_out : float
            Desired x coordinate for interpolated value.

        Returns
        -------
        interpolant : function
            Interpolating function of the form y_out = interpolant(x_out).
        """
        interval = self._get_interval(x_out)
        self._interpolant_cache = self._interpolator(
            self._x_in[interval], self._y_in[interval])
