"""
Projection utilities for astra-viso.
"""
from __future__ import division
import numpy as np

def pinhole_project(vectors, focal_len, pixel_size, resolution):
    """
    Convert body-fixed position vector to image-plane coordinates. Uses the
    pinhole projection model.

    Parameters
    ----------
    vectors : ndarray
        Body vectors to be projected into the image plane. Array should be
        Nx3 where N is the number of vectors.
    focal_len : float
        Focal length in mm.
    pixel_size : float
        Physical pixel size in mm. Pixels are assumed square.
    resolution : int
        Resolution of the sensor.

    Returns
    -------
    img_x : ndarray
        Array of x-coordinates (N elements).
    img_y : ndarray
        Array of y-coordinates (N elements).

    Examples
    --------
    >>> pinhole_project(np.array([0, 0, 1]), 93, 0.016, 1024)
    (array([ 512.5]), array([ 512.5]))
    """

    # Check input
    if resolution <= 0 or not isinstance(resolution, int):
        raise ValueError("Resolution must be integer-valued and positive.")
    if focal_len <= 0:
        raise ValueError("Focal length must be >= 0.")
    if pixel_size <= 0:
        raise ValueError("Physical pixel dimension must be >= 0.")

    # Enforce shape for single vector input
    if len(vectors.shape) == 1:
        vectors = vectors.reshape(1, 3)

    # Intermediate values
    f_over_s = focal_len / pixel_size
    half_res = (resolution + 1)/2

    # Projection equation
    img_x = f_over_s * np.divide(vectors[:, 0], vectors[:, 2]) + half_res
    img_y = f_over_s * np.divide(vectors[:, 1], vectors[:, 2]) + half_res

    # Return result
    return img_x, img_y

def polynomial_project(params):
    """
    """

    raise NotImplementedError("Not yet implemented!")
