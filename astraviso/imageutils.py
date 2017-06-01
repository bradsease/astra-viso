"""
Image utilities for astra-viso.
"""
from __future__ import division
import random
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

def poisson_noise(image, delta_t, dark_current, read_noise):
    """
    Add poisson-distributed noise to an image.

    Parameters
    ----------
    image : ndarray
        Input image in the form of an MxN numpy array.
    delta_t : float
        Sensor integration time in seconds.
    dark_current : float
        Sensor dark current noise level. Measured in photoelectrons per second.
    read_noise : float
        Sensor read noise. Measured in photoelectrons.

    Returns
    -------
    image : ndarray
        Input image with added noise.

    Examples
    --------
    >>> image = np.zeros(512)
    >>> noisy_image = poisson_noise(image, 0.1, 1200, 200)
    """

    # Check for invalid input
    if np.any(np.array([delta_t, dark_current, read_noise]) < 0):
        raise ValueError("Input values for delta_t, dark_current, and read_noise must be positive.")

    # Add shot noise
    image = np.random.poisson(image)

    # Add dark current
    image += np.random.poisson(dark_current*delta_t, image.shape)

    # Add read noise
    image += np.random.poisson(read_noise, image.shape)

    # Return noisy image
    return image

def gaussian_noise(image, delta_t, dark_current, read_noise):
    """
    Add gaussian-distributed noise to an image. Approximates poisson-
    distributed noise.

    Parameters
    ----------
    image : ndarray
        Input image in the form of an MxN numpy array.
    delta_t : float
        Sensor integration time in seconds.
    dark_current : float
        Sensor dark current noise level. Measured in photoelectrons per second.
    read_noise : float
        Sensor read noise. Measured in photoelectrons.

    Returns
    -------
    image : ndarray
        Input image with added noise.

    Examples
    --------
    >>> image = np.zeros(512)
    >>> noisy_image = gaussian_noise(image, 0.1, 1200, 200)
    """

    # Check for invalid input
    if np.any(np.array([delta_t, dark_current, read_noise]) < 0):
        raise ValueError("Input values for delta_t, dark_current, and read_noise must be positive.")

    # Add shot noise
    image += np.round(np.sqrt(image) * np.random.randn(*image.shape))

    # Add dark current
    image += np.round(dark_current*delta_t + np.sqrt(dark_current*delta_t) *                       \
                                                                      np.random.randn(*image.shape))

    # Add read noise
    image += np.round(read_noise + np.sqrt(read_noise) * np.random.randn(*image.shape))

    return image

def vismag2photon(vismags, delta_t, aperture, mv0_flux):
    """
    Convert visible magnitude to photoelectron count.

    Parameters
    ----------
    vismags : int, float, or ndarray
        Object visible magnitude.
    delta_t : float
        Sensor exposure time in seconds.
    aperture : float
        Aperture area in mm^2.
    mv0_flux : float
        Photoelectrons per second per mm^2 of aperture area.

    Returns
    -------
    photons : float or ndarray
        Total photon count for each input visible magnitude.

    Notes
    -----
    Based on: Liebe, Carl Christian. "Accuracy performance of star trackers-a
              tutorial." IEEE Transactions on aerospace and electronic systems
              38.2 (2002): 587-599.

    Examples
    --------
    >>> vismag2photon(0, 1, 1, 19000)
    19000.0
    >>> vismags = np.array([1, 0, -1])
    >>> vismag2photon(vismags,1,1,19000)
    array([  7600.,  19000.,  47500.])
    """

    # Check for incorrect input
    if np.any(np.array([delta_t, aperture, mv0_flux]) < 0):
        raise ValueError("Input value for delta_t, aperture, and mv0_flux cannot be negative.")

    # Return total photon count
    return mv0_flux * (1 / (2.5**np.asarray(vismags))) * delta_t * aperture

def apply_constant_quantum_efficiency(photon_image, quantum_efficiency):
    """
    Apply a constant quantum efficiency to an input image.

    Parameters
    ----------
    photon_image : ndarray
        Input image where each pixel contains a photon count.
    quantum_efficiency : float
        Relationship between photons and photoelectrons. Measured as the number
        of photoelectrons per photon.

    Returns
    -------
    photoelectron_image : ndarray
        Scaled, discrete-valued image where each pixel contains a photo-
        electron count.

    Examples
    --------
    >>> apply_constant_quantum_efficiency(5*np.ones((4,4)), 0.2)
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])
    """

    # Validate input
    if quantum_efficiency < 0:
        raise ValueError("Quantum efficiency parameter must be positive.")

    # Scale image & return result
    return np.floor(photon_image * quantum_efficiency)

def apply_polynomial_quantum_efficiency(photon_image, poly_coefficients):
    """
    Apply a polynomial-valued quantum efficiency array to an input image.

    Parameters
    ----------
    photon_image : ndarray
        Input image where each pixel contains a photon count.
    poly_coefficients : ndarray
        Polynomial coefficient array. Elements designated such that 
        poly_coefficients[i,j] * x^i * y^j. The origin of the (x,y) pixel
        coordinate system is the geometric center of the image. Coefficient
        array must be 2 dimensional.

    Returns
    -------
    photoelectron_image : ndarray
        Scaled, discrete-valued image where each pixel contains a photo-
        electron count.

    Notes
    -----
    See documentation for numpy.polynomial.polynomial.polyval2d for more
    information on constructing the polynomial coefficient matrix.

    Examples
    --------
    >>> poly_coefficients = np.array([[1,0,1], [0,0,0], [1,0,0]])
    >>> apply_polynomial_quantum_efficiency(np.ones((4,4)), poly_coefficients)
    array([[ 5.  3.  3.  5.]
           [ 3.  1.  1.  3.]
           [ 3.  1.  1.  3.]
           [ 5.  3.  3.  5.]])
    """

    # Check input shape
    if len(photon_image.shape) == 1:
        raise ValueError("Input image must be 2-dimensional.")

    # Generate quantum efficiency map
    x, y = np.meshgrid(range(photon_image.shape[0]), range(photon_image.shape[1]))
    quantum_efficiency = np.polynomial.polynomial.polyval2d(x-np.max(x)/2, y-np.max(y)/2,          \
                                                                                  poly_coefficients)

    # Scale image & return result
    return np.floor(photon_image * quantum_efficiency)

def apply_gaussian_quantum_efficiency(photon_image, mean_quantum_efficiency, sigma, seed=None):
    """
    Apply a spatially gaussian random quantum efficiency to an input image.

    Parameters
    ----------
    photon_image : ndarray
        Input image where each pixel contains a photon count.
    mean_quantum_efficiency : float
        Relationship between photons and photoelectrons (mean value). Measured
        as the number of photoelectrons per photon.
    sigma : float
        Desired standard deviation of the resulting random values.
    seed : float, optional
        Random number generator seed.

    Returns
    -------
    photoelectron_image : ndarray
        Scaled, discrete-valued image where each pixel contains a photo-
        electron count.

    Notes
    -----
    Warning: this function may result in negative quantum efficiencies. In that
    event, the negative entry is replaced with its absolute value. Thus the
    resulting distribution is only approximately gaussian. For large values of
    mean_quantum_efficiency/sigma this effect will not be significant.

    Examples
    --------
    >>> apply_gaussian_quantum_efficiency(100*np.ones((4,4)), 0.2, 0.01, seed=1)
    array([[ 22.,  22.,  20.,  18.],
           [ 17.,  20.,  17.,  17.],
           [ 20.,  20.,  21.,  18.],
           [ 20.,  19.,  16.,  21.]])
    """

    # Validate input
    if mean_quantum_efficiency < 0 or sigma < 0:
        raise ValueError("Mean quantum efficiency and sigma parameter must be positive.")

    # Set up RNG
    rng = random.Random()
    rng.seed(a=seed)
    #if seed is not None:
    #    rng.seed(a=seed)

    # Generate efficiencies
    rows, cols = photon_image.shape
    quantum_efficiency = np.array([[rng.gauss(mean_quantum_efficiency, sigma) for col in           \
                                                               range(cols)] for row in range(rows)])

    # Scale image & return result
    return np.floor(np.abs(photon_image * quantum_efficiency))

def saturate(image, bit_depth):
    """
    Apply a saturation threshold to an input image.

    Parameters
    ----------
    image : ndarray
        Input image where each pixel represents a photoelectron count.
    bit_depth : int
        Number of bits to store each pixel. Maximum value of a particular pixel
        is 2**bit_depth - 1.

    Returns
    -------
    saturated_image : ndarray
        Image with no values above the saturation threshold.

    Examples
    --------
    >>> saturate(40*np.ones((2,2)), 4)
    """

    # Check for incorrect input
    if not isinstance(bit_depth, int):
        raise ValueError("Bit depth must be an integer value.")
    if bit_depth < 0:
        raise ValueError("Bit depth must be positive.")

    # Saturate image
    image[image > 2**bit_depth-1] = 2**bit_depth-1

    # Return result
    return image

@jit(nopython=True)
def conv2(img_in, kernel):
    """
    Convolve image with input kernel.

    Parameters
    ----------
    img_in : ndarray
        Input image.
    kernel : ndarray
        Input kernel. Must be square and have an odd number of rows.

    Returns
    -------
    img : ndarray
        Convolved image.

    Examples
    --------
    >>> image = np.random.rand(512,512)
    >>> kernel = np.ones((7,7))
    >>> image_conv = conv2(image, kernel)
    """

    # Check for valid kernel
    if kernel.shape[0] % 2 != 1:
        raise ValueError("Kernel size must be odd.")
    if kernel.shape[0] != kernel.shape[1]:
        raise NotImplementedError("Non-square kernels not currently supported.")

    # Allocate variables
    size = kernel.shape[0]
    size_half = int(np.floor(kernel.shape[0]/2))
    rows, cols = img_in.shape
    img = np.copy(img_in)
    img_pad = np.zeros((rows+2*size_half, cols+2*size_half))
    img_pad[size_half:-(size_half), size_half:-(size_half)] = img

    # Convolve image with kernel
    for row in range(rows):
        for col in range(cols):
            img[row, col] = np.sum(img_pad[row:size+row, col:size+col]*kernel)

    # Return result
    return img

def in_frame(resolution, img_x, img_y, buffer=0.5):
    """
    Return a list of indices corresponding to coordinates in the bounds of the
    image frame.

    Parameters
    ----------
    resolution : tuple
        Resolution along the x and y axes, respectively.
    img_x : ndarray
        Input coordinates along the x axis.
    img_y : ndarray
        Input coordinates along the y axis.
    buffer: float, optional
        Buffer width for inclusion of slightly out-of-frame coordinates.
        Default is half a pixel.

    Returns
    -------
    in_bounds : list
        List of indices corresponding to the (x,y) coordinates within the
        image frame.
    """

    return [idx for idx in range(len(img_x)) if (img_x[idx] >= 0-buffer                 and
                                                 img_x[idx] <= resolution[0]-1+buffer   and
                                                 img_y[idx] >= 0-buffer                 and
                                                 img_y[idx] <= resolution[1]-1+buffer)]

def imshow(img, scale=None):
    """
    MATLAB-like imshow function.

    Parameters
    ----------
    image : ndarray
        Input image.
    scale : ndarray, optional
        Minimum and maximum scale. Defaults to the minimum and maximum values
        in the image.

    Returns
    -------
    None

    Notes
    -----
    Will halt script until user exits the image window.

    Examples
    --------
    >>> imshow(np.random.rand((512,512))
    """

    # Assign default scale
    if not scale:
        scale = [np.min(img), np.max(img)]

    # Set up image plot
    plt.imshow(img, cmap='gray', vmin=scale[0], vmax=scale[1])
    plt.xticks([]), plt.yticks([])

    # Show
    plt.show()
