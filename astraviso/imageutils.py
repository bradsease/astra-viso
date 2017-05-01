"""
Image utilities for astra-viso.
"""
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

def constant_quantum_efficiency(photon_image, quantum_efficiency):
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
    >>> constant_quantum_efficiency(5*np.ones((4,4)), 0.2)
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

@jit
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

def imshow(img, scale=[]):
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
