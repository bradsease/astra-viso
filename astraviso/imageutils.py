"""
Image utilities for astra-viso.
"""
import numpy as np

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
    Add gaussian-distributed noise to an image. Approximates poisson-distributed noise.

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
        Aperture size in mm^2.
    mv0_flux : float
        Photoelectrons per second per mm^2 of aperture area.

    Returns
    -------
    photons : float or ndarray
        Total photon count for each input visible magnitude.

    Notes
    -----
    Based on: Liebe, Carl Christian. "Accuracy performance of star trackers-a tutorial."
              IEEE Transactions on aerospace and electronic systems 38.2 (2002): 587-599.

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
    return mv0_flux * (1 / (2.5**vismags)) * delta_t * aperture
