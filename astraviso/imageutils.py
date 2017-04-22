"""
Image utilities for astra-viso.
"""
import numpy as np

def poisson_noise(image, delta_t, dark_current, read_noise):
    """
    """

    # Add shot noise
    image = np.random.poisson(image)

    # Add dark current
    image += np.random.poisson(dark_current*delta_t, image.shape)

    # Add read noise
    image += np.random.poisson(read_noise, image.shape)

    return image

def gaussian_noise(image, delta_t, dark_current, read_noise):
    """
    """

    # Add shot noise
    image += np.round(np.sqrt(image) * np.random.randn(*image.shape))

    # Add dark current
    image += np.round(dark_current*delta_t + np.sqrt(dark_current*delta_t) *                       \
                                                                      np.random.randn(*image.shape))

    # Add read noise
    image += np.round(read_noise + np.sqrt(read_noise) * np.random.randn(*image.shape))

    return image
