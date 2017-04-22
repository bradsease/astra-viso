"""
Imageutils unit tests.
"""
import unittest
import numpy as np
from astraviso import imageutils as iu

class imageutilstests(unittest.TestCase):
    """
    Imageutils unit test class.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

class test_poisson_noise(imageutilstests):
    """
    Test poisson_noise function.
    """

    def test_empty_image(self):
        """
        Test output value and type.
        """

        # Allocate placeholder image
        image = np.zeros((512))

        # Add noise
        noisy_image = iu.poisson_noise(image, 0, 1200, 200)

        # Check result
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image >= 0), "Image with noise should be strictly positive.")

class test_gaussian_noise(imageutilstests):
    """
    Test gaussian_noise function.
    """

    def test_empty_image(self):
        """
        Test output value and type.
        """

        # Allocate placeholder image
        image = np.zeros((512))

        # Add noise
        noisy_image = iu.gaussian_noise(image, 0, 1200, 200)

        # Check result
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, image.shape, "Image shape should be preserved.")
        