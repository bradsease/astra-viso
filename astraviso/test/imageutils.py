"""
Imageutils unit tests.
"""
from __future__ import division
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

class test_vismag2photon(imageutilstests):
    """
    Test vismag2photon function.
    """

    def test_single(self):
        """
        Test output value and type for single input.
        """

        # Set up visible magnitudes
        vismags = -1

        # Convert to photons
        photons = iu.vismag2photon(vismags, 1, 1, 1)

        # Check output
        self.assertIsInstance(photons, float, "Output type should be float.")
        self.assertGreater(photons, 0, "Photon count must be positive.")

    def test_single(self):
        """
        Test output value and type for multiple input.
        """

        # Set up visible magnitudes
        vismags = np.array([1, 0, -1])

        # Convert to photons
        photons = iu.vismag2photon(vismags, 1, 1, 1)

        # Check output
        self.assertEqual(len(photons), len(vismags), "Output size not equal to input.")
        self.assertIsInstance(photons, np.ndarray, "Output type should be float.")
        self.assertTrue(np.all(photons>0), "Photon counts must be positive.")
        self.assertGreater(photons[2], photons[0], "Incorrect output values.")
        self.assertEqual(photons[1], 1, "Incorrect output value for input 0.")

class test_apply_constant_qe(imageutilstests):
    """
    Test apply_constant_quantum_efficiency function.
    """

    def test_zero(self):
        """
        Test output value and type for zero QE.
        """

        # Convert to photoelectrons
        photo_electrons = iu.apply_constant_quantum_efficiency(16*np.ones((16,16)), 0)

        # Check output
        self.assertIsInstance(photo_electrons, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(photo_electrons==0), "Output values should all be equal to 0.")

    def test_positive(self):
        """
        Test output value and type for positive QE.
        """

        # Convert to photoelectrons
        photo_electrons = iu.apply_constant_quantum_efficiency(16*np.ones((16,16)), 0.4)

        # Check output
        self.assertIsInstance(photo_electrons, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(photo_electrons==6), "Output values should all be equal to 6.")

class test_apply_gaussian_qe(imageutilstests):
    """
    Test apply_gaussian_quantum_efficiency function.
    """

    def test_zero(self):
        """
        Test output value and type for zero QE.
        """

        # Create test image
        test_image = 16*np.ones((16,16))

        # Convert to photoelectrons
        photo_electrons = iu.apply_gaussian_quantum_efficiency(test_image, 0, 0)

        # Check output
        self.assertIsInstance(photo_electrons, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(photo_electrons==0), "Output values should all be equal to 0.")

    def test_seed(self):
        """
        Test RNG seed capability.
        """

        # Create test image
        test_image = 16*np.ones((16,16))

        # Convert to photoelectrons
        photo_electrons_1 = iu.apply_gaussian_quantum_efficiency(test_image, 0.2, 0.01, seed=1)
        photo_electrons_2 = iu.apply_gaussian_quantum_efficiency(test_image, 0.2, 0.01, seed=1)

        # Check output
        self.assertIsInstance(photo_electrons_1, np.ndarray, "Output type should be ndarray.")
        self.assertIsInstance(photo_electrons_2, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(photo_electrons_1==photo_electrons_2),                              \
                                                        "Seed does not lead to consistent results.")

    def test_positive(self):
        """
        Test RNG seed capability.
        """

        # Create test image
        test_image = 16*np.ones((128,128))

        # Convert to photoelectrons
        photo_electrons = iu.apply_gaussian_quantum_efficiency(test_image, 0, 1, seed=1)

        # Check output
        self.assertIsInstance(photo_electrons, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(photo_electrons>=0), "Quantum efficiency must be strictly positive.")

class test_saturate(imageutilstests):
    """
    Test saturate function.
    """

    def test_no_clipping(self):
        """
        Test output value and type for array input and sufficient bit_depth.
        """

        # Compute saturated image
        saturated = iu.saturate(16*np.ones((16,16)), 8)

        # Check output
        self.assertIsInstance(saturated, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(saturated==16), "Output values should all be equal to 16.")

    def test_clipping(self):
        """
        Test output value and type for array input and insufficient bit_depth.
        """

        # Compute saturated image
        saturated = iu.saturate(16*np.ones((16,16)), 2)

        # Check output
        self.assertIsInstance(saturated, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(np.all(saturated==3), "Output values should all be equal to 3.")

class test_conv2(imageutilstests):
    """
    Test conv2 function.
    """

    def test_3by3(self):
        """
        Test 3x3 convoltuion kernel.
        """

        # Create kernel & image
        kernel = np.ones((3,3))
        image = np.ones((64,64))

        # Convolve
        result = iu.conv2(image, kernel)

        # Check result
        self.assertIsInstance(result, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(image.shape, result.shape, "Image shape must be preserved.")
        self.assertTrue(np.all(result[1:-2,1:-2] == 9), "Incorrect pixel values.")

    def test_exceptions(self):
        """
        Verify conv2 exceptions.
        """

        # Create kernel & image
        kernel = np.ones((3,3))
        image = np.ones((64,64))

        # Test even kernel
        with self.assertRaises(ValueError):
            iu.conv2(image, np.ones((2,2)))

        # Test rectangular kernel
        with self.assertRaises(ValueError):
            iu.conv2(image, np.ones((2,3)))

class test_in_frame(imageutilstests):
    """
    Test in_frame function.
    """

    def test_diagonal(self):
        """
        Test all diagonal coordinates.
        """

        # Create diagonal
        img_x = np.array(range(16))
        img_y = np.array(range(16))

        # Compute result
        result = iu.in_frame((16,16), img_x, img_y)

        # Check result
        self.assertIsInstance(result, list, "Incorrect output type.")
        self.assertEqual(len(result), len(img_x), "Incorrect output size.")
