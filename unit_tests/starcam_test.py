"""
StarCam unit tests.
"""
import unittest
from astraviso import starcam as cam
import numpy as np

class starcamtests(unittest.TestCase):
    """
    StarCam unit test class.
    """

    def setUp(self):
        self.starcam = cam.StarCam()

    def tearDown(self):
        del self.starcam

class default_tests(starcamtests):
    """
    Test default parameters.
    """

    def test_focal_length(self):
        """
        Test default focal length.
        """

        self.assertEqual(self.starcam.focal_len, 93, "Default focal length incorrect.")

class test_body2plane(starcamtests):
    """
    Test body2plane method.
    """

    def test_single_pinhole(self):
        """
        Test single point projection.
        """

        #  Convert vector and check dimensions
        output = self.starcam.body2plane([0, 0, 1])
        self.assertEqual(len(output), 2, "Incorrect coordinate dimensions.")
        self.assertEqual(len(output[0]), 1, "Incorrect number of dimensions.")
        self.assertEqual(len(output[0]), len(output[1]), "Mis-matched x and y coordinates.")
        self.assertEqual(output[0], output[1], "Incorrect projection.")
        self.assertEqual(output[0], 512.5, "Incorrect projection.")

    def test_multiple_pinhole(self):
        """
        Test multiple point projection.
        """

        #  Convert vector and check dimensions
        output = self.starcam.body2plane([[0, 0, 1], [0, 0, -1]])
        self.assertEqual(len(output), 2, "Incorrect coordinate dimensions.")
        self.assertEqual(len(output[0]), 2, "Incorrect number of dimensions.")
        self.assertEqual(len(output[0]), len(output[1]), "Mis-matched x and y coordinates.")
        self.assertTrue(all(output[0] == output[1]), "Incorrect projection.")

class test_integrate(starcamtests):
    """
    Test integrate method.
    """

    def test_empty_star_image(self):
        """
        Test empty image.
        """

        # Generate empty image
        image = self.starcam.integrate(0)
        self.assertEqual(image.shape[0], self.starcam.resolution, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], self.starcam.resolution, "Y Resolution incorrect.")
        self.assertEqual(np.sum(image), 0, "Image not strictly positive.")

    def test_nonempty_star_image(self):
        """
        Test simple non-empty star image.
        """

        # Generate image
        image = self.starcam.integrate(1)
        self.assertEqual(image.shape[0], self.starcam.resolution, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], self.starcam.resolution, "Y Resolution incorrect.")
        self.assertTrue((image >= 0).all(), "Image not strictly positive.")

class test_setpsf(starcamtests):
    """
    Test setpsf method.
    """

    def test_nominal_dimensions(self):
        """
        Test nominal case dimensions.
        """

        # Set PSF -- nominal case
        self.starcam.setpsf(11, 1)
        self.assertEqual(self.starcam.psf.shape[0], 11, "PSF size incorrect.")
        self.assertEqual(self.starcam.psf.shape[1], 11, "PSF size incorrect.")
        self.assertEqual(np.sum(self.starcam.psf), 1, "PSF sum incorrect.")

    def test_off_nominal_dimensions(self):
        """
        Test off nominal dimensions.
        """

        # Set PSF -- off-nominal case
        self.starcam.setpsf(10, 1)
        self.assertEqual(self.starcam.psf.shape[0], 11, "PSF size incorrect.")
        self.assertEqual(self.starcam.psf.shape[1], 11, "PSF size incorrect.")
        self.assertEqual(np.sum(self.starcam.psf), 1, "PSF sum incorrect.")

class test_set(starcamtests):
    """
    Test set method.
    """

    def test_identical_set(self):
        """
        Test identical overwrite scenario.
        """

        # Store current values
        focal_len = self.starcam.focal_len
        pixel_size = self.starcam.pixel_size
        resolution = self.starcam.resolution

        # Update values
        self.starcam.set(focal_len=focal_len, resolution=resolution, pixel_size=pixel_size)

        # Check result
        self.assertEqual(focal_len, self.starcam.focal_len, "Focal length not preserved.")
        self.assertEqual(pixel_size, self.starcam.pixel_size, "Pixel size not preserved.")
        self.assertEqual(resolution, self.starcam.resolution, "Resolution not preserved.")
        self.assertTrue(isinstance(self.starcam.resolution, int), "Resolution not integer-valued")

    def test_resolution_calc(self):
        """
        Test set without resolution.
        """

		# Test values
        focal_len = self.starcam.focal_len
        pixel_size = self.starcam.pixel_size
        fov = 10.0679286799

        # Update
        self.starcam.set(focal_len=focal_len, pixel_size=pixel_size, fov=fov)

        # Check result
        self.assertEqual(focal_len, self.starcam.focal_len, "Focal length not preserved.")
        self.assertEqual(pixel_size, self.starcam.pixel_size, "Pixel size not preserved.")
        self.assertTrue(np.isclose(self.starcam.resolution, 1024))
        self.assertTrue(isinstance(self.starcam.resolution, int), "Resolution not integer-valued")

    def test_pixel_size_calc(self):
        """
        Test set without pixel_size.
        """

		# Test values
        focal_len = self.starcam.focal_len
        resolution = 1024
        fov = 10.0679286799

        # Update
        self.starcam.set(focal_len=focal_len, resolution=resolution, fov=fov)

        # Check result
        self.assertEqual(focal_len, self.starcam.focal_len, "Focal length not preserved.")
        self.assertEqual(resolution, self.starcam.resolution, "Resolution is not preserved.")
        self.assertTrue(np.isclose(self.starcam.pixel_size, 0.016))
        self.assertTrue(isinstance(self.starcam.resolution, int), "Resolution not integer-valued")

    def test_focal_len_calc(self):
        """
        Test set without focal length.
        """

		# Test values
        resolution = 1024
        pixel_size = self.starcam.pixel_size
        fov = 10.0679286799

        # Update
        self.starcam.set(resolution=resolution, pixel_size=pixel_size, fov=fov)

        # Check result
        self.assertEqual(pixel_size, self.starcam.pixel_size, "Pixel size not preserved.")
        self.assertEqual(resolution, self.starcam.resolution, "Resolution is not preserved.")
        self.assertTrue(np.isclose(self.starcam.focal_len, 93), "Incorrect focal length.")
        self.assertTrue(isinstance(self.starcam.resolution, int), "Resolution not integer-valued")
