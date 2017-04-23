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
        img_x, img_y = self.starcam.body2plane(np.array([0, 0, 1]))
        self.assertIsInstance(img_x, np.ndarray, "X coordinate output should be ndarray.")
        self.assertIsInstance(img_y, np.ndarray, "X coordinate output should be ndarray.")
        self.assertEqual(len(img_x), 1, "Output dimension should equal input dimension")
        self.assertEqual(len(img_x), len(img_y), "Number of x and y coordinates should be equal.")
        self.assertEqual(img_x[0], img_y[0], "For this case, coordinates should be equal.")
        self.assertEqual(img_x[0], 512.5, "For this case, coordinate value should be 512.5.")

    def test_multiple_pinhole(self):
        """
        Test multiple point projection.
        """

        #  Convert vector and check dimensions
        img_x, img_y = self.starcam.body2plane(np.array([[0, 0, 1], [0, 0, -1]]))
        self.assertIsInstance(img_x, np.ndarray, "X coordinate output should be ndarray.")
        self.assertIsInstance(img_y, np.ndarray, "X coordinate output should be ndarray.")
        self.assertEqual(len(img_x), 2, "Output dimension should equal input dimension")
        self.assertEqual(len(img_x), len(img_y), "Number of x and y coordinates should be equal.")
        self.assertTrue(all(img_x == img_y), "For this case, coordinates should be equal.")

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
        self.assertEqual(np.sum(image), 0, "Images must be strictly positive.")

    def test_nonempty_star_image(self):
        """
        Test simple non-empty star image.
        """

        # Generate image
        image = self.starcam.integrate(1)
        self.assertEqual(image.shape[0], self.starcam.resolution, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], self.starcam.resolution, "Y Resolution incorrect.")
        self.assertTrue((image >= 0).all(), "Images must be strictly positive.")

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

class test_set_noise_fcn(starcamtests):
    """
    Test set_noise_fcn method.
    """

    def test_constant_set(self):
        """
        Test constant value function.
        """

        # Set up environment
        noise_fcn = lambda image, delta_t: image + 2
        test_image = np.zeros((256,256))

        # Set noise function and 
        self.starcam.set_noise_fcn(noise_fcn)
        noisy_image = self.starcam.noise_fcn(test_image, 0)

        # Check internal noise function
        self.assertIs(noise_fcn, self.starcam.noise_fcn, "Function set failed.")
        self.assertTrue(np.all(noisy_image == 2), "Incorrect internal noise function.")

class test_set_noise_preset(starcamtests):
    """
    Test set_noise_preset method.
    """

    def test_poisson(self):
        """
        Test poisson noise model.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set poisson preset
        self.starcam.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        noisy_image = self.starcam.noise_fcn(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image >= 0), "Image with noise should be strictly positive.")

    def test_gaussian(self):
        """
        Test poisson noise model.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set poisson preset
        self.starcam.set_noise_preset("gaussian", dark_current=1200, read_noise=200)
        noisy_image = self.starcam.noise_fcn(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")

class test_add_noise(starcamtests):
    """
    Test set_noise_preset method.
    """

    def test_none(self):
        """
        Test add_noise with no noise function set.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set noise_fcn to None
        self.starcam.noise_fcn = None
        noisy_image = self.starcam.add_noise(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image == test_image), "Should contents should be unchanged.")

    def test_poisson(self):
        """
        Test poisson noise model.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set poisson preset
        self.starcam.set_noise_preset("poisson", dark_current=1200, read_noise=200)
        noisy_image = self.starcam.add_noise(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image >= 0), "Image with noise should be strictly positive.")

    def test_gaussian(self):
        """
        Test poisson noise model.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set poisson preset
        self.starcam.set_noise_preset("gaussian", dark_current=1200, read_noise=200)
        noisy_image = self.starcam.add_noise(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
