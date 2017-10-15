"""
StarCam unit tests.
"""
from __future__ import division
import unittest
from astraviso import starcam as cam
from astraviso import worldobject
import numpy as np

class starcamtests(unittest.TestCase):
    """
    StarCam unit test class.
    """

    def setUp(self):
        self.starcam = cam.StarCam()

    def tearDown(self):
        del self.starcam

class test_add_worldobject(starcamtests):
    """
    Test add_worldobject method.
    """

    def test_multi(self):
        """
        Test multiple input objects
        """

        # Add object iteratively
        for idx in range(5):

            # Create object
            obj = worldobject.WorldObject()

            # Add to cam
            self.starcam.add_worldobject(obj)

            # Check result
            self.assertEqual(len(self.starcam.external_objects), idx+1, "Incorrect number of       \
                                                                                 catalog elements.")
            self.assertIsInstance(self.starcam.external_objects[-1], worldobject.WorldObject,      \
                                                                  "Incorrect catalog element type.")

    def test_multi_auto(self):
        """
        Test multiple auto-created objects
        """

        # Add object iteratively
        for idx in range(5):

            # Add to cam
            self.starcam.add_worldobject()

            # Check result
            self.assertEqual(len(self.starcam.external_objects), idx+1, "Incorrect number of       \
                                                                                 catalog elements.")
            self.assertIsInstance(self.starcam.external_objects[-1], worldobject.WorldObject,      \
                                                                  "Incorrect catalog element type.")

class test_delete_worldobject(starcamtests):
    """
    Test delete_worldobject method.
    """

    def test_multi(self):
        """
        Test multiple objects
        """

        # Build object list
        self.starcam.external_objects = [worldobject.WorldObject for idx in range(5)]

        # Add object iteratively
        for idx in reversed(range(len(self.starcam.external_objects))):

            # Delete last object
            self.starcam.delete_worldobject(idx)

            # Check result
            self.assertEqual(len(self.starcam.external_objects), idx, "Incorrect number of       \
                                                                                 catalog elements.")

class test_get_boresight(starcamtests):
    """
    Test get_boresight method.
    """

    def test_single(self):
        """
        Test get_boresight.
        """

        # Get test result
        result = self.starcam.get_boresight(0)
        expected_result = np.array([0, 0, 1])

        # Check result
        self.assertIsInstance(result, np.ndarray, "Output must be ndarray.")
        self.assertEqual(len(result), 3, "Output must have three elements.")
        self.assertTrue(np.all(np.isclose(result, expected_result)), "Incorrect boresight result.")

class test_integrate(starcamtests):
    """
    Test integrate method.
    """

    def test_empty_star_image(self):
        """
        Test empty image.
        """

        # Generate empty image
        image = self.starcam.integrate(0, 0)
        self.assertEqual(image.shape[0], 1024, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], 1024, "Y Resolution incorrect.")
        self.assertEqual(np.sum(image), 0, "Image be all zero.")

    def test_nonempty_star_image(self):
        """
        Test simple non-empty star image.
        """

        # Generate image
        image = self.starcam.integrate(0, 1)
        self.assertEqual(image.shape[0], 1024, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], 1024, "Y Resolution incorrect.")
        self.assertTrue((image >= 0).all(), "Images must be strictly positive.")

    def test_worldobject_and_stars(self):
        """
        Test an external object with stars.
        """

        # Add default worldobject
        self.starcam.add_worldobject()

        # Generate image
        image = self.starcam.integrate(0, 1)

        # Check result
        self.assertEqual(image.shape[0], 1024, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], 1024, "Y Resolution incorrect.")
        self.assertTrue((image >= 0).all(), "Images must be strictly positive.")

    def test_worldobject_only(self):
        """
        Test an external object and no stars.
        """

        # Empty star catalog & remove noise
        self.starcam.star_catalog.load_preset("random", 0)
        self.starcam.set_noise_preset("off")

        # Add default worldobject
        self.starcam.add_worldobject()

        # Generate image
        image = self.starcam.integrate(0, 1)

        # Check result
        self.assertEqual(image.shape[0], 1024, "X Resolution incorrect.")
        self.assertEqual(image.shape[1], 1024, "Y Resolution incorrect.")
        self.assertGreater(np.sum(image), 0, "Image must contain non-zero pixels.")

class test_sequence(starcamtests):
    """
    Test sequence method.
    """

    def test_creation(self):
        """
        Simple sequence creation tests.
        """

        seq = self.starcam.sequence(0, 1, 10)
        self.assertTrue(hasattr(seq, "__iter__"), "Sequence is not iterable.")

        seq = self.starcam.sequence(0, 1, 10, delay=2)
        self.assertTrue(hasattr(seq, "__iter__"), "Sequence is not iterable.")

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

    def test_off(self):
        """
        Test zero image noise option.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set poisson preset
        self.starcam.set_noise_preset("off")
        noisy_image = self.starcam.noise_fcn(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image == 0), "All image values should be zero.")

class test_add_noise(starcamtests):
    """
    Test set_noise_preset method.
    """

    def test_none(self):
        """
        Test add_noise with "off" preset.
        """

        # Set up environment
        test_image = np.zeros((256,256))

        # Set noise_fcn to None
        self.starcam.set_noise_preset("off")
        noisy_image = self.starcam.add_noise(test_image, 1)

        # Check output
        self.assertIsInstance(noisy_image, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(noisy_image.shape, test_image.shape, "Image shape should be preserved.")
        self.assertTrue(np.all(noisy_image == test_image), "Image contents should be unchanged.")

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

class test_set_sensitivity_fcn(starcamtests):
    """
    Test set_sensitivity_fcn method.
    """

    def test_constant_set(self):
        """
        Test constant value function.
        """

        # Set up photon function
        sensitivity_fcn = lambda vismags, delta_t : vismags

        # Set function
        self.starcam.set_sensitivity_fcn(sensitivity_fcn)

        # Check function set
        self.assertTrue(callable(self.starcam.sensitivity_fcn), "Function not callable.")
        self.assertIs(self.starcam.sensitivity_fcn, sensitivity_fcn, "Function set failed.")

class test_set_sensitivity_preset(starcamtests):
    """
    Test set_sensitivity_preset method.
    """

    def test_default(self):
        """
        Test default photon model with input options.
        """

        # Set default preset with default options
        self.starcam.set_sensitivity_preset("default", aperture=1, mv0_flux=1)

        # Check function
        self.assertTrue(callable(self.starcam.sensitivity_fcn), "Function not callable.")
        self.assertEqual(self.starcam.sensitivity_fcn(0,2), 2, "Incorrect result.")

class test_get_photons(starcamtests):
    """
    Test set_get_photons method.
    """

    def test_default(self):
        """
        Test default photon model.
        """

        # Set default preset with default options
        self.starcam.set_sensitivity_preset("default", aperture=1, mv0_flux=1)
        result = self.starcam.get_photons(0,2)
        result_multi = self.starcam.get_photons([0,0], 2)

        # Check function
        self.assertEqual(result, 2, "Incorrect result for single input.")
        self.assertTrue(np.all(result_multi == 2), "Incorrect result for multi-input case.")

class test_set_projection_fcn(starcamtests):
    """
    Test set_projection_fcn method.
    """

    def test_simple_set(self):
        """
        Test simple projection function.
        """

        # Set up projection function
        def proj_fcn(vectors):
            img_x = np.divide(vectors[:, 0], vectors[:, 2])
            img_y = np.divide(vectors[:, 1], vectors[:, 2])
            return img_x, img_y

        # Set function
        self.starcam.set_projection_fcn(proj_fcn, 1024)

        # Check function set
        self.assertTrue(callable(self.starcam.projection_fcn), "Function not callable.")
        self.assertIs(self.starcam.projection_fcn, proj_fcn, "Function set failed.")

class test_set_projection_preset(starcamtests):
    """
    Test set_projection_preset method.
    """

    def test_pinhole(self):
        """
        Test pinhole preset.
        """

        # Set function
        self.starcam.set_projection_preset("pinhole", focal_len=93, pixel_size=0.016)

        # Compute result
        img_x, img_y = self.starcam.projection_fcn(np.array([[0, 0, 1]]))
        true_img_x = np.array([511.5])
        true_img_y = np.array([511.5])

        # Check function set
        self.assertTrue(callable(self.starcam.projection_fcn), "Function not callable.")
        self.assertTrue(np.all(img_x == true_img_x), "Incorrect x coordinate.")
        self.assertTrue(np.all(img_y == true_img_y), "Incorrect x coordinate.")

class test_get_projection(starcamtests):
    """
    Test get_projection method.
    """

    def test_pinhole(self):
        """
        Test pinhole model.
        """

        # Set function
        self.starcam.set_projection_preset("pinhole", focal_len=93, pixel_size=0.016)

        # Compute result
        img_x, img_y = self.starcam.get_projection(np.array([[0, 0, 1]]))
        true_img_x = np.array([511.5])
        true_img_y = np.array([511.5])

        # Check function set
        self.assertTrue(callable(self.starcam.projection_fcn), "Function not callable.")
        self.assertTrue(np.all(img_x == true_img_x), "Incorrect x coordinate.")
        self.assertTrue(np.all(img_y == true_img_y), "Incorrect x coordinate.")

class test_set_quantum_efficiency_fcn(starcamtests):
    """
    Test set_quantum_efficiency_fcn method.
    """

    def test_floor_set(self):
        """
        Test floor function.
        """

        # Set up quantum efficiency function
        qe_fcn = np.floor

        # Set function
        self.starcam.set_quantum_efficiency_fcn(qe_fcn)

        # Check function set
        self.assertTrue(callable(self.starcam.quantum_efficiency_fcn), "Function not callable.")
        self.assertIs(self.starcam.quantum_efficiency_fcn, qe_fcn, "Function set failed.")

class test_set_quantum_efficiency_preset(starcamtests):
    """
    Test set_quantum_efficiency_preset method.
    """

    def test_constant(self):
        """
        Test constant QE model with input options.
        """

        # Test image
        test_image = 5*np.ones((16,16))

        # Set constant model
        self.starcam.set_quantum_efficiency_preset("constant", quantum_efficiency=0.2)
        test_result = self.starcam.quantum_efficiency_fcn(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.quantum_efficiency_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result==1), "Incorrect result.")

    def test_gaussian(self):
        """
        Test gaussian QE model with input options.
        """

        # Test image
        test_image = 5*np.ones((16,16))

        # Set gaussian model
        self.starcam.set_quantum_efficiency_preset("gaussian", quantum_efficiency=0.2, sigma=0.01)
        test_result = self.starcam.quantum_efficiency_fcn(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.quantum_efficiency_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result >= 0), "Image should be strictly positive.")

    def test_polynomial(self):
        """
        Test polynomial QE model with input options.
        """

        # Test data
        coeffs = np.array([[1,0,1], [0,0,0], [1,0,0]])
        test_image = 5*np.ones((16,16))

        # Set gaussian model
        self.starcam.set_quantum_efficiency_preset("polynomial", poly=coeffs)
        test_result = self.starcam.quantum_efficiency_fcn(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.quantum_efficiency_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result >= 0), "Image should be strictly positive.")

class test_get_photoelectrons(starcamtests):
    """
    Test get_photoelectrons method.
    """

    def test_constant(self):
        """
        Test using constant model.
        """

        # Test image
        test_image = 5*np.ones((16,16))

        # Set no_bleed model
        self.starcam.set_quantum_efficiency_preset("constant", quantum_efficiency=0.2)
        test_result = self.starcam.get_photoelectrons(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.saturation_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result==1), "Incorrect result.")

class test_set_saturation_fcn(starcamtests):
    """
    Test set_saturation_fcn method.
    """

    def test_floor_set(self):
        """
        Test floor function.
        """

        # Set up saturation function
        saturation_fcn = np.floor

        # Set function
        self.starcam.set_saturation_fcn(saturation_fcn)

        # Check function set
        self.assertTrue(callable(self.starcam.saturation_fcn), "Function not callable.")
        self.assertIs(self.starcam.saturation_fcn, saturation_fcn, "Function set failed.")

class test_set_saturation_preset(starcamtests):
    """
    Test set_saturation_preset method.
    """

    def test_no_bleed(self):
        """
        Test no_bleed model with input options.
        """

        # Test image
        test_image = 16*np.ones((16,16))

        # Set no_bleed model
        self.starcam.set_saturation_preset("no_bleed", bit_depth=2)
        test_result = self.starcam.saturation_fcn(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.saturation_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result==3), "Incorrect result.")

    def test_off(self):
        """
        Test no saturation model.
        """

        # Test image
        test_image = 1e8*np.ones((16,16))

        # Set no_bleed model
        self.starcam.set_saturation_preset("off")
        test_result = self.starcam.saturation_fcn(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.saturation_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result==1e8), "Incorrect result.")

class test_get_saturation(starcamtests):
    """
    Test get_saturation method.
    """

    def test_no_bleed(self):
        """
        Test default photon model.
        """

        # Test image
        test_image = 16*np.ones((16,16))

        # Set no_bleed model
        self.starcam.set_saturation_preset("no_bleed", bit_depth=2)
        test_result = self.starcam.get_saturation(test_image)

        # Check function
        self.assertTrue(callable(self.starcam.saturation_fcn), "Function not callable.")
        self.assertTrue(np.all(test_result==3), "Incorrect result.")
