"""
WorldObject unit tests.
"""
import unittest
from astraviso import worldobject as obj
import numpy as np

class worldobjecttests(unittest.TestCase):
    """
    World object test class.
    """

    def setUp(self):
        self.worldobject = obj.WorldObject()

    def tearDown(self):
        del self.worldobject

class test_initialization(worldobjecttests):
    """
    Test world object initialization.
    """

    def test_types(self):
        """
        Test for correct attribute types
        """

        # Check types
        pass

class test_set_pointing_fcn(worldobjecttests):
    """
    Test set_pointing_fcn method.
    """

    def test_ode(self):
        """
        Test ODE input format.
        """

        # Create static "state" function
        fcn = lambda t, state: [0, 0, 0, 0, 0, 0, 0]

        # Set function
        self.worldobject.set_pointing_fcn(fcn, "ode", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Check properties
        self.assertEqual(self.worldobject.model_pointing, "on", "Incorrect modeling option.")
        self.assertEqual(self.worldobject.pointing_mode, "ode", "Incorrect pointing mode.")

        # Check function output
        test_result = self.worldobject.pointing_fcn(1)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

    def test_explicit(self):
        """
        Test explicit input format.
        """

        # Create "state" function
        fcn = lambda t: np.array([0, 0, 0, 1, 0, 0, 0])

        # Set function
        self.worldobject.set_pointing_fcn(fcn, "explicit")

        # Check properties
        self.assertEqual(self.worldobject.model_pointing, "on", "Incorrect modeling option.")
        self.assertEqual(self.worldobject.pointing_mode, "explicit", "Incorrect pointing mode.")

        # Check function output
        test_result = self.worldobject.pointing_fcn(1)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

class test_set_pointing_preset(worldobjecttests):
    """
    Test set_pointing_preset method.
    """

    def test_kinematic_preset(self):
        """
        Test rigid body kinematic preset.
        """

        # Set preset
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Check function static output
        test_result = self.worldobject.pointing_fcn(1)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(np.all(test_result == expected_result),                                   \
                                                 "For zero angular rate, result should be static.")

        # Set preset
        self.worldobject.angular_rate = np.array([2*np.pi, 0, 0])
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Check function single rotation output
        test_result = self.worldobject.pointing_fcn(2)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])
        self.assertTrue(np.all(np.isclose(test_result, expected_result)),              \
                                                   "Quaternion should return after two rotations.")

class test_get_pointing(worldobjecttests):
    """
    Test get_pointing method.
    """

    def test_single_quaternion(self):
        """
        Test single quaternion output.
        """

        # Set pointing to kinematic
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Get pointing
        quaternion = self.worldobject.get_pointing(1)

        # Check type, size, and value
        self.assertIsInstance(quaternion, np.ndarray, "Output type should be ndarray.")
        self.assertEqual(len(quaternion), 4, "Output size should be 4.")
        self.assertTrue(np.all(quaternion == np.array([0, 0, 0, 1])),                             \
                                                                    "Quaternion should be static.")

    def test_multi_quaternion(self):
        """
        Test single quaternion output.
        """

        # Number of test cases
        num = 9

        # Set pointing to kinematic
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Get pointing
        quaternion = self.worldobject.get_pointing(np.arange(num)+1)

        # Check type, size, and value
        self.assertIsInstance(quaternion, np.ndarray, "Output type should be ndarray.")
        self.assertTrue(quaternion.shape == (num, 4), "Output size should be 4.")
        for idx in range(num):
            self.assertTrue(np.all(quaternion[idx, :] == np.array([0, 0, 0, 1])),                  \
                                                                                "Incorrect result.")

    def test_single_dcm(self):
        """
        Test single dcm output.
        """

        # Set pointing to kinematic
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Get pointing
        dcm = self.worldobject.get_pointing(1, mode="dcm")

        # Check type and value
        self.assertIsInstance(dcm, np.ndarray, "Incorrect output type.")
        self.assertTrue(dcm.shape == (3, 3), "Incorrect output size.")
        self.assertTrue(np.all(dcm == np.eye(3)), "Incorrect result.")

    def test_multi_dcm(self):
        """
        Test single quaternion output.
        """

        # Number of test cases
        num = 9

        # Set pointing to kinematic
        self.worldobject.set_pointing_preset("kinematic", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Get pointing
        dcm = self.worldobject.get_pointing(np.arange(num)+1, mode="dcm")

        # Check type, size, and value
        self.assertIsInstance(dcm, np.ndarray, "Incorrect output type.")
        self.assertTrue(dcm.shape == (num, 3, 3), "Incorrect output size.")
        for idx in range(num):
            self.assertTrue(np.all(dcm[idx, :, :] == np.eye(3)), "Incorrect result.")

class test_set_integrator(worldobjecttests):
    """
    Test set_integrator method.
    """

    def test_set(self):
        """
        Test basic integrator set.
        """
        # Set integrator
        self.worldobject.set_integrator("vode", 1e-8, 1e-9)

        # Check properties
        self.assertEqual(self.worldobject.integrator, "vode", "Incorrect integrator.")
        self.assertEqual(self.worldobject.integrator_atol, 1e-8, "Incorrect absolute tolerance.")
        self.assertEqual(self.worldobject.integrator_rtol, 1e-9, "Incorrect relative tolerance.")

    def test_set_with_pointing_ode(self):
        """
        Test integrator set with a pointing ode that requires re-initialization.
        """

        # Create "state" function
        fcn = lambda t, state: [0, 0, 0, 0, 0, 0, 0]

        # Set function
        self.worldobject.set_pointing_fcn(fcn, "ode", np.array([0, 0, 0, 1, 0, 0, 0]))

        # Set integrator
        self.worldobject.set_integrator("vode", 1e-8, 1e-9)
