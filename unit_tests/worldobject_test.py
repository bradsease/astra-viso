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

        # Compute result
        test_result = self.worldobject.pointing_fcn(1)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

    def test_explicit(self):
        """
        Test explicit input format.
        """

        # Create "state" function
        fcn = lambda t: np.array([0, 0, 0, 1, 0, 0, 0])

        # Set function
        self.worldobject.set_pointing_fcn(fcn, "explicit")

        # Compute result
        test_result = self.worldobject.pointing_fcn(1)
        expected_result = np.array([0, 0, 0, 1, 0, 0, 0])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
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

class test_set_position_fcn(worldobjecttests):
    """
    Test set_position_fcn method.
    """

    def test_ode_static(self):
        """
        Test ODE input format.
        """

        # Create static "state" function
        fcn = lambda t, state: [0, 0, 0]

        # Set function
        self.worldobject.set_position_fcn(fcn, "ode", np.array([1, 1, 1]))

        # Compute result
        test_result = self.worldobject.position_fcn(1)
        expected_result = np.array([1, 1, 1])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

    def test_ode_drift(self):
        """
        Test ODE input format.
        """

        # Create static "state" function
        fcn = lambda t, state: [1, 1, 1]

        # Set function
        self.worldobject.set_position_fcn(fcn, "ode", np.array([0, 0, 0]))

        # Compute result
        test_result = self.worldobject.position_fcn(1)
        expected_result = np.array([1, 1, 1])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.allclose(test_result, expected_result), "Incorrect function result.")

    def test_explicit(self):
        """
        Test explicit input format.
        """

        # Create "state" function
        fcn = lambda t: np.array([t, t, t])

        # Set function
        self.worldobject.set_position_fcn(fcn, "explicit")

        # Compute result
        test_result = self.worldobject.position_fcn(1)
        expected_result = np.array([1, 1, 1])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

class test_set_position_preset(worldobjecttests):
    """
    Test set_position_preset method.
    """

    def test_kinematic(self):
        """
        Test kinematic preset.
        """

        # Set function
        self.worldobject.set_position_preset("kinematic", initial_position=np.array([1, 1, 1]),    \
                                                               initial_velocity=np.array([1, 1, 1]))

        # Compute result
        test_result = self.worldobject.position_fcn(1)
        expected_result = np.array([2, 2, 2])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

class test_get_position(worldobjecttests):
    """
    Test get_position method.
    """

    def test_kinematic(self):
        """
        Test kinematic preset.
        """

        # Set function
        self.worldobject.set_position_preset("kinematic", initial_position=np.array([1, 1, 1]),    \
                                                               initial_velocity=np.array([1, 1, 1]))

        # Compute result
        test_result = self.worldobject.get_position(1)
        expected_result = np.array([2, 2, 2])

        # Check function output
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(test_result == expected_result), "Incorrect function result.")

class test_set_vismag_fcn(worldobjecttests):
    """
    Test set_vismag_fcn method.
    """

    def test_constant(self):
        """
        Test static input function
        """

        # Create static "state" function
        fcn = lambda t, *_: 5.0

        # Set function
        self.worldobject.set_vismag_fcn(fcn)

        # Compute result
        test_result = self.worldobject.vismag_fcn(1)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 5.0, "Incorrect function result.")

    def test_sine(self):
        """
        Test sinusoidal input function
        """

        # Create static "state" function
        fcn = lambda t, *_: 7 + 2*np.sin(2*np.pi*t/30)

        # Set function
        self.worldobject.set_vismag_fcn(fcn)

        # Compute result
        test_result = self.worldobject.vismag_fcn(0)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 7.0, "Incorrect function result.")

class test_set_vismag_preset(worldobjecttests):
    """
    Test set_vismag_preset method.
    """

    def test_constant(self):
        """
        Test static preset.
        """

        # Set function
        self.worldobject.set_vismag_preset("constant", vismag=5.0)

        # Compute result
        test_result = self.worldobject.vismag_fcn(1)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 5.0, "Incorrect function result.")

    def test_sine(self):
        """
        Test sine preset.
        """

        # Set function
        self.worldobject.set_vismag_preset("sine", vismag=7, amplitude=2, frequency=30)

        # Compute result
        test_result = self.worldobject.vismag_fcn(0)
        test_result2 = self.worldobject.vismag_fcn(7.5)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 7.0, "Incorrect function result.")
        self.assertEqual(test_result2, 9.0, "Incorrect function result.")

class test_get_vismag(worldobjecttests):
    """
    Test get_vismag method.
    """

    def test_constant(self):
        """
        Test static preset.
        """

        # Set function
        self.worldobject.set_vismag_preset("constant", vismag=5.0)

        # Compute result
        test_result = self.worldobject.get_vismag(1, None)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 5.0, "Incorrect function result.")

    def test_sine(self):
        """
        Test sine preset.
        """

        # Set function
        self.worldobject.set_vismag_preset("sine", vismag=7, amplitude=2, frequency=30)

        # Compute result
        test_result = self.worldobject.get_vismag(0, None)
        test_result2 = self.worldobject.vismag_fcn(7.5)

        # Check function output
        self.assertIsInstance(test_result, float, "Incorrect output type.")
        self.assertEqual(test_result, 7.0, "Incorrect function result.")
        self.assertEqual(test_result2, 9.0, "Incorrect function result.")

class test_relative_to(worldobjecttests):
    """
    Test relative_to method.
    """

    def test_identical(self):
        """
        Test identical objects
        """

        # Create secondary object
        ext_object = obj.WorldObject()

        # Compute relative position_fcn
        rel_pos = self.worldobject.relative_to(ext_object, 0)

        # Check result
        self.assertIsInstance(rel_pos, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(rel_pos == 0), "Incorrect output value.")

    def test_different(self):
        """
        Test differing objects
        """

        # Create secondary object
        ext_object = obj.WorldObject()
        ext_object.set_position_preset("kinematic", initial_position=np.array([1, 1, 2]),          \
                                                               initial_velocity=np.array([0, 0, 0]))

        # Compute relative position_fcn
        rel_pos = ext_object.relative_to(self.worldobject, 0)
        rel_pos_flipped = self.worldobject.relative_to(ext_object, 0)

        # Check result
        self.assertIsInstance(rel_pos, np.ndarray, "Incorrect output type.")
        self.assertTrue(np.all(rel_pos == 1), "Incorrect output value.")
        self.assertTrue(np.all(rel_pos_flipped == -1), "Incorrect output value.")
