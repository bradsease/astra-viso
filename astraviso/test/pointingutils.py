"""
Pointingutils unit tests.
"""
from __future__ import division
import unittest
import numpy as np
from scipy.integrate import ode
from astraviso import pointingutils as point

class pointingutilstests(unittest.TestCase):
    """
    Pointingutils unit test class.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

class test_rigid_body_kinematics(pointingutilstests):
    """
    Test rigid_body_kinematics function.
    """

    def test_output(self):
        """
        Test output value and type.
        """

		# Set up inputs
        quaternion = np.array([0, 0, 0, 1])
        angular_rate = np.array([0, 0, 0])

        # Call method
        deriv = point.rigid_body_kinematic(quaternion, angular_rate)

        # Check dimensions and type
        self.assertEqual(len(deriv), 7, "Incorrect output dimension.")
        self.assertTrue(isinstance(deriv, np.ndarray), "Incorrect output type.")

        # Check values
        self.assertTrue(np.array_equal(deriv, np.array([0, 0, 0, 0, 0, 0, 0])),                    \
                                                                          "Incorrect output value.")

    def test_closure(self):
        """
        Test closure of quaternion after two rotations.
        """

        # Iterate through all three dimensions
        for dim in range(3):

            # Set up inputs
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            angular_rate = np.array([0.0, 0.0, 0.0])
            angular_rate[dim] = 2.0 * np.pi

            # Build function
            fcn = lambda t, state: point.rigid_body_kinematic(state[0:4], state[4:])

            # Set up integrator
            integ = ode(fcn)
            integ.set_integrator("dopri5", atol=1e-10, rtol=1e-10)
            integ.set_initial_value(np.hstack((quaternion, angular_rate)))
            result = integ.integrate(2)

            # Check result
            self.assertTrue(np.allclose(result, np.hstack((quaternion, angular_rate))), \
                                             "Quaternion must return after 2 rotations.")

class test_quat_conversion(pointingutilstests):
    """
    Test conversions between quaternions and DCMs.
    """

    def test_quat2dcm(self):
        """
        Test input / output types.
        """

        # Set up scenarios
        quaternion = np.array([0, 0, 0, 1])
        quaternion_list = [quaternion, quaternion, quaternion]

        # Call function
        dcm_single = point.quaternion2dcm(quaternion)
        dcm_multi = point.quaternion2dcm(quaternion_list)

        # Check types
        self.assertEqual(type(dcm_single), np.ndarray, "Incorrect output type.")
        self.assertEqual(type(dcm_multi), list, "Incorrect output type.")


        # Set up scenarios
        quaternion = np.array([0, 0, 0, 1])
        quaternion_list = [quaternion, quaternion, quaternion]

        # Call function
        dcm_single = point.quaternion2dcm(quaternion)
        dcm_multi = point.quaternion2dcm(quaternion_list)

        # Check result
        self.assertTrue((dcm_single == np.eye(3)).all(), "Incorrect identity result.")
        for dcm in dcm_multi:
            self.assertTrue((dcm == np.eye(3)).all(), "Incorrect identity result.")


        # Set up scenario
        quaternion = np.array([0.5, 0.5, 0.5, 0.5])

        # Call function
        dcm_single = point.quaternion2dcm(quaternion)

        # Set up expected result
        result = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        # Check result
        self.assertTrue((dcm_single == result).all(), "Incorrect value result.")
        for row in dcm_single:
            self.assertTrue(np.isclose(np.linalg.norm(row), 1), "Incorrect dcm row magnitude.")

    def test_quat_dcm_closure(self):
        """
        Test closure in quat > dcm > quat > dcm conversions.
        """

        # Set up identity scenario
        dcm = np.eye(3)

        # Compute closures
        quat = point.dcm2quaternion(dcm)
        dcm2 = point.quaternion2dcm(quat)
        quat2 = point.dcm2quaternion(dcm2)

        # Check values
        self.assertTrue(np.allclose(dcm, dcm2), "Closure failed: dcm > quat > dcm.")
        self.assertTrue(np.allclose(quat, quat2), "Closure failed: dcm > quat > dcm > quat.")

        # Test random
        np.random.seed(1)
        for idx in range(50):
            angles = 2*np.pi*np.random.rand(3)
            dcm = np.dot(np.dot(point.rot1(angles[0]), point.rot2(angles[1])), \
                                                          point.rot3(angles[2]))
            quat = point.dcm2quaternion(dcm)
            dcm2 = point.quaternion2dcm(quat)
            self.assertTrue(np.allclose(dcm, dcm2), "Closure failed: dcm > quat > dcm.")

class test_quaternion_functions(pointingutilstests):
    """
    Test quaternion-related functions.
    """

    def test_qmultiply(self):
        """
        Test qmultiply function.
        """

        # Test multiplication by identity
        quat1 = np.array([0, 0, 0, 1])
        quat2 = np.array([0, np.cos(np.pi/4), 0, np.cos(np.pi/4)])
        result = point.qmultiply(quat1, quat2)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, quat2), "Incorrect output.")

        # Test reversed multiplication by identity
        quat1 = np.array([0, 0, 0, 1])
        quat2 = np.array([0, np.cos(np.pi/4), 0, np.cos(np.pi/4)])
        result = point.qmultiply(quat2, quat1)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, quat2), "Incorrect output.")

        # Test multiplication by self
        quat1 = np.array([0.5, 0.5, 0.5, 0.5])
        expected_result = np.array([0.5, 0.5, 0.5, -0.5])
        result = point.qmultiply(quat1, quat1)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, expected_result), "Incorrect output.")

    def test_qinv(self):
        """
        Test qinv function.
        """

        # Define identity quaternion
        identity = np.array([0, 0, 0, 1])

        # Test trivial case
        quat = identity
        result = point.qinv(quat)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, identity), "Incorrect output.")

        # Test with qmultiply
        quat = np.array([0.5, 0.5, 0.5, 0.5])
        result = point.qinv(quat)
        result = point.qmultiply(quat, result)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, identity), "Incorrect output.")

    def test_qrotate(self):
        """
        Test qrotate function.
        """

        # Define identity quaternion
        identity = np.array([0, 0, 0, 1])

        # Test rotation of identity
        quat = np.array([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])
        result = point.qrotate(identity, quat)
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, quat), "Incorrect output.")

        # Test combination of rotations
        quat = np.array([0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])
        result = point.qrotate(quat, quat)
        expected_result = np.array([0, np.sin(np.pi/2), 0, np.cos(np.pi/2)])
        self.assertIsInstance(result, np.ndarray, "Output type must be ndarray.")
        self.assertEqual(len(result), 4, "Output must have 4 elements.")
        self.assertTrue(np.allclose(result, quat), "Incorrect output.")

class test_dcm_functions(pointingutilstests):
    """
    
    """

    def check_rotation_matrix(self, R):
        """
        Rotation matrix verification function.
        """

        # Check dimensions
        self.assertEqual(R.shape[0], R.shape[1], "Rotation matrix must be square.")
        self.assertEqual(R.shape[0], 3, "Rotation matrix must be 3x3.")

        # Check determinant is one
        self.assertTrue(np.isclose(np.linalg.det(R), 1.0), 
                                         "Determinant of a rotation matrix must be one.")

        # Check column norms
        for idx in range(len(R)):
            self.assertTrue(np.isclose(np.linalg.norm(R[idx,:]), 1.0), 
                                            "Rotation matrix rows must have unity norm.")

    def test_x_rotation(self):
        """
        Test rot1 function for positive, negative, and zero rotations.
        """

        # Test zero rotation
        dcm = point.rot1(0)
        self.check_rotation_matrix(dcm)

        # Test positive rotation
        dcm = point.rot1(np.pi/3)
        self.check_rotation_matrix(dcm)

        # Test negative rotation
        dcm = point.rot1(-np.pi/3)
        self.check_rotation_matrix(dcm)

    def test_y_rotation(self):
        """
        Test rot2 function for positive, negative, and zero rotations.
        """

        # Test zero rotation
        dcm = point.rot2(0)
        self.check_rotation_matrix(dcm)

        # Test positive rotation
        dcm = point.rot2(np.pi/3)
        self.check_rotation_matrix(dcm)

        # Test negative rotation
        dcm = point.rot2(-np.pi/3)
        self.check_rotation_matrix(dcm)

    def test_z_rotation(self):
        """
        Test rot3 function for positive, negative, and zero rotations.
        """

        # Test zero rotation
        dcm = point.rot3(0)
        self.check_rotation_matrix(dcm)

        # Test positive rotation
        dcm = point.rot3(np.pi/3)
        self.check_rotation_matrix(dcm)

        # Test negative rotation
        dcm = point.rot3(-np.pi/3)
        self.check_rotation_matrix(dcm)

class test_radec_functions(pointingutilstests):
    """
    Test right ascension & declination functions.
    """

    def test_vector_to_ra_dec(self):
        """
        Test vector_to_ra_dec function.

        TODO: Tests for d_ra and d_dec.
        """

        # Set up test cases
        vector = np.array([0, 0, 1])
        vector_rate = np.array([0, 0.1, 1])

        # Test vector-only mode
        ra, dec = point.vector_to_ra_dec(vector)
        self.assertEqual(ra, 0, "Right ascension should be zero.")
        self.assertTrue(np.isclose(dec, 90), "Declination should be 90.")

        # Test vector & rate mode
        ra, dec, d_ra, d_dec = point.vector_to_ra_dec(vector, vector_rate)
        self.assertEqual(ra, 90, "Right ascension should be 90.")
        self.assertTrue(np.isclose(dec, 90), "Declination should be 90.")

        # Test vector-only mode with radian output
        ra, dec = point.vector_to_ra_dec(vector, output="rad")
        self.assertEqual(ra, 0, "Right ascension should be zero.")
        self.assertTrue(np.isclose(dec, np.pi/2), "Declination should be pi/2.")

        # Test vector & rate mode with radian output
        ra, dec, d_ra, d_dec = point.vector_to_ra_dec(vector, 
                                                               vector_rate, output="rad")
        self.assertEqual(ra, np.pi/2, "Right ascension should be pi/2.")
        self.assertTrue(np.isclose(dec, np.pi/2), "Declination should be pi/2.")

        # Set up test cases
        vector = np.array([1, 0, 0])
        
        # Test vector-only mode with radian output
        ra, dec = point.vector_to_ra_dec(vector, output="rad")
        self.assertEqual(ra, 0, "Right ascension should be zero.")
        self.assertTrue(np.isclose(dec, 0), "Declination should be 0.")
