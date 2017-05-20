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

class test_quaternion2dcm(pointingutilstests):
    """
    Test quaternion to DCM conversion function.
    """

    def test_types(self):
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

    def test_identity(self):
        """
        Test identical assignment.
        """

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

    def test_values(self):
        """
        Test values for a specific case.
        """

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
