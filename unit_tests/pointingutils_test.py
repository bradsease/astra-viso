"""
Pointingutils unit tests.
"""
import unittest
from astraviso import pointingutils as point
import numpy as np

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
