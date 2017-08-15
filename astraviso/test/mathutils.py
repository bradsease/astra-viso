"""
Mathutils unit tests.
"""
from __future__ import division
import unittest
import numpy as np
from astraviso import mathutils as math

class mathutilstests(unittest.TestCase):
    """
    Mathutils unit test class.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

class test_dot_sequence(mathutilstests):
    """
    Test dot_sequence function.
    """

    def test_types(self):
        """
        Test several types of dot products.
        """

        # Test scalars
        result = math.dot_sequence(1, 2, 3, 4)
        self.assertEqual(result, 24, "Incorrect output value for scalar ints.")
        result = math.dot_sequence(1., 2., 3., 4.)
        self.assertEqual(result, 24, "Incorrect output value for floats.")

        # Test tuple/scalar mixture
        result = math.dot_sequence((1, 1), (1, 1), 2)
        self.assertEqual(result, 4, "Incorrect output value for mixed input.")

        # Test identity matrices
        result = math.dot_sequence(np.eye(3), np.eye(3), np.eye(3))
        self.assertTrue(np.array_equal(result, np.eye(3)),
                        "Incorrect output value for identity matrix input.")

        # Test 90 deg rotations
        dcm = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        result = math.dot_sequence(dcm, dcm, dcm, dcm)
        self.assertTrue(np.array_equal(result, np.eye(3)),
                        "Incorrect output value for 90 deg rotation input.")
