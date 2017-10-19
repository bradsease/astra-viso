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

class test_vector_functions(mathutilstests):
    """
    Tests for vector tools in math module.
    """

    def test_unit(self):
        """
        Test unit function.
        """
        np.random.seed(1)
        for idx in range(10):
            unit_vector = math.unit(np.random.randn(3))
            self.assertTrue(np.isclose(np.linalg.norm(unit_vector), 1),
                            "Magnitude of unit vector should be 1.")

    def test_angle(self):
        """
        Test angle function.
        """
        vector1 = np.array([0, 0, 1])
        vector2 = np.array([1, 0, 0])
        np.testing.assert_almost_equal(math.angle(vector1, vector1), 0)
        np.testing.assert_almost_equal(math.angle(vector1, vector2), np.pi/2)
        np.testing.assert_almost_equal(math.angle(vector2, vector2), 0)

class test_lagrange_interpolating_polynomial(mathutilstests):
    """
    Test lagrange_interpolating_polynomial function.
    """

    def test_basic(self):
        """
        Test...
        """

        t = np.arange(7)
        for order in np.arange(1, 6):
            y = t ** order
            for test_val in np.arange(7, step=0.25):
                result = math.lagrange_interpolating_polynomial(t, y, test_val)
                np.testing.assert_almost_equal(result, test_val**order)

class test_MovingWindowInterpolator(mathutilstests):
    """
    Test MovingWindowInterpolator class.
    """

    def test_lagrange(self):
        """
        Test lagrange MovingWindowInterpolator.
        """

        # Test over range of polynomial degrees
        for order in np.arange(1, 5):

            t = np.arange(-20, 20)
            fcn = lambda time: time ** order

            mwi = math.MovingWindowInterpolator(t, fcn(t),
                math.build_lagrange_interpolator)

            for test_val in np.arange(-20, 20, step=0.25):
                result = mwi(test_val)
                np.testing.assert_almost_equal(result, fcn(test_val))

        # Test for LEO orbit-like case (mm level accuracy)
        t = np.arange(5400, step=20)
        fcn = lambda time: 6600000 * np.cos(2*np.pi*time / 5400)

        mwi = math.MovingWindowInterpolator(t, fcn(t),
            math.build_lagrange_interpolator, window_size=6)

        for test_val in np.arange(0, 5400, step=30):
            result = mwi(test_val)
            np.testing.assert_almost_equal(result, fcn(test_val), decimal=3)

        # Test for GEO orbit-like case (mm level accuracy)
        t = np.arange(86400, step=60)
        fcn = lambda time: 42000000 * np.cos(2*np.pi*time / 86400)

        mwi = math.MovingWindowInterpolator(t, fcn(t),
            math.build_lagrange_interpolator, window_size=6)

        for test_val in np.arange(0, 86400, step=70):
            result = mwi(test_val)
            np.testing.assert_almost_equal(result, fcn(test_val), decimal=3)
