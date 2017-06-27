"""
Positionutils unit tests.
"""
from __future__ import division
import unittest
import numpy as np
from astraviso import positionutils as pos

class positionutilstests(unittest.TestCase):
    """
    positionutils unit test class.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

class test_light_time(positionutilstests):
    """
    Test light_time function.
    """

    def test_light_time_static(self):
        """
        Test light_time function for static objects.
        """

        # Set up scenario
        target = lambda time: 4.2e7*np.array([1, 0, 0])
        observer = lambda time: np.zeros(3)
        expected_result = 4.2e7 / 299792458

        # Check results
        result = pos.light_time(target, observer, 0)
        self.assertIsInstance(result, float, 'Output must be float.')
        self.assertGreater(result, 0, 'Light travel time must be greater than zero.')
        self.assertEqual(result, expected_result, 'Incorrect result.')

    def test_light_time_dynamic(self):
        """
        Test light_time function for dynamic objects.
        """

        # Set up scenario
        target = lambda time: 4.2e7*np.array([1, 0, 0]) + 3070*time*np.array([0, 1, 0])
        observer = lambda time: np.zeros(3) - 460*np.array([0, 1, 0])

        # Check results
        result = pos.light_time(target, observer, 0)
        self.assertIsInstance(result, float, 'Output must be float.')
        self.assertGreater(result, 0, 'Light travel time must be greater than zero.')
