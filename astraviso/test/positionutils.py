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

class test_orbit_functions(positionutilstests):
    """
    Test orbit generation functions.
    """

    def test_earth_orbit_ode(self):
        """
        Test earth_orbit_ode function.
        """

        # Choose test orbit
        elem = 1e3*np.array([6778, 0, 0, 0, 7.660477, 0])

        # Test without J2
        result = pos.earth_orbit_ode(elem)
        self.assertIsInstance(result, np.ndarray,
                              'Output type must be ndarray.')
        self.assertEqual(len(result), 6, 'Output must have 6 elements.')
        self.assertTrue(np.all(result[:3] == elem[-3:]),
                        'Position acceleration must be equal to velocity.')

        # Test with J2
        result = pos.earth_orbit_ode(elem, nonspherical="on")
        self.assertIsInstance(result, np.ndarray,
                              'Output type must be ndarray.')
        self.assertEqual(len(result), 6, 'Output must have 6 elements.')
        self.assertTrue(np.all(result[:3] == elem[-3:]),
                        'Position acceleration must be equal to velocity.')

    def test_earth_orbit(self):
        """
        Test earth_orbit function.
        """

        # Define helper function
        def energy_calc(state):
            return np.linalg.norm(state[-3:])**2/2 -                           \
                                  pos.EARTH_GRAV_PARAM/np.linalg.norm(state[:3])

        # Choose test orbit #1
        elem = 1e3*np.array([6778, 0, 0, 0, 7.660477, 0])
        energy = energy_calc(elem)

        # Test with no options
        fcn = pos.earth_orbit(elem)
        self.assertIsInstance(fcn(0), np.ndarray,
                              'Output type must be ndarray.')
        for time in range(0, 10000, 1000):
            energy_at_time = energy_calc(fcn(time))
            self.assertTrue(np.allclose(energy, energy_at_time),
                                        'Energy must be conserved.')

        # Test with J2
        fcn = pos.earth_orbit(elem, nonspherical="on")
        self.assertIsInstance(fcn(0), np.ndarray,
                              'Output type must be ndarray.')

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
        self.assertGreater(result, 0,
                           'Light travel time must be greater than zero.')
        self.assertEqual(result, expected_result, 'Incorrect result.')

    def test_light_time_dynamic(self):
        """
        Test light_time function for dynamic objects.
        """

        # Set up scenario
        target = lambda time: 4.2e7*np.array([1, 0, 0])     \
                              + 3070*time*np.array([0, 1, 0])
        observer = lambda time: np.zeros(3) - 460*np.array([0, 1, 0])

        # Check results
        result = pos.light_time(target, observer, 0)
        self.assertIsInstance(result, float, 'Output must be float.')
        self.assertGreater(result, 0,
                           'Light travel time must be greater than zero.')
