"""
Ephem module unit tests.
"""
from __future__ import division
import unittest
import numpy as np
import datetime as dt
import pkg_resources as pkg
from astraviso import ephemeris as ephem


class EphemTests(unittest.TestCase):
    """
    Ephem module unit test class.
    """
    pass

class TestEphemerisClass(EphemTests):
    """
    Test top-level ephem.Ephemeris class.
    """

    def test_sample_ephem(self):
        """
        Test sample ephemeris.
        """

        # Verify ephemeris load
        sample_ephem_path = pkg.resource_filename("astraviso",
                                                  "test/data/test_sat.e")
        test_ephem = ephem.OrbitEphemeris(sample_ephem_path)
        self.assertEqual(test_ephem._initial_epoch,
                         dt.datetime(2017, 6, 24, 16, 0, 0),
                         "Incorrect initial epoch.")
        self.assertEqual(test_ephem._central_body, "Earth",
                         "Incorrect central body.")
        self.assertEqual(test_ephem._coord_sys, "J2000",
                         "Incorrect coordinate system.")

        # Verify position sample
        test_result = test_ephem.get_position(10)
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertEqual(len(test_result), 3, "Incorrect output dimension.")

        # Verify position sample by datetime
        test_result_dt = test_ephem.get_position(
            dt.datetime(2017, 6, 24, 16, 0, 0) + dt.timedelta(seconds=10))
        np.testing.assert_almost_equal(test_result_dt, test_result)

        # Verify exact results at supplied times
        with open('astraviso/test/data/test_sat.e', 'r') as ephem_file_handle:
            ephem_file_contents = ephem_file_handle.read().split('\n')
        state_data = ephem.read_stk_orbit_state_data(ephem_file_contents,
            "EphemerisTimePosVel")
        for state in state_data.T:
            interp_solution = test_ephem.get_position(state[0])
            np.testing.assert_almost_equal(state[1:4], interp_solution)

        # Verify accuracy between nominal (60 s) and thinned ephemeris (120 s).
        # Currently accepting mm level accuracy.
        thin_ephem_path = pkg.resource_filename("astraviso",
                                                "test/data/test_sat_thinned.e")
        thin_ephem = ephem.OrbitEphemeris(thin_ephem_path)
        for time in np.arange(0.0, 82560.0, 60.0):
            diff = np.linalg.norm(test_ephem.get_position(time) - \
                                  thin_ephem.get_position(time))
            np.testing.assert_almost_equal(diff, 0, decimal=3)
