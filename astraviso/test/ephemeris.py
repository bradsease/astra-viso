"""
Ephem module unit tests.
"""
from __future__ import division
import unittest
import numpy as np
import datetime as dt
import pkg_resources as pkg
from astraviso import ephemeris as ephem
from astraviso import mathutils


class EphemTests(unittest.TestCase):
    """
    Ephem module unit test class.
    """
    pass

class TestOrbitEphemerisClass(EphemTests):
    """
    Test top-level ephemeris.OrbitEphemeris class.
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

class TestAttitudeEphemerisClass(EphemTests):
    """
    Test top-level ephemeris.AttitudeEphemeris class.
    """

    def setUp(self):
        self.att_time_quat = pkg.resource_filename("astraviso",
            "test/data/AttitudeTimeQuaternions.a")
        self.att_time_quat_thinned = pkg.resource_filename("astraviso",
            "test/data/AttitudeTimeQuaternions_thinned.a")
        self.att_time_quat_ang_vels = pkg.resource_filename("astraviso",
            "test/data/AttitudeTimeQuatAngVels.a")

    def test_attitudetimequaternions_ephem(self):

        # Verify ephemeris load
        test_ephem = ephem.AttitudeEphemeris(self.att_time_quat)
        self.assertEqual(test_ephem._initial_epoch,
                         dt.datetime(2017, 10, 10, 16, 0, 0),
                         "Incorrect initial epoch.")
        self.assertEqual(test_ephem._coord_sys, "J2000",
                         "Incorrect coordinate system.")

        # Test interpolated solution
        test_result = test_ephem.get_attitude(0.5)
        np.testing.assert_almost_equal(np.sqrt(np.sum(test_result**2)), 1)
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertEqual(len(test_result), 4, "Incorrect output dimension.")

        # Verify AttitudeTimeQuaternions sample
        test_result = test_ephem.get_attitude(0)
        np.testing.assert_almost_equal(test_result,
            np.array([-3.61538786711038e-01, -6.07692026507889e-01,
                      3.61538863884271e-01, 6.07692156049029e-01]))
        test_result_dt = test_ephem.get_attitude(
            dt.datetime(2017, 10, 10, 16, 0, 0))
        np.testing.assert_almost_equal(test_result_dt, test_result)

        # Verify AttitudeTimeQuaternions at end of ephemeris
        test_result = test_ephem.get_attitude(
            dt.datetime(2017, 10, 10, 16, 0, 0) + dt.timedelta(seconds=600))
        np.testing.assert_almost_equal(test_result,
            np.array([-2.17009849997894e-01, -7.78157334236009e-01,
                      4.62955040362287e-01, 3.64760906323243e-01]))


    def test_attitudetimequatangvels_ephem(self):

        # Verify ephemeris load
        test_ephem = ephem.AttitudeEphemeris(self.att_time_quat_ang_vels)
        self.assertEqual(test_ephem._initial_epoch,
                         dt.datetime(2017, 10, 10, 16, 0, 0),
                         "Incorrect initial epoch.")
        self.assertEqual(test_ephem._coord_sys, "J2000",
                         "Incorrect coordinate system.")

        # Test interpolated solution
        test_result = test_ephem.get_attitude(0.5)
        np.testing.assert_almost_equal(np.sqrt(np.sum(test_result**2)), 1)
        self.assertIsInstance(test_result, np.ndarray, "Incorrect output type.")
        self.assertEqual(len(test_result), 4, "Incorrect output dimension.")

        # Verify AttitudeTimeQuaternions sample
        test_result = test_ephem.get_attitude(0)
        np.testing.assert_almost_equal(test_result,
            np.array([-3.61538786711038e-01, -6.07692026507889e-01,
                      3.61538863884271e-01, 6.07692156049029e-01]))
        test_result_dt = test_ephem.get_attitude(
            dt.datetime(2017, 10, 10, 16, 0, 0))
        np.testing.assert_almost_equal(test_result_dt, test_result)

        # Verify AttitudeTimeQuaternions at end of ephemeris
        test_result = test_ephem.get_attitude(
            dt.datetime(2017, 10, 10, 16, 0, 0) + dt.timedelta(seconds=600))
        np.testing.assert_almost_equal(test_result,
            np.array([-2.17009849997894e-01, -7.78157334236009e-01,
                      4.62955040362287e-01, 3.64760906323243e-01]))


    def test_interpolation_accuracy(self):

        # Load ephemerides
        primary_ephem = ephem.AttitudeEphemeris(self.att_time_quat)
        thinned_ephem = ephem.AttitudeEphemeris(self.att_time_quat_thinned)

        # Test interpolated solution
        for time in np.arange(30, step=0.2):
            np.testing.assert_almost_equal(primary_ephem.get_attitude(time),
                                           thinned_ephem.get_attitude(time))
