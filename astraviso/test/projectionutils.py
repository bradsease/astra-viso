"""
projectionutils unit tests.
"""
from __future__ import division
import unittest
import numpy as np
from astraviso import projectionutils

class imageutilstests(unittest.TestCase):
    """
    Imageutils unit test class.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

class test_pinhole_project(imageutilstests):
    """
    Test pinhole_project function.
    """

    def test_single_pinhole(self):
        """
        Test single point projection.
        """

        # Build input vector
        vector = np.array([0, 0, 1])
        
        #  Convert
        img_x, img_y = projectionutils.pinhole_project(vector, 93, 0.016, 1024)

        # Check result
        self.assertIsInstance(img_x, np.ndarray, "X coordinate output should be ndarray.")
        self.assertIsInstance(img_y, np.ndarray, "X coordinate output should be ndarray.")
        self.assertEqual(len(img_x), 1, "Output dimension should equal input dimension")
        self.assertEqual(len(img_x), len(img_y), "Number of x and y coordinates should be equal.")
        self.assertEqual(img_x[0], img_y[0], "For this case, coordinates should be equal.")
        self.assertEqual(img_x[0], 512.5, "For this case, coordinate value should be 512.5.")

    def test_multiple_pinhole(self):
        """
        Test multiple point projection.
        """

        # Build input vector
        vector = np.array([[0, 0, 1], [0, 0, -1]])

        #  Convert
        img_x, img_y = projectionutils.pinhole_project(vector, 93, 0.016, 1024)

        # Check result
        self.assertIsInstance(img_x, np.ndarray, "X coordinate output should be ndarray.")
        self.assertIsInstance(img_y, np.ndarray, "X coordinate output should be ndarray.")
        self.assertEqual(len(img_x), 2, "Output dimension should equal input dimension")
        self.assertEqual(len(img_x), len(img_y), "Number of x and y coordinates should be equal.")
        self.assertTrue(all(img_x == img_y), "For this case, coordinates should be equal.")
