"""
StarMap unit tests.
"""
import unittest
from astraviso import starmap as sm
import numpy as np

class starmaptests(unittest.TestCase):
    """
    Starmap unit test class.
    """

    def setUp(self):
        self.starmap = sm.StarMap()

    def tearDown(self):
        self.assertEqual(self.starmap.size, len(self.starmap.catalog), "Internal catalog size      \
                                                         property not consistent with actual size.")
        del self.starmap

class test_selectrange(starmaptests):
    """
    Test selectrange method.
    """

    def test_none(self):
        """
        Test range with zero results.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select range outside of catalog
        self.starmap.selectrange(-8, 12)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 6)
        self.assertEqual(len(self.starmap.catalog), 6)

    def test_all(self):
        """
        Test range containing whole catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select range outside of catalog
        self.starmap.selectrange(13, 17)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 0)
        self.assertEqual(len(self.starmap.catalog), 0)

    def test_segment(self):
        """
        Test range containing a portion of the catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select range outside of catalog
        self.starmap.selectrange(4, 8)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 2)
        self.assertTrue(np.array_equal(self.starmap.catalog, np.array([[0, 0, -1], [0, 1, 0]])))

class test_selectdimmer(starmaptests):
    """
    Test selectdimmer method.
    """

    def test_none(self):
        """
        Test range with zero results.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select brighter than 12
        self.starmap.selectdimmer(12)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 0)
        self.assertEqual(len(self.starmap.catalog), 0)

    def test_all(self):
        """
        Test range containing whole catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select dimmer than -9
        self.starmap.selectdimmer(-9)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 6)
        self.assertEqual(len(self.starmap.catalog), 6)

    def test_bottomtwo(self):
        """
        Test range containing a portion of the catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select brighter than zero
        self.starmap.selectdimmer(4)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 2)
        self.assertTrue(np.array_equal(self.starmap.catalog, np.array([[0, 0, 1], [0, 0, -1]])))

class test_selectbrighter(starmaptests):
    """
    Test selectbrighter method.
    """

    def test_none(self):
        """
        Test range with zero results.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select brighter than -8
        self.starmap.selectbrighter(-8)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 0)
        self.assertEqual(len(self.starmap.catalog), 0)

    def test_all(self):
        """
        Test range containing whole catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select brighter than 13
        self.starmap.selectbrighter(13)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 6)
        self.assertEqual(len(self.starmap.catalog), 6)

    def test_uppertwo(self):
        """
        Test range containing a portion of the catalog.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Select brighter than zero
        self.starmap.selectbrighter(0)

        # Check remaining elements
        self.assertEqual(self.starmap.size, 2)
        self.assertTrue(np.array_equal(self.starmap.catalog, np.array([[1, 0, 0], [-1, 0, 0]])))

class test_get_region(starmaptests):
    """
    Test get_region method.
    """

    def test_dimensions(self):
        """
        Test output dimensions for several cases.
        """

        # Load six faces catalog
        self.starmap.load_preset("sixfaces")

        # Check single star along boresight
        region = self.starmap.get_region([0, 0, 1], 0)
        self.assertTrue(isinstance(region["catalog"], np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(region["magnitude"], np.ndarray), "Incorrect type.")
        self.assertEqual(len(region["catalog"]), 1, "Incorrect region extract.")
        self.assertEqual(len(region["magnitude"]), 1, "Incorrect region extract.")

        # Check 90 degree angle along y-axis
        region = self.starmap.get_region([0, 1, 0], 90)
        self.assertTrue(isinstance(region["catalog"], np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(region["magnitude"], np.ndarray), "Incorrect type.")
        self.assertEqual(len(region["catalog"]), 5, "Incorrect region extract.")
        self.assertEqual(len(region["magnitude"]), 5, "Incorrect region extract.")

        # Check 180 degree angle along negative x-axis
        region = self.starmap.get_region([-1, -1, -1], 55)
        self.assertTrue(isinstance(region["catalog"], np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(region["magnitude"], np.ndarray), "Incorrect type.")
        self.assertEqual(len(region["catalog"]), 3, "Incorrect region extract.")
        self.assertEqual(len(region["magnitude"]), 3, "Incorrect region extract.")

class test_load_preset(starmaptests):
    """
    Test load_preset method.
    """

    def test_singlecenter(self):
        """
        Test singlecenter preset.
        """

        # Set up catalog and check
        self.starmap.load_preset("singlecenter")
        self.assertEqual(len(self.starmap.catalog), 1, "Incorrect catalog length")
        self.assertEqual(len(self.starmap.magnitude), 1, "Incorrect magnitude length")
        self.assertEqual(self.starmap.size, 1)

        # Check vector norms
        norms_squared = np.array([np.sum(el**2) for el in self.starmap.catalog])
        self.assertTrue((norms_squared == 1).all())

        # Check types
        self.assertTrue(isinstance(self.starmap.catalog, np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(self.starmap.magnitude, np.ndarray), "Incorrect type.")

    def test_sixfaces(self):
        """
        Test sixfaces preset.
        """

        # Set up catalog and check
        self.starmap.load_preset("sixfaces")
        self.assertEqual(len(self.starmap.catalog), 6, "Incorrect catalog length")
        self.assertEqual(len(self.starmap.magnitude), 6, "Incorrect magnitude length")
        self.assertEqual(self.starmap.size, 6)

        # Check vector norms
        norms_squared = np.array([np.sum(el**2) for el in self.starmap.catalog])
        self.assertTrue((norms_squared == 1).all())

        # Check types
        self.assertTrue(isinstance(self.starmap.catalog, np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(self.starmap.magnitude, np.ndarray), "Incorrect type.")

    def test_random(self):
        """
        Test random preset.
        """

        # Set up catalog and check
        self.starmap.load_preset("random", 100)
        self.assertEqual(len(self.starmap.catalog), 100, "Incorrect catalog length")
        self.assertEqual(len(self.starmap.magnitude), 100, "Incorrect magnitude length")
        self.assertEqual(self.starmap.size, 100)

        # Check vector norms
        norms_squared = np.array([np.sum(el**2) for el in self.starmap.catalog])
        self.assertTrue(np.allclose(norms_squared, 1))

        # Check types
        self.assertTrue(isinstance(self.starmap.catalog, np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(self.starmap.magnitude, np.ndarray), "Incorrect type.")

    @unittest.skip("Test not required.")
    def test_hipparcos(self):
        """
        Test Hipparcos preset.
        """

        # Set up catalog and check
        self.starmap.load_preset("hipparcos")
        self.assertEqual(len(self.starmap.catalog), 117955, "Incorrect catalog length")
        self.assertEqual(len(self.starmap.magnitude), 117955, "Incorrect magnitude length")
        self.assertEqual(self.starmap.size, 117955)

        # Check vector norms
        norms_squared = np.array([np.sum(el**2) for el in self.starmap.catalog])
        self.assertTrue(np.allclose(norms_squared, 1))

        # Check types
        self.assertTrue(isinstance(self.starmap.catalog, np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(self.starmap.magnitude, np.ndarray), "Incorrect type.")

    @unittest.skip("Test not required.")
    def test_tycho(self):
        """
        Test Tycho preset.
        """

        # Set up catalog and check
        self.starmap.load_preset("tycho")
        self.assertEqual(len(self.starmap.catalog), 1055115, "Incorrect catalog length")
        self.assertEqual(len(self.starmap.magnitude), 1055115, "Incorrect magnitude length")
        self.assertEqual(self.starmap.size, 1055115)

        # Check vector norms
        norms_squared = np.array([np.sum(el**2) for el in self.starmap.catalog])
        self.assertTrue(np.allclose(norms_squared, 1))

        # Check types
        self.assertTrue(isinstance(self.starmap.catalog, np.ndarray), "Incorrect type.")
        self.assertTrue(isinstance(self.starmap.magnitude, np.ndarray), "Incorrect type.")
