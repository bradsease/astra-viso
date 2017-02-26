import unittest
import copy
import starmap as map
import numpy as np

class starmaptests(unittest.TestCase):
    
    def setUp(self):
        self.starmap = map.starmap()
        
    def tearDown(self):
        del self.starmap

class test_selectrange(starmaptests):

	def test_none(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select range outside of catalog
		self.starmap.selectrange(-8, 12)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 6 )
		self.assertEqual( len(self.starmap.catalog), 6 )
		
	def test_all(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select range outside of catalog
		self.starmap.selectrange(13, 17)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 0 )
		self.assertEqual( len(self.starmap.catalog), 0 )
		
	def test_segment(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select range outside of catalog
		self.starmap.selectrange(4, 8)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 2 )
		self.assertTrue( np.array_equal(self.starmap.catalog,
												np.array([[0,0,-1],[0,1,0]])) )
		
class test_selectdimmer(starmaptests):
	
	def test_none(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select brighter than 12
		self.starmap.selectdimmer(12)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 0 )
		self.assertEqual( len(self.starmap.catalog), 0 )
		
	def test_all(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select dimmer than -9
		self.starmap.selectdimmer(-9)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 6 )
		self.assertEqual( len(self.starmap.catalog), 6 )
	
	def test_bottomtwo(self):
		
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select brighter than zero
		self.starmap.selectdimmer(4)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 2 )
		self.assertTrue( np.array_equal(self.starmap.catalog,
												np.array([[0,0,1],[0,0,-1]])) )
		
class test_selectbrighter(starmaptests):
	
	def test_none(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select brighter than -8
		self.starmap.selectbrighter(-8)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 0 )
		self.assertEqual( len(self.starmap.catalog), 0 )
		
	def test_all(self):
	
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select brighter than 13
		self.starmap.selectbrighter(13)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 6 )
		self.assertEqual( len(self.starmap.catalog), 6 )
	
	def test_uppertwo(self):
		
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Select brighter than zero
		self.starmap.selectbrighter(0)
		
		# Check remaining elements
		self.assertEqual( self.starmap.size, 2 )
		self.assertTrue( np.array_equal(self.starmap.catalog,
												np.array([[1,0,0],[-1,0,0]])) )
		
class test_getregion(starmaptests):
	
	def test_dimensions(self):
		
		# Load six faces catalog
		self.starmap.loadpreset("sixfaces")
		
		# Check single star along boresight
		region = self.starmap.getregion([0,0,1], 0)
		self.assertTrue( type(region["catalog"]) is np.ndarray )
		self.assertTrue( type(region["magnitude"]) is np.ndarray )
		self.assertEqual( len(region["catalog"]), 1, 
												  "Incorrect region extract." )
		self.assertEqual(len(region["magnitude"]), 1,
												  "Incorrect region extract." )
												  
		# Check 90 degree angle along y-axis
		region = self.starmap.getregion([0,1,0], 90)
		self.assertTrue( type(region["catalog"]) is np.ndarray )
		self.assertTrue( type(region["magnitude"]) is np.ndarray )
		self.assertEqual( len(region["catalog"]), 5, 
												  "Incorrect region extract." )
		self.assertEqual(len(region["magnitude"]), 5,
												  "Incorrect region extract." )
												  
		# Check 180 degree angle along negative x-axis
		region = self.starmap.getregion([-1,-1,-1], 55)
		self.assertTrue( type(region["catalog"]) is np.ndarray )
		self.assertTrue( type(region["magnitude"]) is np.ndarray )
		self.assertEqual( len(region["catalog"]), 3, 
												  "Incorrect region extract." )
		self.assertEqual(len(region["magnitude"]), 3,
												  "Incorrect region extract." )

class test_loadpreset(starmaptests):
	
	def test_singlecenter(self):
		
		# Set up catalog and check
		self.starmap.loadpreset("singlecenter")
		self.assertEqual( len(self.starmap.catalog), 1,
											        "Incorrect catalog length")
		self.assertEqual( len(self.starmap.magnitude), 1,
											      "Incorrect magnitude length")
		self.assertEqual( self.starmap.size, 1 )
		
		# Check vector norms
		norms_squared = np.array([ np.sum(el**2) for el in self.starmap.catalog])
		self.assertTrue( (norms_squared == 1).all() )
		
		# Check types
		self.assertTrue( type(self.starmap.catalog) is np.ndarray )
		self.assertTrue( type(self.starmap.magnitude) is np.ndarray )
		
	def test_sixfaces(self):
		
		# Set up catalog and check
		self.starmap.loadpreset("sixfaces")
		self.assertEqual( len(self.starmap.catalog), 6,
											        "Incorrect catalog length")
		self.assertEqual( len(self.starmap.magnitude), 6,
											      "Incorrect magnitude length")
		self.assertEqual( self.starmap.size, 6 )
		
		# Check vector norms
		norms_squared = np.array([ np.sum(el**2) for el in self.starmap.catalog])
		self.assertTrue( (norms_squared == 1).all() )
		
		# Check types
		self.assertTrue( type(self.starmap.catalog) is np.ndarray )
		self.assertTrue( type(self.starmap.magnitude) is np.ndarray )
		
	def test_random(self):
		
		# Set up catalog and check
		self.starmap.loadpreset("random", 100)
		self.assertEqual( len(self.starmap.catalog), 100,
											        "Incorrect catalog length")
		self.assertEqual( len(self.starmap.magnitude), 100,
											      "Incorrect magnitude length")
		self.assertEqual( self.starmap.size, 100 )
		
		# Check vector norms
		norms_squared = np.array([ np.sum(el**2) for el in self.starmap.catalog])
		self.assertTrue( np.allclose(norms_squared, 1) )
		
		# Check types
		self.assertTrue( type(self.starmap.catalog) is np.ndarray )
		self.assertTrue( type(self.starmap.magnitude) is np.ndarray )
	
	@unittest.skip("Test not required.")
	def test_hipparcos(self):
		
		# Set up catalog and check
		self.starmap.loadpreset("hipparcos")
		self.assertEqual( len(self.starmap.catalog), 117955,
											        "Incorrect catalog length")
		self.assertEqual( len(self.starmap.magnitude), 117955,
											      "Incorrect magnitude length")
		self.assertEqual( self.starmap.size, 117955 )
		
		# Check vector norms
		norms_squared = np.array([ np.sum(el**2) for el in self.starmap.catalog])
		self.assertTrue( np.allclose(norms_squared, 1) )
		
		# Check types
		self.assertTrue( type(self.starmap.catalog) is np.ndarray )
		self.assertTrue( type(self.starmap.magnitude) is np.ndarray )
		
	@unittest.skip("Test not required.")
	def test_tycho(self):
		
		# Set up catalog and check
		self.starmap.loadpreset("tycho")
		self.assertEqual( len(self.starmap.catalog), 1055115,
											        "Incorrect catalog length")
		self.assertEqual( len(self.starmap.magnitude), 1055115,
											      "Incorrect magnitude length")
		self.assertEqual( self.starmap.size, 1055115 )
		
		# Check vector norms
		norms_squared = np.array([ np.sum(el**2) for el in self.starmap.catalog])
		self.assertTrue( np.allclose(norms_squared, 1) )
		
		# Check types
		self.assertTrue( type(self.starmap.catalog) is np.ndarray )
		self.assertTrue( type(self.starmap.magnitude) is np.ndarray )