import unittest
import copy
import worldobject as obj
import numpy as np

class worldobjecttests(unittest.TestCase):
    
    def setUp(self):
        self.worldobject = obj.worldobject()
        
    def tearDown(self):
        del self.worldobject

class test_init(worldobjecttests):

	def test_types(self):
		
		# Check types
		self.assertTrue( type(self.worldobject.quaternion) is np.ndarray,
											 "Incorrect quaternion data type.")
		self.assertTrue( type(self.worldobject.angular_rate) is np.ndarray,
										  "Incorrect angular rate data type." )
		self.assertTrue( type(self.worldobject.position) is np.ndarray,
											  "Incorrect position data type." )
		self.assertTrue( type(self.worldobject.velocity) is np.ndarray,
											  "Incorrect velocity data type." )