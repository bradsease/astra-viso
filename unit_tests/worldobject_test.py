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
											  
class test_pointing_preset_kinematic(worldobjecttests):

	def test_output(self):
	
		# Set up inputs
		quaternion   = np.array([0,0,0,1])
		angular_rate = np.array([0,0,0])
		
		# Call method
		quaternion_deriv =                                                    \
		   self.worldobject.pointing_preset_kinematic(quaternion, angular_rate)
		   
		# Check dimensions and type
		self.assertEqual(len(quaternion_deriv), 4, 
												 "Incorrect output dimension.")
		self.assertTrue(type(quaternion_deriv) is np.ndarray, 
													  "Incorrect output type.")
													  
		# Check values
		self.assertTrue( np.array_equal(quaternion_deriv, np.array([0,0,0,0])), 
													 "Incorrect output value.")