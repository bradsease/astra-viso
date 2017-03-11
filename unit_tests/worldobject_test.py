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
                                                     
class test_set_pointing_fcn(worldobjecttests):

    def test_ode(self):
    
        # Create "state" function
        f = lambda t,state : [0,0,0,0,0,0,0]
        
        # Set function
        self.worldobject.set_pointing_fcn(f, "ode")
        
        # Check properties
        self.assertEqual(self.worldobject.model_pointing, "on", 
                                                  "Incorrect modeling option.")
        self.assertEqual(self.worldobject.pointing_mode, "ode", 
                                                    "Incorrect pointing mode.")
                                                
        # Check function output
        test_result     = self.worldobject.pointing_fcn(1)
        expected_result = np.hstack((self.worldobject.quaternion,
                                                self.worldobject.angular_rate))
        self.assertTrue(np.all(test_result == expected_result),
                                                  "Incorrect function result.")
                                                  
    def test_explicit(self):
    
        # Create "state" function
        f = lambda t : np.hstack((self.worldobject.quaternion,
                                                self.worldobject.angular_rate))
        
        # Set function
        self.worldobject.set_pointing_fcn(f, "explicit")
        
        # Check properties
        self.assertEqual(self.worldobject.model_pointing, "on", 
                                                  "Incorrect modeling option.")
        self.assertEqual(self.worldobject.pointing_mode, "explicit", 
                                                    "Incorrect pointing mode.")
                                                
        # Check function output
        test_result     = self.worldobject.pointing_fcn(1)
        expected_result = np.hstack((self.worldobject.quaternion,
                                                self.worldobject.angular_rate))
        self.assertTrue(np.all(test_result == expected_result),
                                                  "Incorrect function result.")

class test_set_integrator(worldobjecttests):

    def test_set(self):
        
        # Set integrator
        self.worldobject.set_integrator("vode", 1e-8, 1e-9)
        
        # Check properties
        self.assertEqual(self.worldobject.integrator, "vode", 
                                                       "Incorrect integrator.")
        self.assertEqual(self.worldobject.integrator_atol, 1e-8, 
                                               "Incorrect absolute tolerance.")
        self.assertEqual(self.worldobject.integrator_rtol, 1e-9, 
                                               "Incorrect relative tolerance.")
        
    def test_set_with_pointing_ode(self):
    
        # Create "state" function
        f = lambda t,state : [0,0,0,0,0,0,0]
        
        # Set function
        self.worldobject.set_pointing_fcn(f, "ode")
        
        # Set integrator
        self.worldobject.set_integrator("vode", 1e-8, 1e-9)
        
        
        