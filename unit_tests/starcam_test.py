import unittest
import copy
import starcam as cam
import numpy as np

class starcamtests(unittest.TestCase):
    
    def setUp(self):
        self.starcam = cam.StarCam()
        
    def tearDown(self):
        del self.starcam
        
class default_tests(starcamtests):
    def test_focal_length(self):
        self.assertEqual( self.starcam.focal_len, 93, 
                                            "Default focal length incorrect." )

class test_body2plane(starcamtests):
	
	def test_single_pinhole(self):
		
		#  Convert vector and check dimensions
		output = self.starcam.body2plane([0,0,1])
		self.assertEqual( len(output), 2, 
		                                   "Incorrect coordinate dimensions." )
		self.assertEqual( len(output[0]), 1, 
		                                    "Incorrect number of dimensions." )
		self.assertEqual( len(output[0]), len(output[1]),
										   "Mis-matched x and y coordinates." )
		self.assertEqual( output[0], output[1], "Incorrect projection." )
		self.assertEqual( output[0], 512.5, "Incorrect projection." )
		
	def test_multiple_pinhole(self):
		
		#  Convert vector and check dimensions
		output = self.starcam.body2plane([[0,0,1],[0,0,-1]])
		self.assertEqual( len(output), 2, 
		                                   "Incorrect coordinate dimensions." )
		self.assertEqual( len(output[0]), 2, 
		                                    "Incorrect number of dimensions." )
		self.assertEqual( len(output[0]), len(output[1]),
										   "Mis-matched x and y coordinates." )
		self.assertTrue( all(output[0] == output[1]), "Incorrect projection." )
		
class test_integrate(starcamtests):
    
    def test_empty_star_image(self):
        
        # Generate empty image
        image = self.starcam.integrate(0)
        self.assertEqual( image.shape[0], self.starcam.resolution,
                                                    "X Resolution incorrect." )
        self.assertEqual( image.shape[1], self.starcam.resolution,
                                                    "Y Resolution incorrect." )
        self.assertEqual( np.sum(image), 0, "Image not strictly positive." )
        
    def test_nonempty_star_image(self):
    
        # Generate image
        image = self.starcam.integrate(1)
        self.assertEqual( image.shape[0], self.starcam.resolution,
                                                    "X Resolution incorrect." )
        self.assertEqual( image.shape[1], self.starcam.resolution,
                                                    "Y Resolution incorrect." )
        self.assertTrue( (image >= 0).all(), "Image not strictly positive." )
        
        
class test_setpsf(starcamtests):
    
    def test_nominal_dimensions(self):
        
        # Set PSF -- nominal case
        self.starcam.setpsf(11,1)
        self.assertEqual( self.starcam.psf.shape[0], 11, "PSF size incorrect." )
        self.assertEqual( self.starcam.psf.shape[1], 11, "PSF size incorrect." )
        self.assertEqual( np.sum(self.starcam.psf), 1, "PSF sum incorrect." )
        
    def test_off_nominal_dimensions(self):
        
        # Set PSF -- nominal case
        self.starcam.setpsf(10,1)
        self.assertEqual( self.starcam.psf.shape[0], 11, "PSF size incorrect." )
        self.assertEqual( self.starcam.psf.shape[1], 11, "PSF size incorrect." )
        self.assertEqual( np.sum(self.starcam.psf), 1, "PSF sum incorrect." )
        
        
class test_set(starcamtests):
    
    def test_identical_set(self):
        
        # Store current values
        f = self.starcam.focal_len
        s = self.starcam.pixel_size
        r = self.starcam.resolution
		
        # Update values
        self.starcam.set(focal_len = f, resolution = r, pixel_size = s)
        
        # Check result
        self.assertEqual( f, self.starcam.focal_len, "Focal length not preserved." )
        self.assertEqual( s, self.starcam.pixel_size, "Pixel size not preserved."   )
        self.assertEqual( r, self.starcam.resolution, "Resolution not preserved."   )
        self.assertTrue( type(self.starcam.resolution) is int, 
                                              "Resolution not integer-valued" )
		
    def test_fov_set(self):
		
		# Test values
        f   = self.starcam.focal_len
        s   = self.starcam.pixel_size
        fov = 10.0679286799
        
        # Update
        self.starcam.set(focal_len = f, pixel_size = s, fov = fov)
        
        # Check result
        self.assertEqual( f, self.starcam.focal_len, "Focal length not preserved." )
        self.assertEqual( s, self.starcam.pixel_size, "Pixel size not preserved." )
        self.assertTrue( np.isclose(self.starcam.resolution, 1024) )
        self.assertTrue( type(self.starcam.resolution) is int, 
                                              "Resolution not integer-valued" )

    def test_focal_len_set(self):
		
		# Test values
        r   = 1024
        s   = self.starcam.pixel_size
        fov = 10.0679286799
        
        # Update
        self.starcam.set(resolution = r, pixel_size = s, fov = fov)
        
        # Check result
        self.assertEqual( r, self.starcam.resolution, "Focal length not preserved." )
        self.assertEqual( s, self.starcam.pixel_size, "Pixel size not preserved." )
        self.assertTrue( np.isclose(self.starcam.focal_len, 93), 
                                              "Incorrect focal length calculated." )