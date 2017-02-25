import unittest
import copy
import starcam as cam
import numpy as np

class starcamtests(unittest.TestCase):
    
    def setUp(self):
        self.starcam = cam.starcam()
        
    def tearDown(self):
        del self.starcam
        
class default_tests(starcamtests):
    def test_focal_length(self):
        self.assertEqual( self.starcam.f, 93 )

class integ_tests(starcamtests):
    
    def test_empty_star_image(self):
        
        # Generate empty image
        image = self.starcam.integ(0)
        self.assertEqual( image.shape[0], self.starcam.r )
        self.assertEqual( image.shape[1], self.starcam.r )
        self.assertEqual( np.sum(image), 0 )
        
    def test_nonempty_star_image(self):
    
        # Generate image
        image = self.starcam.integ(1)
        self.assertEqual( image.shape[0], self.starcam.r )
        self.assertEqual( image.shape[1], self.starcam.r )
        self.assertTrue( (image >= 0).all() )
        
        
class setpsf_tests(starcamtests):
    
    def test_nominal_dimensions(self):
        
        # Set PSF -- nominal case
        self.starcam.setpsf(11,1)
        self.assertEqual( self.starcam.psf.shape[0], 11 )
        self.assertEqual( self.starcam.psf.shape[1], 11 )
        self.assertEqual( np.sum(self.starcam.psf), 1 )
        
    def test_off_nominal_dimensions(self):
        
        # Set PSF -- nominal case
        self.starcam.setpsf(10,1)
        self.assertEqual( self.starcam.psf.shape[0], 11 )
        self.assertEqual( self.starcam.psf.shape[1], 11 )
        self.assertEqual( np.sum(self.starcam.psf), 1 )
        
        
class set_tests(starcamtests):
    
    def test_identical_set(self):
        
        # Store current values
        f = self.starcam.f
        s = self.starcam.s
        r = self.starcam.r
		
        # Update values
        self.starcam.set(f = f, r = r, s = s)
        
        # Check result
        self.assertEqual( f, self.starcam.f )
        self.assertEqual( s, self.starcam.s )
        self.assertEqual( r, self.starcam.r )
        self.assertTrue( type(self.starcam.r) is int )
		
    def test_fov_set(self):
		
		# Test values
        f   = self.starcam.f
        s   = self.starcam.s
        fov = 10.0679286799
        
        # Update
        self.starcam.set(f = f, s = s, fov = fov)
        
        # Check result
        self.assertEqual( f, self.starcam.f )
        self.assertEqual( s, self.starcam.s )
        self.assertTrue( np.isclose(self.starcam.r, 1024) )
        self.assertTrue( type(self.starcam.r) is int )