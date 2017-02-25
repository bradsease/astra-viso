import unittest
import copy
import starmap as map
import numpy as np

class starmaptests(unittest.TestCase):
    
    def setUp(self):
        self.starmap = map.starmap()
        
    def tearDown(self):
        del self.starmap
