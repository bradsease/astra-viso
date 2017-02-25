import unittest
import copy
import worldobject as obj
import numpy as np

class worldobjecttests(unittest.TestCase):
    
    def setUp(self):
        self.worldobject = obj.worldobject()
        
    def tearDown(self):
        del self.worldobject
