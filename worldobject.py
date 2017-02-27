import numpy as np

class worldobject:
    
    def __init__(self, name=None):
        
        # Attitude
        self.model_attitude = "on"
        self.quaternion     = np.array([0,0,0,1])
        self.angular_rate   = np.array([0,0,0])
        
        # Attitude dynamics
        self.pointing_mode  = "ode"  # "ode", "explicit", or "sampled" 
        self.pointing_fcn   = None   # Pointing dynamics
        
        # Position
        self.model_position = "off"
        self.position       = np.array([0,0,0])
        self.velocity       = np.array([0,0,0])
        
        # Position dynamics
        self.position_mode  = "ode"  # "ode", "explicit", or "sampled" 
        self.position_fcn   = None   # Position dynamics
        
        
    def set_pointing_fcn(self, fcn, mode):
    
        # To be implemented...
        pass
        
    def set_pointing_preset(self, preset):
    
        # To be implemented...
        pass
        
    def get_pointing(self, t):
    
        # To be implemented...
        pass
        
    def pointing_preset_kinematic(self, quaternion, angular_rate):
    
        # To be implemented
        pass
        
        
    def set_position_fcn(self, fcn, mode):
    
        # To be implemented...
        pass
        
    def get_position(self, t):
        
        # To be implemented...
        pass
        
    def set_position_preset(self, preset):
    
        # To be implemented
        pass
        
    def position_preset_kinematic(self, position, velocity, time):
    
        # To be implemented...
        pass