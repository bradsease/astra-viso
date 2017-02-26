import numpy as np

class worldobject:
    
    def __init__(self, name=None):
        
        # Attitude
        self.model_attitude = "on"
        self.quaternion     = np.array([0,0,0,1])
        self.angular_rate   = np.array([0,0,0])
        
        # Attitude dynamics
        self.pointing_mode  = "ode"  # "ode", "function", or "sampled" 
        self.pointing_fcn   = None   # Pointing dynamics
        
        # Position
        self.model_position = "off"
        self.position       = np.array([0,0,0])
        self.velocity       = np.array([0,0,0])
        
        # Position dynamics
        self.position_mode  = "ode"  # "ode", "function", or "sampled" 
        self.position_fcn   = None   # Position dynamics