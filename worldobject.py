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
    
        # Normalize quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)
    
        # Set up angular rate matrix
        M = np.zeros((4,4))
        M[0,1]   =  angular_rate[2]
        M[1,0]   = -angular_rate[2]
        M[0,2]   = -angular_rate[1]
        M[2,0]   =  angular_rate[1]
        M[1,2]   =  angular_rate[0]
        M[2,1]   = -angular_rate[0]
        M[0:3,3] =  angular_rate
        M[3,0:3] = -angular_rate
    
        # Return quaternion rate
        return np.dot(M, quaternion)/2
        
        
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