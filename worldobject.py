import numpy as np
from scipy.integrate import ode

class worldobject:
    
    def __init__(self, name=None):
        
        # Time
        self.epoch           = 0           # Currently unused
        self.epoch_format    = "seconds"   # Currently unused
        
        # Attitude
        self.model_attitude  = "on"
        self.quaternion      = np.array([0,0,0,1])
        self.angular_rate    = np.array([0,0,0])
        
        # Attitude dynamics
        self.pointing_mode   = "ode"       # "ode", "explicit", or "sampled" 
        self.pointing_ode    = None        # Storage for ode, if necessary
        self.pointing_fcn    = None        # Pointing dynamics
        
        # Position
        self.model_position  = "off"
        self.position        = np.array([0,0,0])
        self.velocity        = np.array([0,0,0])
        
        # Position dynamics
        self.position_mode   = "ode"       # "ode", "explicit", or "sampled" 
        self.position_ode    = None        # Storage for ode, if necessary
        self.position_fcn    = None        # Position dynamics
        
        # Integrator properties
        self.integrator      = "dopri5"
        self.integrator_atol = 1e-6        # Currently unused
        self.integrator_rtol = 1e-6        # Currently unused
        
        # Interpolation properties
        self.interp_order    = 5           # Currently unused
        
        
    def set_pointing_fcn(self, fcn, mode):
    
        # Verify input
        #test_result = fcn(0, np.array([0,0,0,0,0,0,0]))
        #if ( len(test_result) != 7 ):
        #    raise ValueError("Invalid pointing function output length.")
        #if ( type(test_result) is not np.ndarray ):
        #    raise TypeError("Invalid pointing function output type.")
        if ( mode.lower() not in ["ode", "explicit"] ):
            raise ValueError("Invalid pointing mode:" + mode)
    
        # Update attitude modeling mode
        self.model_attitude = "on"
        
        # Set pointing mode
        self.pointing_mode = mode.lower()
        
        # Handle ODE option
        if ( self.pointing_mode == "ode" ):
            
            # Store ODE for later
            self.pointing_ode = fcn
            
            # Set up integrator and store
            explicit_fcn      = ode(fcn)
            explicit_fcn.set_integrator(self.integrator, verbosity=-1)
            explicit_fcn.set_initial_value(
                            np.hstack((self.quaternion, self.angular_rate)), 0)
            self.pointing_fcn = lambda t : explicit_fcn.integrate(t)
        
        # Handle explicit option
        elif ( self.pointing_mode == "explicit" ):
            
            # Set function
            self.pointing_fcn = fcn
        
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