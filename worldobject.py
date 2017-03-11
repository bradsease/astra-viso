import numpy as np
from scipy.integrate import ode

class worldobject:
    
    def __init__(self, name=None):
        
        # Time
        self.epoch           = 0           # Currently unused
        self.epoch_format    = "seconds"   # Currently unused
        
        # Attitude
        self.model_pointing  = "on"
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
    
        # Verify input mode
        if ( mode.lower() not in ["ode", "explicit"] ):
            raise ValueError("Invalid pointing mode:" + mode)
        
        # Handle ODE option
        if ( mode.lower() == "ode" ):
            
            # Store ODE for later
            self.pointing_ode = fcn
            
            # Set up integrator and store ode
            explicit_fcn      = ode(fcn)
            explicit_fcn.set_integrator(self.integrator)
            explicit_fcn.set_initial_value(
                            np.hstack((self.quaternion, self.angular_rate)), 0)
            
            # Set pointing function
            pointing_fcn = lambda t : explicit_fcn.integrate(t)
        
        # Handle explicit option
        elif ( mode.lower() == "explicit" ):
            
            # Set function
            pointing_fcn = fcn
            
        # Verify input function
        # Integrates over a short span because setting to zero yields 
        # "too small step size" warning..
        test_result = pointing_fcn(1e-6)
        if ( len(test_result) != 7 ):
            raise ValueError("Invalid pointing function output length.")
        if ( type(test_result) is not np.ndarray ):
            raise TypeError("Invalid pointing function output type.")
            
        # Set internal pointing function and properties
        self.model_pointing = "on"
        self.pointing_mode  = mode.lower()
        self.pointing_fcn   = pointing_fcn
            
        
    def set_pointing_preset(self, preset):
    
        # To be implemented...
        pass
        
    def set_pointing(self, q, w=np.array([0,0,0]), t=0):
    
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
        
    def set_integrator(self, integrator, atol=None, rtol=None):
    
        # Check integrator input
        accepted_values = ["vode", "isoda", "dopri5", "dop853"]
        if ( integrator not in accepted_values ):
            raise ValueError("Unsupported integrator. Valid options: " 
                                                        + str(accepted_values))
            
        # Set integrator
        self.integrator = integrator
        
        # Set absolute tolerance
        if ( atol is not None ):
            if ( atol <= 0 ):
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.integrator_atol = atol
                
        # Set relative tolerance
        if ( rtol is not None ):
            if ( rtol <= 0 ):
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.integrator_rtol = rtol
        
        # Update pointing function if necessary
        if ( self.pointing_ode is not None ):
            self.set_pointing_fcn( self.pointing_ode, "ode" )
        
        # Update position function if necessary
        if ( self.position_ode is not None ):
            self.set_position_fcn( self.position_ode, "ode" )
