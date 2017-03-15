import numpy as np

def rigid_body_kinematic(quaternion, angular_rate):

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
    return np.hstack((np.dot(M, quaternion)/2, np.zeros(3)))

def quaternion2dcm(quaternion):

    # Check input, must be array or list of arrays
    input_type = type(quaternion)
    if   ( input_type is np.ndarray ):
        quaternion = [quaternion]
    elif ( input_type is not list ):
        raise ValueError("Input type must be numpy ndarray or list of arrays.")
        
    # Initalize output list
    dcm_list = []
    
    # Iterate through quaternions
    for q in quaternion:
        
        # Check current quaternion
        if ( len(q) != 4 ):
            raise ValueError("Quaternion input dimension invalid.")
        q = q / np.linalg.norm(q)
        
        # Build DCM
        # Convention: intertial >> body frame
        dcm = np.zeros((3,3))
        dcm[0,0] = 1 - 2 * (q[1]**2) - 2 * (q[2]**2)
        dcm[0,1] = 2 * q[0] * q[1] - 2 * q[3] * q[2]
        dcm[0,2] = 2 * q[0] * q[2] + 2 * q[3] * q[1]
        dcm[1,0] = 2 * q[0] * q[1] + 2 * q[3] * q[2]
        dcm[1,1] = 1 - 2 * (q[0]**2) - 2 * (q[2]**2)
        dcm[1,2] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
        dcm[2,0] = 2 * q[0] * q[2] - 2 * q[3] * q[1]
        dcm[2,1] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
        dcm[2,2] = 1 - 2 * (q[0]**2) - 2 * (q[1]**2)
        
        # Add DCM to list
        dcm_list.append(dcm)

    # Return result with type matching input (np.ndarray or list)
    if ( input_type is np.ndarray ):
        return dcm_list[0]
    else:
        return dcm_list