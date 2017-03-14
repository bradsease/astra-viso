import numpy as np

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