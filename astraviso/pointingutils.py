"""
Pointing utilities for astra-viso.
"""
from __future__ import division
import numpy as np

def rigid_body_kinematic(quaternion, angular_rate):
    """
    Rigid body kinematic ODE.

    Parameters
    ----------
    quaternion : ndarray
        Input quaternion vector (length 4). Scalar component is designated as
        the last element.
    angular_rate : ndarray
        Input angular rate vector (length 3).

    Returns
    -------
    state_deriv : ndarray
        Derivative with respect to time of the combined state. Quaternion
        occupies first 4 elements of the state followed by the angular rate.

    Notes
    -----
    Uses the quaternion convention where quaternion[3] is the scalar component.

    Examples
    --------
    >>> quaternion = np.array([0, 0, 0, 1])
    >>> angular_rate = np.array([0.1, 0.1, 0.1])
    >>> rigid_body_kinematic(quaternion, angular_rate)
    array([ 0.05,  0.05,  0.05,  0.  ,  0.  ,  0.  ,  0.  ])
    """

    # Normalize quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)

    # Set up angular rate matrix
    matrix = np.zeros((4, 4))
    matrix[0, 1] = angular_rate[2]
    matrix[1, 0] = -angular_rate[2]
    matrix[0, 2] = -angular_rate[1]
    matrix[2, 0] = angular_rate[1]
    matrix[1, 2] = angular_rate[0]
    matrix[2, 1] = -angular_rate[0]
    matrix[0:3, 3] = angular_rate
    matrix[3, 0:3] = -angular_rate

    # Return quaternion rate
    return np.hstack((np.dot(matrix, quaternion)/2, np.zeros(3)))

def quaternion2dcm(quaternion_list):
    """
    Convert quaternions to direction cosine matrices (DCMs).

    Parameters
    ----------
    quaternion_list : list
        Input quaternion list where each element is a quaternion in an
        ndarray. Also accepts a single ndarray as input.

    Returns
    -------
    dcm_list : list
        Direction cosine matrices (DCMs) corresponding to input quaternions.
        Each DCM is a 3x3 ndarray.
    """

    # Check input, must be array or list of arrays
    input_type = type(quaternion_list)
    if input_type is np.ndarray:
        quaternion_list = [quaternion_list]
    elif input_type is not list:
        raise ValueError("Input type must be numpy ndarray or list of arrays.")

    # Initalize output list
    dcm_list = []

    # Iterate through quaternions
    for quat in quaternion_list:

        # Check current quaternion
        if len(quat) != 4:
            raise ValueError("Quaternion input dimension invalid.")
        quat = quat / np.linalg.norm(quat)

        # Build DCM
        # Convention: intertial >> body frame
        dcm = np.zeros((3, 3))
        dcm[0, 0] = 1 - 2 * (quat[1]**2) - 2 * (quat[2]**2)
        dcm[0, 1] = 2 * quat[0] * quat[1] - 2 * quat[3] * quat[2]
        dcm[0, 2] = 2 * quat[0] * quat[2] + 2 * quat[3] * quat[1]
        dcm[1, 0] = 2 * quat[0] * quat[1] + 2 * quat[3] * quat[2]
        dcm[1, 1] = 1 - 2 * (quat[0]**2) - 2 * (quat[2]**2)
        dcm[1, 2] = 2 * quat[1] * quat[2] - 2 * quat[3] * quat[0]
        dcm[2, 0] = 2 * quat[0] * quat[2] - 2 * quat[3] * quat[1]
        dcm[2, 1] = 2 * quat[1] * quat[2] + 2 * quat[3] * quat[0]
        dcm[2, 2] = 1 - 2 * (quat[0]**2) - 2 * (quat[1]**2)

        # Add DCM to list
        dcm_list.append(dcm)

    # Return result with type matching input (np.ndarray or list)
    if input_type is np.ndarray:
        return dcm_list[0]
    else:
        return dcm_list
