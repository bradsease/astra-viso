"""
Pointing utilities for astra-viso.
"""
from __future__ import division
from astraviso import mathutils
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

def rigid_body_track(observer, target):
    """
    Rigid body tracking dynamics.

    Parameters
    ----------
    observer : WorldObject
        Observing object. Must have position dynamics enabled.
    target : WorldObject
        Object to track. Must have position dynamics enabled.

    Returns
    -------
    state_fcn : ndarray
        Attitude history function for the observer. Defined as an explicit
        function of time.

    Notes
    -----
    Uses the quaternion convention where quaternion[3] is the scalar component.

    Examples
    --------
    >>> cam = StarCam()
    >>> obj = WorldObject()
    >>> state_fcn = rigid_body_track(cam, obj)
    >>> state_fcn(0)
    array([ 0.85090352,  0.        ,  0.        ,  0.52532199])
    """

    def state_fcn(time):
        """ Tracking trajectory function. """
        ra, dec = vector_to_ra_dec(target.get_position(time)-
                                   observer.get_position(time), output="rad")

        dcm = mathutils.dot_sequence(rot3(ra), rot2(-dec), rot3(-np.pi/2),
                                     rot1(-np.pi/2))

        return dcm2quaternion(dcm)

    # Return f(t)
    return state_fcn

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

def dcm2quaternion(dcm):
    """
    Convert a direction-cosine matrix to a quaternion.

    Parameters
    ----------
    dcm : ndarray
        Direction-cosine matrix (DCM).

    Returns
    -------
    quaternion : ndarray
        Output quaternion. q[3] is the scalar component.
    """

    # Initial setup
    trace = np.trace(dcm)
    quaternion = np.zeros(4)

    # Choose method to avoid singularity
    if trace > 0:
        denom = 2*np.sqrt(trace + 1)
        quaternion[3] = denom/4
        quaternion[0] = (dcm[2, 1] - dcm[1, 2])/denom
        quaternion[1] = (dcm[0, 2] - dcm[2, 0])/denom
        quaternion[2] = (dcm[1, 0] - dcm[0, 1])/denom

    elif (dcm[0, 0] > dcm[1, 1]) and (dcm[0, 0] > dcm[2, 2]):
        denom = 2*np.sqrt(1 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2])
        quaternion[3] = (dcm[2, 1] - dcm[1, 2])/denom
        quaternion[0] = denom/4
        quaternion[1] = (dcm[0, 1] + dcm[1, 0])/denom
        quaternion[2] = (dcm[0, 2] + dcm[2, 0])/denom

    elif (dcm[1, 1] > dcm[2, 2]):
        denom = 2*np.sqrt(1 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2])
        quaternion[3] = (dcm[0, 2] - dcm[2, 0])/denom
        quaternion[0] = (dcm[0, 1] + dcm[1, 0])/denom
        quaternion[1] = denom/4
        quaternion[2] = (dcm[1, 2] + dcm[2, 1])/denom

    else:
        denom = 2*np.sqrt(1 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1])
        quaternion[3] = (dcm[1, 0] - dcm[0, 1])/denom
        quaternion[0] = (dcm[0, 2] + dcm[2, 0])/denom
        quaternion[1] = (dcm[1, 2] + dcm[2, 1])/denom
        quaternion[2] = denom/4

    # Enforce normalization and return
    return quaternion / np.linalg.norm(quaternion)

def qmultiply(q, r):
    """
    Multiply two quaternions.

    Parameters
    ----------
    q : ndarray
        First quaternion. Scalar component designated as q[3].
    r : ndarray
        Second quaternion. Scalar component designated as q[3].

    Returns
    -------
    result : ndarray
        Result of q*r.
    """

    # Compute multiplication
    result = np.zeros(4)
    result[0] = r[3]*q[0] + r[0]*q[3] - r[1]*q[2] + r[2]*q[1]
    result[1] = r[3]*q[1] + r[0]*q[2] + r[1]*q[3] - r[2]*q[0]
    result[2] = r[3]*q[2] - r[0]*q[1] + r[1]*q[0] + r[2]*q[3]
    result[3] = r[3]*q[3] - r[0]*q[0] - r[1]*q[1] - r[2]*q[2]

    # Return
    return result

def qinv(quaternion):
    """
    Quaternion inverse.

    Parameters
    ----------
    quaternion : ndarray
        Input quaternion. Scalar component designated as q[3].

    Returns
    -------
    result : ndarray
        Inverted quaternion.
    """
    return quaternion * np.array([-1, -1, -1, 1])

def qrotate(q, r):
    """
    Rotate one quaternion by another.

    Parameters
    ----------
    q : ndarray
        First quaternion. Scalar component designated as q[3].
    r : ndarray
        Second quaternion. Scalar component designated as q[3].

    Returns
    -------
    result : ndarray
        Result of rotating q by r.
    """
    return qmultiply(qmultiply(q, r), qinv(q))

def rot1(theta):
    """
    Create direction-cosine matrix (dcm) for a rotation about the x-axis.
    The rotation matrix is formed such that numpy.dot(dcm, vec) rotates vec
    counter-clockwise about the x-axis.

    Parameters
    ----------
    theta : float
        Angular displacement of desired rotation. Measured in radians.

    Returns
    -------
    dcm : ndarray
        Rotation matrix corresponding to input rotation angle.

    Examples
    --------
    >>> from ssa.attitude import rot1
    >>> rot1(1.5)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.0707372 ,  0.99749499],
           [ 0.        , -0.99749499,  0.0707372 ]])
    """

    # Store sine and cosine values
    ct = np.cos(theta)
    st = np.sin(theta)

    # Construct dcm
    return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])

def rot2(theta):
    """
    Create direction-cosine matrix (dcm) for a rotation about the y-axis.
    The rotation matrix is formed such that numpy.dot(dcm, vec) rotates vec
    counter-clockwise about the y-axis.

    Parameters
    ----------
    theta : float
        Angular displacement of desired rotation. Measured in radians.

    Returns
    -------
    dcm : ndarray
        Rotation matrix corresponding to input rotation angle.

    Examples
    --------
    >>> from ssa.attitude import rot2
    >>> rot2(1.5)
    array([[ 0.0707372 ,  0.        , -0.99749499],
           [ 0.        ,  1.        ,  0.        ],
           [ 0.99749499,  0.        ,  0.0707372 ]])
    """

    # Store sine and cosine values
    ct = np.cos(theta)
    st = np.sin(theta)

    # Construct dcm
    return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

def rot3(theta):
    """
    Create direction-cosine matrix (dcm) for a rotation about the z-axis.
    The rotation matrix is formed such that numpy.dot(dcm, vec) rotates vec
    counter-clockwise about the z-axis.

    Parameters
    ----------
    theta : float
        Angular displacement of desired rotation.  Measured in radians.

    Returns
    -------
    dcm : ndarray
        Rotation matrix corresponding to input rotation angle.

    Examples
    --------
    >>> from ssa.attitude import rot3
    >>> rot3(1.5)
    array([[ 0.0707372 ,  0.99749499,  0.        ],
           [-0.99749499,  0.0707372 ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])
    """

    # Store sine and cosine values
    ct = np.cos(theta)
    st = np.sin(theta)

    # Construct dcm
    return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])

def vector_to_ra_dec(vector, vector_rate=None, output="deg"):
    """
    Compute right ascension and declination from an input vector and optional
    vector rate.

    Parameters
    ----------
    vector : ndarray
        Input vector.
    vector_rate : ndarray, optional
        Derivative of the input vector. Only required for ra/dec derivatives
        and resolution of the singularity at zenith.
    output : str
        Output units. Options are "deg" or "rad". Default is "deg".

    Returns
    -------
    ra : float
        Right ascension.
    dec : float
        Declination.
    d_ra : float
        Right ascension derivative. Only returned with vector_rate input.
    d_dec : float
        Declination derivative. Only returned with vector_rate input.

    Notes
    -----
    Based on David Vallado's astrodynamics code for MATLAB.
    https://celestrak.com/software/vallado-sw.asp
    """

    # Compute right ascension and declination
    ra = np.arctan2(vector[1], vector[0])
    dec = np.arcsin(vector[2] / np.linalg.norm(vector))

    # Handle vector rates
    if vector_rate is not None:

        # Compute temporary values
        tmp1 = np.linalg.norm(vector[:2])
        tmp2 = - vector[1]**2 - vector[0]**2

        # Update right ascension if singular
        if np.isclose(tmp1, 0):
            ra = np.arctan2(vector_rate[1], vector_rate[0])

        # Compute right ascension derivative
        if np.isclose(tmp2, 0):
            d_ra = 0
        else:
            d_ra = (vector_rate[0]*vector[1]-vector_rate[1]*vector[0])/tmp1

        # Compute declination derivative
        if np.isclose(tmp1, 0):
            d_dec = 0
        else:
            d_dec = (vector_rate[2]-np.dot(vector, vector_rate)*np.sin(dec))/tmp1

    # Convert to correct units
    if output.lower() == "deg":

        # Convert angles
        ra = np.rad2deg(ra)
        dec = np.rad2deg(dec)

        # Convert rates
        if vector_rate is not None:
            d_ra = np.rad2deg(d_ra)
            d_dec = np.rad2deg(d_dec)

    # Choose output configuration
    if vector_rate is None:
        return ra, dec
    else:
        return ra, dec, d_ra, d_dec
