"""
Position utilities for astra-viso.
"""
from __future__ import division
import numpy as np
import scipy.integrate
from astraviso import mathutils as math

# Constants
SPEED_OF_LIGHT_VACUUM = 299792458
EARTH_GRAV_PARAM = 3.986004418e14
EARTH_EQUATORIAL_RADIUS = 6378137
EARTH_J2 = 0.0010826267

def earth_orbit(coordinates, coord_type="cartesian", nonspherical="off",
                integrator="dopri5", **integ_opts):
    """
    Create a function, f(t), describing the position and velocity of an object
    in an Earth-bound orbit.

    Parameters
    ----------
    coordinates : ndarray
        State vector (6-elements) describing the desired oribt. Function
        accepts both cartesian and keplerian parameterizations. See kep_to_cart
        documentation for proper formatting of Keplerian elements. Angles must
        be provided in degrees. Positions and velocities in m and m/s.
    coord_type : str, optional
        Type of input coordinate. Options are "cartesian" and "keplerian".
        Default is "cartesian".
    nonspherical : str, optional
        Option to add Earth oblatness effects (J2 only). Options are "on" and
        "off". Default is "off".
    integrator : str, optional
        Integrator to use for orbit propagation. See scipy.integrate.ode
        documenation for valid options. Default is "dopri5".
    **integ_opts
        Any addition keyword arguments pass directly into the ode.set_integrator
        method. See scipy.integrate.ode for valid options.

    Returns
    -------
    orbit_fcn : function
        Cartesian position and velocity as a function of time. Call with
        state = orbit_fcn(time). State measured in m and m/s.
    """

    # Check coordinate type
    if coord_type.lower() == "keplerian":
        #cartesian_coords = kep_to_cart(coordinates, angle_input="deg")
        raise ValueError('Option not supported.')
    elif coord_type.lower() == "cartesian":
        cartesian_coords = coordinates
    else:
        raise ValueError(
            "Invalid coordinate type. Options are 'cartesian'and 'keplerian'.")

    # Integrator options
    ode_args = {"atol" : 1e-8,           \
                "rtol" : 1e-8,           \
                "max_step" : 60,         \
                "nsteps" : 1e2}
    ode_args.update(integ_opts)

    # Set up orbit function
    def orbit_ode(t, x):
        return earth_orbit_ode(x, nonspherical=nonspherical)

    # Set up integrator
    orbit_fcn = scipy.integrate.ode(orbit_ode)
    orbit_fcn.set_integrator(integrator, **ode_args)
    orbit_fcn.set_initial_value(cartesian_coords)

    # Return f(t)
    return orbit_fcn.integrate

def earth_orbit_ode(coordinates, nonspherical="off"):
    """
    Compute the Cartesian position and velocity derivatives for a given
    Earth orbiting object.

    Parameters
    ----------
    coordinates : ndarray
        Cartesian state vector (6-elements) describing the desired oribt.
    nonspherical : str, optional
        Option to add Earth oblatness effects (J2 only). Options are "on" and
        "off". Default is "off".

    Returns
    -------
    diff_eq: ndarray
        Cartesian state derivative. Uses m, m/s, and m/s**2.
    """

    # Extract position from coordinates
    position = coordinates[:3]

    # Two-body
    radius = np.linalg.norm(position)
    diff_eq = -EARTH_GRAV_PARAM*math.unit(position)/(radius**2)

    # Nonspherical
    if nonspherical == "on":

        # Compute leading coefficient
        j2_coeff = -3*EARTH_GRAV_PARAM*EARTH_J2*EARTH_EQUATORIAL_RADIUS**2/(2*radius**5)

        # Compute vector coefficient
        accel_j2 = np.zeros(3)
        accel_j2[0] = position[0]*(1-5*position[2]**2/radius**2)
        accel_j2[1] = position[1]*(1-5*position[2]**2/radius**2)
        accel_j2[2] = position[2]*(3-5*position[2]**2/radius**2)

        # Add acceleration to differential equation
        diff_eq += j2_coeff*accel_j2

    # Return position derivative
    return np.hstack((coordinates[-3:], diff_eq))

def light_time(target, observer, time, guess=0, atol=1e-9):
    """
    Compute the light travel time from a target to an observer at a given time.
    Target and observer positions provided as functions of time.

    Parameters
    ----------
    target : function
        Target position function, target(t). Measured in meters.
    observer : function
        Target position function, target(t). Measured in meters.
    time : float
        Time, in seconds, to compute the light travel time.
    guess : float
        Initial guess of the light travel time, in seconds.
    atol : float, optional
        Absolute tolerance for convergence. Iteration completes when
        norm(travel_time[i] - travel_time[i-1]) <= atol, where i is the
        iteration number. Default is 1 ns.

    Returns
    -------
    travel_time : float
        Time, in seconds, for light to travel from the target to the observer.

    Notes
    -----
    Target and observer function output may have any number of states as long
    as the first three values represent position in an inertial frame.
    """

    # Initial setup
    delta = float('inf')
    if guess > 0:
        travel_time = guess
    else:
        travel_time = np.linalg.norm(target(time) -                      \
                                     observer(time))/SPEED_OF_LIGHT_VACUUM

    # Iterate
    while delta > atol:

        # Store current travel time
        delta = travel_time

        # Compute travel time at light-corrected time
        travel_time = np.linalg.norm(target(time-travel_time)    \
                      - observer(time-travel_time))/SPEED_OF_LIGHT_VACUUM

        # Compute delta
        delta = np.abs(delta - travel_time)

    # Return result
    return travel_time
