"""
Position utilities for astra-viso.
"""
from __future__ import division
import numpy as np

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
    speed_of_light = 299792458
    delta = float('inf')
    if guess > 0:
        travel_time = guess
    else:
        travel_time = np.linalg.norm(target(time)-observer(time))/speed_of_light

    # Iterate
    while delta > atol:

        # Store current travel time
        delta = travel_time

        # Compute travel time at light-corrected time
        travel_time = np.linalg.norm(target(time-travel_time)    \
                      - observer(time-travel_time))/speed_of_light

        # Compute delta
        delta = np.abs(delta - travel_time)

    # Return result
    return travel_time
