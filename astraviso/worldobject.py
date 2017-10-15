"""
Astra-Viso world object module.
"""
from __future__ import division
import numpy as np
from scipy.integrate import ode
from astraviso import pointingutils
from astraviso import positionutils
from astraviso import ephemeris

class WorldObject:
    """
    World object class.
    """

    def __init__(self, name="Object1"):
        """
        WorldObject class initialization.

        Parameters
        ----------
        name : str, optional
            User-defined object name.

        Returns
        -------
        object : WorldObject()
            Default WorldObject instance.

        Examples
        --------
        >>> obj = WorldObject()
        """

        # Allocate settings variable
        self.__settings = {}

        # Name
        self.__settings["name"] = name                       # Object name

        # Time
        self.__settings["epoch"] = "1 Jan 2017 00:00:00.000"
        #self.__settings["epoch_format"] = "seconds"         # Type of epoch

        # Attitude dynamics
        self.pointing_fcn = None
        self.set_pointing_preset("kinematic", initial_quaternion=np.array([0, 0, 0, 1]),           \
                                                           initial_angular_rate=np.array([0, 0, 0]))

        # Position dynamics
        self.position_fcn = None
        self.set_position_preset("kinematic", initial_position=np.array([0, 0, 1]),                \
                                                               initial_velocity=np.array([0, 0, 0]))

        # Visible intensity function
        self.vismag_fcn = None
        self.set_vismag_preset("constant", vismag=-1)

    def attach_to(self, target, rel_quaternion, rel_position):
        """
        Set internal pointing and position dynamics by attaching this object to
        a secondary object.

        Parameters
        ----------
        target : WorldObject
            Target object from which to derive pointing and position dynamics.
        rel_quaternion : ndarray
            Relative orientation quaternion. Defined as the orientation of this
            object (self) in the body-fixed frame of the target object.
        rel_position : ndarray
            Relative position vector. Defined as the position of this object
            (self) in the body-fixed frame of the target object.

        Returns
        -------
        None
        """

        # Check if target object is valid
        if target.position_fcn is None or target.pointing_fcn is None:
            raise ValueError("Target must have defined pointing and position functions.")

        # Set relative pointing dynamics
        def pointing_fcn(time):
            return pointingutils.qrotate(target.get_pointing(time), rel_quaternion)
        self.set_pointing_fcn(pointing_fcn, mode="explicit")

        # Set relative position dynamics
        def position_fcn(time):
            return target.get_position(time) +                              \
                   np.dot(target.get_pointing(time, mode="dcm"), rel_position)
        self.set_position_fcn(position_fcn, mode="explicit")

    def set_pointing_fcn(self, fcn, mode, initial_state=None, integrator="dopri5", **ode_args):
        """
        Set internal pointing dynamics. Accepts both ODEs and explicit
        functions of time. Any number of states are allowed as long as the
        first four elements correspond to the quaternion attitude
        parameterization. The scalar component of the quaternion should be the
        fourth element.

        Parameters
        ----------
        fcn : function
            Input pointing function.
        mode : str
            String descripting the type of function input. Options are "ode"
            and "explicit".
        initial_state : ndarray, optional
            Optional array describing the initial pointing state. Required for
            "ode" mode only.
        integrator : str, optional
            Integrator to use for "ode" mode. Default is "dopri5". See
            documentation for scipy.integrate.ode for valid settings.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_pointing_preset, WorldObject.get_pointing

        Notes
        -----
        For "ode" mode, any keyword arguments after "integrator" will pass
        directly into the ode.set_integrator. See scipy.integrate.ode for
        valid settings.

        Examples
        --------
        >>> obj = WorldObject()
        >>> fcn = lambda t, state: [0, 0, 0, 0, 0, 0, 0]
        >>> obj.set_pointing_fcn(fcn, "ode", np.array([0, 0, 0, 1, 0, 0, 0]))
        >>> obj.get_pointing(1)
        array([ 0.,  0.,  0.,  1.])
        """

        # Verify input mode
        if mode.lower() not in ["ode", "explicit"]:
            raise ValueError("Invalid pointing mode:" + mode)

        # Handle ODE option
        if mode.lower() == "ode":

            # Define defaults, import user settings
            kwargs = {"atol" : 1e-9,           \
                      "rtol" : 1e-9,           \
                      "max_step" : 1e-3,       \
                      "nsteps" : 1e8}
            kwargs.update(ode_args)

            # Set up integrator and store ode
            ode_fcn = ode(fcn)
            ode_fcn.set_integrator(integrator, **kwargs)
            ode_fcn.set_initial_value(initial_state, 0)
            pointing_fcn = ode_fcn.integrate

        # Handle explicit option
        elif mode.lower() == "explicit":

            # Set function
            pointing_fcn = fcn

        # Set internal pointing function
        self.pointing_fcn = pointing_fcn

    def set_pointing_preset(self, preset, **kwargs):
        """
        Set internal pointing dynamics to pre-defined attitude function.
        Current options are:

        "static"    -- static pointing direction
        "kinematic" -- rigidy-body kinematic motion with a constant angular
                       rate.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        initial_state : ndarray, optional
            Optional 7-element array describing the initial pointing state.
            Required as keyword argument for the "kinematic" preset.
        initial_quaternion : ndarray, optional
            Array (4 elements) describing the initial quaternion. Required as
            a keyword argument for the "static" and "kinematic" presets.
        initial_angular_rate : ndarray, optional
            Array (3 elements) describing the initial angular rate of the
            object. Measured in radians per second. Required as a keyword
            argument for the "kinematic" preset.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_pointing_fcn, WorldObject.get_pointing

        Notes
        -----
        Uses default integrator values. For more fine-grained control, use
        WorldObject.set_pointing_fcn.

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_pointing_preset("kinematic",
        ...                           initial_quaternion=np.array([0, 0, 0, 1]),
        ...                            initial_angular_rate=np.array([0, 0, 0]))
        """

        # Handle static option
        if preset.lower() == "static":

            # Check for missing input
            if "initial_quaternion" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                              'initial_quaternion'")

            # Build lambda function
            function = lambda t: kwargs["initial_quaternion"]

            # Set function
            self.set_pointing_fcn(function, "explicit")

        # Rigid body kinematic option
        elif preset.lower() == "kinematic":

            # Check for missing input
            if "initial_quaternion" not in kwargs or "initial_angular_rate" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                        'initial_quaternion', initial_angular_rate")

            # Build lambda function
            function = lambda t, state: pointingutils.rigid_body_kinematic(state[0:4], state[4:])

            # Set function
            initial_state=np.hstack((kwargs["initial_quaternion"], kwargs["initial_angular_rate"]))
            self.set_pointing_fcn(function, "ode", initial_state)

        # Handle invalid preset
        else:
            raise NotImplementedError("Selected preset not supported.")

    def get_pointing(self, time, mode="quaternion"):
        """
        Get pointing parameters at a given time. Supports both quaternion and
        direction-cosine-matrix parameterizations.

        Parameters
        ----------
        time : float or ndarray
            Desired time(s) to extract pointing information.
        mode : str, optional
            Desired output parameterization. Supports "quaternion" and "dcm".
            Default is "dcm".

        Returns
        -------
        pointing : ndarray
            Array containing pointing data corresponding to each input time. For
            quaternion output, the array is Nx4 where N is the number of times
            requested. For DCM output, the array is Nx3x3.

        See Also
        --------
        WorldObject.set_pointing_fcn, WorldObject.set_pointing_preset

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_pointing_preset("kinematic", np.array([0,0,0,1,0,0,0]))
        >>> obj.get_pointing(1)
        array([ 0.,  0.,  0.,  1.])
        """

        # Ensure that time is a list
        if not isinstance(time, list) and not isinstance(time, np.ndarray):
            time = [time]
        num = len(time)

        # Collect pointing values for quaternion mode
        if mode == "quaternion":

            # Allocate
            output = np.zeros((num, 4))

            # Iterate
            for idx in range(num):
                output[idx, :] = self.pointing_fcn(time[idx])[0:4]

        # Collect pointing values for dcm mode
        elif mode == "dcm":

            # Allocate
            output = np.zeros((num, 3, 3))

            # Iterate
            for idx in range(num):
                output[idx, :, :] = pointingutils.quaternion2dcm(self.pointing_fcn(time[idx])[0:4])

        # Handle invalid mode
        else:
            raise NotImplementedError("Unsupported pointing type. Options: quaternion, dcm.")

        # Remove unnecessary dimension
        if num == 1:
            output = np.squeeze(output)

        # Return values
        return output

    def set_position_fcn(self, fcn, mode, initial_state=None, integrator="dopri5", **ode_args):
        """
        Set internal position dynamics. Accepts both ODEs and explicit
        functions of time. Any number of states are allowed as long as the
        first three elements correspond to the intertial position.

        Parameters
        ----------
        fcn : function
            Input position function. Function must be of the form f(t, state)
            where "state" is a vector.
        mode : str
            String descripting the type of function input. Options are "ode"
            and "explicit".
        initial_state : ndarray, optional
            Optional array describing the initial state. Required for
            "ode" mode only.
        integrator : str, optional
            Integrator to use for "ode" mode. Default is "dopri5". See
            documentation for scipy.integrate.ode for valid settings.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_position_preset, WorldObject.get_position

        Notes
        -----
        For "ode" mode, any keyword arguments after "integrator" will pass
        directly into the ode.set_integrator. See scipy.integrate.ode for
        valid settings.

        Examples
        --------
        >>> obj = WorldObject()
        >>> fcn = lambda t, state: [0, 0, 0]
        >>> obj.set_position_fcn(fcn, "ode", np.array([1, 1, 1]))
        >>> obj.get_position(1)
        array([ 1.,  1.,  1.])
        """

        # Check for valid input
        if not callable(fcn):
            raise ValueError("Must provide callable function.")

        # Handle ODE option
        if mode.lower() == "ode":

            # Define defaults, import user settings
            kwargs = {"atol" : 1e-9,           \
                      "rtol" : 1e-9,           \
                      "max_step" : 1e-3,       \
                      "nsteps" : 1e8}
            kwargs.update(ode_args)

            # Set up integrator and store ode
            ode_fcn = ode(fcn)
            ode_fcn.set_integrator(integrator, **kwargs)
            ode_fcn.set_initial_value(initial_state, 0)
            position_fcn = ode_fcn.integrate

        # Handle explicit option
        elif mode.lower() == "explicit":

            # Set function
            position_fcn = fcn

        # Handle invalid option
        else:
            raise ValueError("Invalid position mode:" + mode)

        # Set internal pointing function
        self.position_fcn = position_fcn

    def set_position_preset(self, preset, **kwargs):
        """
        Set internal position dynamics to preset function.
        Current options are:

        "static"         -- A constant position function.
        "kinematic"      -- Simple kinematic motion from an initial position and
                            velocity.
        "earth_orbit"    -- A simple Earth orbit with two-body dynamics.
        "earth_orbit_j2" -- An Earth orbit with two-body dynamics plus the J2
                            perturbation.

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        initial_position : ndarray, optional
            Initial object position. Required as keyword argument for the
            "static", "kinematic", and "orbit" presets.
        initial_velocity : ndarray, optional
            Initial object position. Required as keyword argument for the
            "kinematic" and "orbit" presets.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_position_fcn, WorldObject.get_position

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_position_preset("kinematic",
        ...                         initial_position=np.ndarray([0, 0, 0]),
        ...                         initial_velocity=np.ndarray([0, 0, 0])
        >>> obj.position_fcn(1)
        array([0, 0, 0])
        """

        # Set static option
        if preset.lower() == "static":

            # Check input
            if "initial_position" not in kwargs:
                raise ValueError("Must provide the following keyword arguments \
                                 for this preset: 'initial_position'")

            # Build function & set
            position_fcn = lambda t: kwargs["initial_position"]
            self.set_position_fcn(position_fcn, mode="explicit")

        # Set kinematic option
        elif preset.lower() == "kinematic":

            # Check input
            if "initial_position" not in kwargs or "initial_velocity" not in kwargs:
                raise ValueError("Must provide the following keyword arguments \
                                 for this preset: 'initial_position',          \
                                 'initial_velocity'")

            # Build function & set
            position_fcn = lambda t: kwargs["initial_position"]   \
                                     + kwargs["initial_velocity"]*t
            self.set_position_fcn(position_fcn, mode="explicit")

        # Set earth_orbit option
        elif preset.lower() == "earth_orbit":

            # Check input
            if "initial_position" not in kwargs or "initial_velocity" not in kwargs:
                raise ValueError("Must provide the following keyword arguments \
                                 for this preset: 'initial_position',          \
                                 'initial_velocity'")

            # Build function and set
            initial_state = np.hstack((kwargs["initial_position"],
                                       kwargs["initial_velocity"]))
            position_fcn = positionutils.earth_orbit(initial_state)
            self.set_position_fcn(position_fcn, mode="explicit")

        # Set earth_orbit option
        elif preset.lower() == "earth_orbit_j2":

            # Check input
            if "initial_position" not in kwargs or "initial_velocity" not in kwargs:
                raise ValueError("Must provide the following keyword arguments \
                                 for this preset: 'initial_position',          \
                                 'initial_velocity'")

            # Build function and set
            initial_state = np.hstack((kwargs["initial_position"],
                                       kwargs["initial_velocity"]))
            position_fcn = positionutils.earth_orbit(initial_state,
                                                     nonspherical="on")
            self.set_position_fcn(position_fcn, mode="explicit")

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def load_position_ephemeris(self, ephemeris_file):
        """
        Set internal position state to be sampled from a provided ephemeris
        file. Currently only supports STK-format ephemerides. See
        astraviso.ephemeris for more information.

        Parameters
        ----------
        ephemeris_file : str
            Path to the target ephemeris file to load.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_position_preset, WorldObject.set_position_fcn,
        WorldObject.get_position

        Notes
        -----
        In the current implementation, care should be taken to keep ephemeris
        step sizes as small as possible to avoid large interpolation error. Step
        sizes of less than 30 seconds for LEO spacecraft are recommended.
        """

        # Load ephemeris
        internal_ephem = ephemeris.OrbitEphemeris(ephemeris_file)
        self.set_position_fcn(internal_ephem.get_position, mode="explicit")

    def get_position(self, time):
        """
        Get object position at a particular time.

        Parameters
        ----------
        time : float or ndarray
            Desired time to extract position information.

        Returns
        -------
        position : ndarray
            Position vector of the WorldObject at given time in intertial space.

        See Also
        --------
        WorldObject.set_position_fcn, WorldObject.set_position_preset

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_position_preset("constant", np.array([0, 0, 0]))
        >>> obj.get_position(1)
        array([0, 0, 0])
        """

        # Compute position
        return self.position_fcn(time)

    def set_vismag_fcn(self, fcn):
        """
        Set internal object visual magnitude model.

        Parameters
        ----------
        fcn : function
            Input visual magnitude function. Output must be scalar. See notes
            for details about the required function format.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_vismag_preset, WorldObject.get_vismag

        Notes
        -----
        Function must be of the form:
                   vismag = f(t, observer_position, object_position)
        Below are two valid function definition templates.

        def user_fcn(t, observer_position, object_position):
            ...
            return vismag

        user_fcn = lambda t, observer_position, object_position: ...

        Examples
        --------
        >>> obj = WorldObject()
        >>> fcn = lambda t, *_: 7 + 2*np.sin(2*np.pi*t/30) # Ignore args after t
        >>> obj.set_vismag_fcn(fcn)
        >>> obj.vismag_fcn(0)
        7.0
        >>> obj.vismag_fcn(7.5)
        9.0
        """

        # Check for valid input
        if not callable(fcn):
            raise ValueError("Must provide callable function.")

        # Set function
        self.vismag_fcn = fcn

    def set_vismag_preset(self, preset, **kwargs):
        """
        Set internal visual magnitude model to preset function. Available
        presets are:

        "constant" -- static user-defined visual magnitude.
        "sine"     -- sinusoidal visual magnitude. Function of the form:
                      vismag + amplitude*np.sin(2*np.pi*t/frequency)

        Parameters
        ----------
        preset : str
            Name of chosen preset.
        vismag : float, optional
            Object visual magnitude. Argument required as keyword for "constant"
            and "sine" preset options.
        amplitude : float, optional
            Visual magnitude oscillation amplitude. Argument required as keyword
            for "sine" preset.
        frequency : float, optional
            Visual magnitude oscillation frequency. Measured in seconds.
            Argument required as keyword for "sine" preset.

        Returns
        -------
        None

        See Also
        --------
        WorldObject.set_vismag_fcn, WorldObject.get_vismag

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_vismag_preset("sine", vismag=7, amplitude=2, frequency=30)
        >>> obj.vismag_fcn(0)
        7.0
        >>> obj.vismag_fcn(7.5)
        9.0
        """

        # Set constant option
        if preset.lower() == "constant":

            # Check input
            if "vismag" not in kwargs:
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                                                          'vismag'")

            # Build function & set
            vismag_fcn = lambda *_: kwargs["vismag"]
            self.set_vismag_fcn(vismag_fcn)

        # Set sine option
        elif preset.lower() == "sine":

            # Check input
            if any([ele not in kwargs for ele in ["vismag", "amplitude", "frequency"]]):
                raise ValueError("Must provide the following keyword arguments for this preset:    \
                                                               'vismag', 'amplitude', 'frequency'.")

            # Build function & set
            vismag_fcn = lambda t, *_: kwargs["vismag"] +                                          \
                                           kwargs["amplitude"]*np.sin(2*np.pi*t/kwargs["frequency"])
            self.set_vismag_fcn(vismag_fcn)

        # Handle invalid option
        else:
            raise NotImplementedError("Invalid preset option.")

    def get_vismag(self, time, observer_position):
        """
        Get visible magnitude at a particular time.

        Parameters
        ----------
        time : float or ndarray
            Desired time to extract visual magnitude information.
        observer_position : ndarray
            Array describing the position of the observer in intertial space.

        Returns
        -------
        vismag : float
            Visual magnitude of the WorldObject at the given time.

        See Also
        --------
        WorldObject.set_vismag_fcn, WorldObject.set_vismag_preset

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj.set_vismag_preset("sine", vismag=7, amplitude=2, frequency=30)
        >>> obj.get_vismag(0)
        7.0
        >>> obj.get_vismag(7.5)
        9.0
        """

        # Compute visual magnitude
        return self.vismag_fcn(time, observer_position, self.get_position(time))

    def relative_to(self, origin_object, time):
        """
        Compute relative position from another WorldObject to self at a
        given time.

        Parameters
        ----------
        origin_object : WorldObject
            Object to compute the relative position from.

        Returns
        -------
        relative_position : ndarray
            Array (3-elements) describing the relative position from
            origin_object to self.

        See Also
        --------
        WorldObject.set_position_fcn, WorldObject.set_position_preset,
        WorldObject.get_position

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj2 = WorldObject()
        >>> obj.relative_to(obj2, 0)
        array([0, 0, 0])
        """

        # Compute relative position
        return self.get_position(time) - origin_object.get_position(time)

    def in_frame_of(self, origin_object, time):
        """
        Compute relative position from another WorldObject to self at a
        given time in the body frame of the target object.

        Parameters
        ----------
        origin_object : WorldObject
            Object to compute the relative position from. Reference frame of
            the result is the origin_object's body frame.

        Returns
        -------
        relative_position : ndarray
            Array (3-elements) describing the relative position from
            origin_object to self.

        See Also
        --------
        WorldObject.set_position_fcn, WorldObject.set_position_preset,
        WorldObject.get_position

        Examples
        --------
        >>> obj = WorldObject()
        >>> obj2 = WorldObject()
        >>> obj.in_frame_of(obj2)
        array([0, 0, 0])
        """

        # Compute relative position
        rel_pos = self.get_position(time) - origin_object.get_position(time)

        # Rotate into body frame
        return np.dot(rel_pos, origin_object.get_pointing(time, mode="dcm"))

    def _estimate_instantaneous_angular_rate(self, time):
        """
        Estimate the angular rate of the sensor at an instant in time. In
        general, a WorldObject instance does not have angular rate information.
        This function computes the angular rate from the internal pointing
        trajectory.

        Parameters
        ----------
        time : float
            Desired time of the angular rate measurement.

        Returns
        -------
        angular_rate : float
            Observed angular rate in radians per second.

        Notes
        -----
        Since the pointing function may be defined in a piecewise fashion, this
        function will attempt multiple approaches to compute the derivative. If
        a central difference fails, this function will attempt a forward or
        backward finite difference.
        """

        #
        delta = 1e-4
        invalid_points = 0

        # Compute first point, if possible
        try:
            initial_pointing = self.get_pointing(time-delta/2, "dcm")
        except:
            initial_pointing = self.get_pointing(time, "dcm")
            invalid_points += 1

        # Compute last point, if possible
        try:
            final_pointing = self.get_pointing(time+delta/2, "dcm")
        except:
            final_pointing = self.get_pointing(time, "dcm")
            invalid_points += 1

        # Switch to forward or backward finite difference, if possible
        if invalid_points == 1:
            delta = delta/2
        elif invalid_points > 1:
            raise AttributeError("Pointing function is undefined at time {} "+
                                 "and {}. Failed to compute derivative for "+
                                 "time: {}".format(time-delta/2, time+delta/2,
                                 time))

        # Compute result
        return pointingutils.angle_between_dcm(final_pointing,
                                               initial_pointing) / delta

    def _estimate_total_angular_displacement(self, start_time,
                                                 exposure_time,
                                                 precision="low"):
        """
        Estimate the total angular displacement of the boresight over a
        specified exposure time. Requires internal pointing model.

        Parameters
        ----------
        start_time : float
            Time to begin sequence, measured in seconds from the initial epoch.
        exposure_time : float
            Duration of each exposure, measured in seconds.
        precision : str
            Desired angle calculation fidelity. Options are "low", "medium", and
            "high". Default is "high".

        Returns
        -------
        total_angle : float
            Total displacement angle over the exposure time, in radians.

        Notes
        -----
        The three precision modes are structured as follows:

            Low: Compute the angular displacement to the first order.
            Medium: Divide the time span into 100 first-order estimates.
            High: Estimate the instantaneous derivative and integrate.

        High precision mode derives an instantaneous angular rate through a
        central difference calculation. If the computation fails for any reason,
        this function will attempt to fall back to medium precision.
        """
        if self.pointing_fcn is None:
            raise ValueError("No internal pointing model provided. Could not "+
                             "compute displacement angle.")

        # Low precision mode
        if precision.lower() == "low":
            initial_pointing = self.get_pointing(start_time, "dcm")
            final_pointing = self.get_pointing(start_time+exposure_time, "dcm")
            angle = pointingutils.angle_between_dcm(initial_pointing,
                                                    final_pointing)

        # Medium precision mode
        elif precision.lower() == "medium":
            angle = 0
            time_step = exposure_time/100

            for step in range(100):
                current_time = start_time + (step+1) * exposure_time/100
                angle += self._estimate_instantaneous_angular_rate(
                         current_time) * (time_step)

        # High precision mode
        elif precision.lower() == "high":
            try:
                angle = scipy.integrate.quad(
                        self._estimate_instantaneous_angular_rate,
                        start_time, start_time+exposure_time)[0]
            except:
                angle = self._estimate_total_angular_displacement(start_time,
                        exposure_time, precision="low")

        # Unsupported mode
        else:
            raise ValueError("Unsupported precision mode.")

        # Return result
        return abs(angle)
