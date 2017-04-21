"""
Astra-Viso world object module.
"""
import numpy as np
from scipy.integrate import ode
from astraviso import pointingutils

class WorldObject:
    """
    World object class.
    """

    def __init__(self, name="Object1"):
        """
        World object initialization.
        """

        # Allocate settings variable
        self.__settings = {}

        # Name
        self.__settings["name"] = name               # Object name (optional)

        # Time
        self.epoch = 0                               # Currently unused
        self.__settings["epoch_format"] = "seconds"  # Type of epoch

        # Attitude dynamics
        self.__settings["model_pointing"] = "on"
        self.__settings["pointing_mode"] = "ode"     # "ode", "explicit", or "sampled"
        self.pointing_fcn = None                     # Pointing dynamics

        # Position dynamics
        self.__settings["model_position"] = "off"
        self.__settings["position_mode"] = "ode"     # "ode", "explicit", or "sampled"
        self.position_fcn = None                     # Position dynamics

        # Integrator properties
        self.__settings["integrator"] = "dopri5"     #
        self.__settings["integrator_atol"] = 1e-9
        self.__settings["integrator_rtol"] = 1e-9

        # Interpolation properties
        self.__settings["interpolant_order"] = 5     # Currently unused

    def set_pointing_fcn(self, fcn, mode, initial_state=None):
        """
        Set internal pointing dynamics.
        """

        # Verify input mode
        if mode.lower() not in ["ode", "explicit"]:
            raise ValueError("Invalid pointing mode:" + mode)

        # Handle ODE option
        if mode.lower() == "ode":

            # Set up integrator and store ode
            pointing_fcn = ode(fcn)
            pointing_fcn.set_integrator(self.__settings["integrator"],                             \
                   atol=self.__settings["integrator_atol"], rtol=self.__settings["integrator_rtol"],
                                                                          max_step=1e-3, nsteps=1e8)
            pointing_fcn.set_initial_value(initial_state, 0)

        # Handle explicit option
        elif mode.lower() == "explicit":

            # Set function
            pointing_fcn = fcn

        # Set internal pointing function and properties
        self.__settings["model_pointing"] = "on"
        self.__settings["pointing_mode"] = mode.lower()
        self.pointing_fcn = pointing_fcn

    def set_pointing_preset(self, preset, initial_state=None):
        """
        Set internal pointing dynamics to preset function.
        """

        # Rigid body kinematic option
        if preset == "kinematic":

            # Build lambda function
            function = lambda t, state: pointingutils.rigid_body_kinematic(state[0:4], state[4:])

            # Set function
            self.set_pointing_fcn(function, "ode", initial_state)

        else:
            raise NotImplementedError("Selected preset not supported.")

    def get_pointing(self, time, mode="quaternion"):
        """
        Get pointing direction at a particular time.
        """

        # Ensure that time is a list
        if not isinstance(time, list) and not isinstance(time, np.ndarray):
            time = [time]
        num = len(time)

        # Set up pointing function
        if self.__settings["pointing_mode"] == "ode":
            pointing_fcn = self.pointing_fcn.integrate
        elif self.__settings["pointing_mode"] == "explicit":
            pointing_fcn = self.pointing_fcn
        else:
            raise NotImplementedError("Unsupported pointing function mode.")

        # Collect pointing values for quaternion mode
        if mode == "quaternion":

            # Allocate
            output = np.zeros((num, 4))

            # Iterate
            for idx in range(num):
                output[idx, :] = pointing_fcn(time[idx])[0:4]

        # Collect pointing values for dcm mode
        elif mode == "dcm":

            # Allocate
            output = np.zeros((num, 3, 3))

            # Iterate
            for idx in range(num):
                output[idx, :, :] = pointingutils.quaternion2dcm(pointing_fcn(time[idx])[0:4])

        # Handle invalid mode
        else:
            raise NotImplementedError("Unsupported pointing type. Options: quaternion, dcm.")

        # Remove unnecessary dimension
        if num == 1:
            output = np.squeeze(output)

        # Return values
        return output

    def set_position_fcn(self, fcn, mode):
        """
        Set interal position dynamics.
        """

        # To be implemented...
        pass

    def get_position(self, time):
        """
        Get position at a particular time.
        """

        # To be implemented...
        pass

    def set_position_preset(self, preset):
        """
        Set internal position dynamics to preset function.
        """

        # To be implemented
        pass

    def set_integrator(self, integrator, atol=None, rtol=None):
        """
        Set internal integrator.
        """

        # Check integrator input
        accepted_values = ["vode", "isoda", "dopri5", "dop853"]
        if integrator not in accepted_values:
            raise NotImplementedError("Unsupported integrator. Options: " + str(accepted_values))

        # Set integrator
        self.__settings["integrator"] = integrator

        # Set absolute tolerance
        if atol is not None:
            if atol <= 0:
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.__settings["integrator_atol"] = atol

        # Set relative tolerance
        if rtol is not None:
            if rtol <= 0:
                raise ValueError("Integrator tolerances must be >= 0.")
            else:
                self.__settings["integrator_rtol"] = rtol

        # Update pointing function if necessary
        if self.__settings["model_pointing"] == "on" and self.__settings["pointing_mode"] == "ode":
            self.pointing_fcn.set_integrator(self.__settings["integrator"],                        \
                   atol=self.__settings["integrator_atol"], rtol=self.__settings["integrator_rtol"])

        # Update position function if necessary
        if self.__settings["model_position"] == "on" and self.__settings["position_mode"] == "ode":
            self.position_fcn.set_integrator(self.__settings["integrator"],                        \
                   atol=self.__settings["integrator_atol"], rtol=self.__settings["integrator_rtol"])
