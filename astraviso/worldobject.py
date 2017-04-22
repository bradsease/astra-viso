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
        self.pointing_fcn = None

        # Position dynamics
        self.position_fcn = None

        # Visible intensity function
        # f(t, position, observer position)
        self.vismag_fcn = None

    def set_pointing_fcn(self, fcn, mode, initial_state=None, integrator="dopri5", **ode_args):
        """
        Set internal pointing dynamics.
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

    def set_position_fcn(self, fcn, mode):
        """
        Set internal position dynamics.
        """

        # To be implemented...
        raise NotImplementedError("Method not yet implemented!")

    def get_position(self, time):
        """
        Get position at a particular time.
        """

        # To be implemented...
        raise NotImplementedError("Method not yet implemented!")

    def set_position_preset(self, preset):
        """
        Set internal position dynamics to preset function.
        """

        # To be implemented
        raise NotImplementedError("Method not yet implemented!")

    def set_vismag_fcn(self, fcn, mode):
        """
        Set internal visual magnitude function.
        """

        # To be implemented...
        raise NotImplementedError("Method not yet implemented!")

    def set_vismag_preset(self, preset):
        """
        Set internal position dynamics to preset function.
        """

        # To be implemented
        raise NotImplementedError("Method not yet implemented!")

    def get_vismag(self, time, observer_position):
        """
        Get visible magnitude at a particular time.
        """

        # To be implemented...
        raise NotImplementedError("Method not yet implemented!")
